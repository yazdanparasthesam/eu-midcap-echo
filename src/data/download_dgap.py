"""Per-ticker news headlines (alt-data event source).

Uses Google News RSS as primary source (plain XML, no API key, works globally)
with yfinance .news as a fallback. Both are free and return real headlines
tied to each ticker with timestamps and URLs.

If both sources fail (offline / blocked), returns an empty frame so the rest
of the pipeline still runs end-to-end (sentiment features default to 0).
"""
import time, random
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from ..config import DATA_RAW, log, load_universe
from ..bootstrap import ensure

CACHE_FILE = DATA_RAW / "news_headlines.parquet"

UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
]

NOISE_PATTERNS = [
    "stock price, news, quote", "stock price and chart", "stock price forecast",
    "should you buy", "price today", "real-time quote", "live quote",
    "dividend history", "interactive chart", "technical analysis gauge",
    "get the latest", "wikipedia",
]

def _from_google_news(ticker, company_name, max_items=30):
    """Fetch headlines via Google News RSS — no auth needed, returns recent articles."""
    query = f"{ticker} {company_name}" if company_name else ticker
    # when:1y = last 12 months, filters out stale decade-old articles
    url = (f"https://news.google.com/rss/search?"
           f"q={requests.utils.quote(query)}+stock+when:1y"
           f"&hl=en-US&gl=US&ceid=US:en")
    r = requests.get(url, headers={"User-Agent": random.choice(UA_POOL),
                                   "Accept":"application/rss+xml, text/xml, */*"},
                     timeout=20)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    rows = []
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=550)
    for it in root.findall('.//item'):
        if len(rows) >= max_items: break
        title = (it.findtext('title') or '').strip()
        pub = it.findtext('pubDate') or ''
        link = it.findtext('link') or ''
        if not title or not pub: continue
        # Skip generic price-quote landing pages
        low = title.lower()
        if any(p in low for p in NOISE_PATTERNS): continue
        if len(title) < 20: continue
        try:
            dt = pd.to_datetime(pub)
        except Exception:
            continue
        if dt < cutoff: continue
        rows.append(dict(Date=dt.date(), Ticker=ticker, title=title, body="", url=link))
    return rows

def _from_yfinance(ticker, max_items=20):
    """Fallback: use yfinance .news."""
    ensure({"yfinance":"yfinance"})
    import yfinance as yf
    t = yf.Ticker(ticker)
    items = t.news or []
    rows = []
    for it in items[:max_items]:
        ts = it.get("providerPublishTime"); title = it.get("title"); link = it.get("link")
        if not title and "content" in it:
            c = it["content"]
            title = c.get("title"); ts = c.get("pubDate") or c.get("providerPublishTime")
            cu = c.get("canonicalUrl")
            link = cu.get("url") if isinstance(cu, dict) else c.get("link")
        if not title or not ts: continue
        try:
            dt = pd.to_datetime(ts, unit="s").date() if isinstance(ts,(int,float)) else pd.to_datetime(ts).date()
        except Exception:
            continue
        rows.append(dict(Date=dt, Ticker=ticker, title=title, body="", url=link or ""))
    return rows

def _fetch_ticker(ticker, company_name):
    """Try Google News first, then yfinance fallback. Returns [] if both fail."""
    for attempt in range(2):
        try:
            rows = _from_google_news(ticker, company_name)
            if rows:
                return rows
        except Exception as e:
            log.debug(f"    google news {ticker} attempt {attempt+1} failed: {e}")
            time.sleep(0.5 + random.random())
    try:
        rows = _from_yfinance(ticker)
        if rows: return rows
    except Exception as e:
        log.debug(f"    yfinance {ticker} failed: {e}")
    log.warning(f"    news {ticker}: no sources returned headlines (network?)")
    return []

def download(tickers=None, force=False, limit_pages=None):
    """Download per-ticker news headlines.

    Args:
        tickers: explicit ticker list (defaults to full universe)
        force: re-download even if cache exists
        limit_pages: ignored, kept for backward compat with old notebook calls
    """
    cfg = load_universe()
    uni = cfg["universe"]
    tickers = tickers or list(uni.keys())

    existing = None
    if CACHE_FILE.exists() and not force:
        existing = pd.read_parquet(CACHE_FILE)
        log.info(f"loaded existing news cache: {len(existing):,} rows")
        have = set(existing["Ticker"].unique()) if len(existing) else set()
        tickers = [t for t in tickers if t not in have]
        log.info(f"need to fetch {len(tickers)} tickers")

    rows = []
    ok = fail = 0
    for i, t in enumerate(tickers, 1):
        name = uni.get(t, {}) if isinstance(uni.get(t), dict) else {}
        company = name.get("name", "") if isinstance(name, dict) else ""
        log.info(f"  news {t} ({i}/{len(tickers)})")
        got = _fetch_ticker(t, company)
        if got:
            rows.extend(got); ok += 1
        else:
            fail += 1
        # Polite delay between requests
        time.sleep(0.3 + random.random()*0.4)

    new = pd.DataFrame(rows, columns=["Date","Ticker","title","body","url"])
    if existing is not None and len(existing):
        new = pd.concat([existing, new], ignore_index=True).drop_duplicates(subset=["Ticker","title","url"])
    if len(new) == 0:
        log.warning("news: 0 rows returned from all sources — writing empty frame. Pipeline will continue with sentiment=0.")
        new = pd.DataFrame(columns=["Date","Ticker","title","body","url"])
    else:
        new["Date"] = pd.to_datetime(new["Date"])
    new.to_parquet(CACHE_FILE, index=False)
    log.info(f"news: {len(new):,} total rows ({ok} tickers ok, {fail} failed)")
    return new

if __name__ == "__main__":
    download()
