"""Download Yahoo Finance daily OHLCV for the configured universe.

Saves one parquet per ticker under data/raw/yf/, then a combined file at
data/raw/yf_all.parquet. Safe to re-run: uses incremental range per ticker.
Compatible with yfinance >= 0.2.31 (which returns MultiIndex columns and
requires `actions=True` for Dividends / Stock Splits).
"""
import os, time
import pandas as pd
from ..bootstrap import ensure
ensure({"yfinance":"yfinance"})
import yfinance as yf
from ..config import DATA_RAW, log, load_universe

REQUIRED_COLS = ["Open","High","Low","Close","Adj Close","Volume"]
ACTION_COLS   = ["Dividends","Stock Splits"]

def download(tickers=None, start="2005-01-01", end=None, sleep=0.3):
    cfg = load_universe()
    if tickers is None:
        tickers = list(cfg["universe"].keys())
    out_dir = DATA_RAW / "yf"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for t in tickers:
        fpath = out_dir / f"{t.replace('.', '_')}.parquet"
        existing = None; new_start = start
        if fpath.exists():
            existing = pd.read_parquet(fpath)
            last_d = pd.to_datetime(existing["Date"]).max()
            if pd.Timestamp(end or pd.Timestamp.today().date()) <= last_d + pd.Timedelta(days=1):
                frames.append(existing); continue
            new_start = (last_d + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        log.info(f"  yf download {t} from {new_start}")
        try:
            df = yf.download(t, start=new_start, end=end, progress=False,
                             auto_adjust=False, actions=True, threads=False)
            if df is None or df.empty:
                if existing is not None: frames.append(existing)
                continue
            df = df.reset_index()
            # Flatten multi-index columns (newer yfinance)
            flat = []
            for c in df.columns:
                if isinstance(c, tuple):
                    # Drop the ticker level (second element) if it equals our ticker
                    name = c[0] if c[0] else c[1] if len(c)>1 else str(c)
                    flat.append(name)
                else:
                    flat.append(str(c))
            df.columns = flat
            df["Ticker"] = t
            # Ensure action columns exist (older yfinance or tickers w/o splits)
            for col in ACTION_COLS:
                if col not in df.columns:
                    df[col] = 0.0
            # Keep only expected columns to avoid surprises
            keep = ["Date"] + REQUIRED_COLS + ACTION_COLS + ["Ticker"]
            df = df[[c for c in keep if c in df.columns]]
            if existing is not None:
                df = pd.concat([existing, df], ignore_index=True).drop_duplicates("Date")
            df.to_parquet(fpath, index=False)
            frames.append(df)
        except Exception as e:
            log.warning(f"    failed {t}: {e}")
            if existing is not None: frames.append(existing)
        time.sleep(sleep)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(DATA_RAW / "yf_all.parquet", index=False)
    log.info(f"yf combined: {len(combined):,} rows, {combined.Ticker.nunique()} tickers")
    return combined

if __name__ == "__main__":
    download()
