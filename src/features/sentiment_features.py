"""Sentiment + event-count features from per-ticker news headlines.

Uses a simple lexicon-based English + German financial sentiment (avoids the
400 MB HF model download in CI). Generates rolling event counts, days-since-
event, and a disclosure-day dummy.
"""
import numpy as np, pandas as pd
from ..config import log

POS_WORDS = {
    # English financial positive (also base forms so "beats" and "beat" both match)
    "beat","beats","beating","raise","raises","raised","surge","surges","surged","soar","soars",
    "record","growth","upgrade","upgraded","outperform","outperforms","bullish",
    "profit","strong","stronger","exceed","exceeds","exceeded","higher","better-than-expected",
    "upside","buyback","accelerate","expansion","new contract","record revenue","beat estimates",
    "positive","gains","gain","gained","jumps","jump","jumped","rallies","rally","rallied",
    "all-time high","profit beat","top expectations","tops expectations","topped",
    "undervalued","beats estimates","exceeded expectations","surpass","surpasses","surpassed",
    "boost","boosts","boosted","rally","up","recover","recovers","recovered","rebound",
    "buy rating","overweight","dividend hike","revenue beat","eps beat","strong results",
    "optimistic","upgrade to buy","raised guidance","increased guidance","share buyback",
    "best quarter","blowout","beats on revenue","beats on earnings","impressive",
    # German positive
    "steigt","erhöht","wachstum","rekord","positiv","gewinn","prognose angehoben",
    "übertrifft","erfolgreich","deutlich über","auftragseingang","dividende erhöht",
    "über den erwartungen","bestes quartal","neuer großauftrag","deutlicher anstieg",
}
NEG_WORDS = {
    # English financial negative
    "miss","misses","missed","cut","cuts","downgrade","downgraded","plunge","plunges","plunged",
    "tumble","tumbles","tumbled","loss","warn","warns","warning","bearish","underperform",
    "underperforms","decline","declines","declined","slump","slumps","slumped",
    "lower-than-expected","recall","investigation","fine","fraud","lawsuit","profit warning",
    "disappointing","disappoints","disappointed","downside","sell-off","selloff",
    "drop","drops","dropped","fall","falls","fell","weak","weaker","weakness","slows","slowdown",
    "delay","layoff","layoffs","bankruptcy","insolvency","charges","write-down","writedown",
    "misses estimates","missed estimates","missed expectations","cuts guidance","lowered guidance",
    "guidance cut","profit warning","downgrade to sell","sell rating","underweight","overvalued",
    "tumble","crash","crashes","sink","sinks","sank","dive","dives","plummet","plummets",
    "bad news","disappointing results","weak results","revenue miss","eps miss",
    "lawsuit","sec probe","antitrust","scandal","fraud","fined","penalty","settlement",
    # German negative
    "sinkt","senkt","rückgang","verlust","warnung","prognose gesenkt","unter den erwartungen",
    "negativ","strafzahlung","rückruf","verzögerung","abschreibung","kurzarbeit",
    "insolvenz","enttäuschend","gewinnwarnung","niedriger als erwartet",
}

def _lexicon_score(text):
    if not isinstance(text, str) or not text: return 0.0
    t = text.lower()
    pos = sum(w in t for w in POS_WORDS)
    neg = sum(w in t for w in NEG_WORDS)
    if pos == 0 and neg == 0: return 0.0
    return (pos - neg) / (pos + neg)   # [-1, +1]

def build(df, dgap_df):
    """Merge DGAP disclosures onto the price panel and generate event features."""
    if dgap_df is None or dgap_df.empty:
        for c in ["sentiment_score","events_pos_5d","events_neg_5d","events_all_10d",
                  "days_since_pos_event","days_since_neg_event","disclosure_today",
                  "disclosure_pos","disclosure_neg","disclosure_neu"]:
            df[c] = 0 if c != "sentiment_score" else 0.0
        return df

    dgap_df = dgap_df.copy()
    dgap_df["Date"] = pd.to_datetime(dgap_df["Date"])
    dgap_df["sentiment_score"] = dgap_df["title"].fillna("") + " " + dgap_df["body"].fillna("")
    dgap_df["sentiment_score"] = dgap_df["sentiment_score"].map(_lexicon_score).astype("float32")
    dgap_df["pos"] = (dgap_df["sentiment_score"] >  0.2).astype("int8")
    dgap_df["neg"] = (dgap_df["sentiment_score"] < -0.2).astype("int8")
    # Aggregate per (Date, Ticker): sum polarity
    agg = dgap_df.groupby(["Date","Ticker"]).agg(
        sentiment_score=("sentiment_score","mean"),
        pos=("pos","sum"), neg=("neg","sum")).reset_index()

    df = df.merge(agg, on=["Date","Ticker"], how="left")
    df["sentiment_score"] = df["sentiment_score"].fillna(0.0).astype("float32")
    df["pos"] = df["pos"].fillna(0).astype("int16")
    df["neg"] = df["neg"].fillna(0).astype("int16")
    df["disclosure_today"] = ((df["pos"] + df["neg"]) > 0).astype("int8")

    g = df.groupby("Ticker", group_keys=False)
    df["events_pos_5d"]  = g["pos"].transform(lambda x: x.rolling(5).sum())
    df["events_neg_5d"]  = g["neg"].transform(lambda x: x.rolling(5).sum())
    df["events_all_10d"] = df["disclosure_today"].groupby(df["Ticker"]).transform(
        lambda x: x.rolling(10).sum())
    # days since last pos / neg event (capped at 60)
    def _days_since(flag):
        out = np.full(len(flag), 60, dtype="int16")
        last = -10_000
        arr = flag.values if hasattr(flag, "values") else np.asarray(flag)
        for i,v in enumerate(arr):
            if v > 0: last = i
            out[i] = min(i - last, 60)
        return out
    df["days_since_pos_event"] = g["pos"].transform(_days_since).astype("int16")
    df["days_since_neg_event"] = g["neg"].transform(_days_since).astype("int16")

    df["disclosure_pos"] = (df["sentiment_score"] >  0.2).astype("int8")
    df["disclosure_neg"] = (df["sentiment_score"] < -0.2).astype("int8")
    df["disclosure_neu"] = ((df["disclosure_today"]==1) & (df["disclosure_pos"]==0) & (df["disclosure_neg"]==0)).astype("int8")
    return df
