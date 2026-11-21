"""Combine raw yf + macro + dgap into one feature-rich parquet."""
import numpy as np, pandas as pd
from ..config import DATA_RAW, DATA_PRO, log
from .ta_features import build as build_ta
from .macro_features import build as build_macro
from .sentiment_features import build as build_sentiment

CATEGORICAL = ["Month","Weekday","ticker_type","sector","disclosure_polarity"]
TO_DROP_BASE = ["Open","High","Low","Close","Adj Close","Volume","Dividends","Stock Splits","Ticker","title","body","url","pos","neg"]

def build_dataset():
    log.info("loading raw yf_all")
    prices = pd.read_parquet(DATA_RAW / "yf_all.parquet")
    prices["Date"] = pd.to_datetime(prices["Date"])
    log.info(f"  prices: {len(prices):,} rows")

    # Sector mapping
    import yaml
    from ..config import UNIVERSE_PATH
    with open(UNIVERSE_PATH) as f:
        meta = yaml.safe_load(f)["universe"]
    sec_map = {t: v["sector"] for t,v in meta.items()}
    prices["sector"] = prices["Ticker"].map(sec_map).fillna("other")
    prices["ticker_type"] = np.where(prices["Ticker"].str.endswith(".DE"), "DE",
                            np.where(prices["Ticker"].str.endswith(".AS"), "NL",
                            np.where(prices["Ticker"].str.endswith(".PA"), "FR",
                            np.where(prices["Ticker"].str.endswith(".WA"), "PL", "OTHER"))))

    log.info("applying TA features")
    df = build_ta(prices); del prices
    log.info(f"  after TA: {df.shape}")

    macro_path = DATA_RAW / "macro_fred.parquet"
    if macro_path.exists():
        macro = pd.read_parquet(macro_path)
        df = build_macro(df, macro); del macro
        log.info(f"  after macro: {df.shape}")
    else:
        log.warning("macro_fred.parquet not found; skipping macro features")

    # News headlines — try both legacy and new filename; tolerate empty/missing
    dgap = None
    for fname in ("news_headlines.parquet", "dgap_adhocs.parquet"):
        p = DATA_RAW / fname
        if p.exists():
            try:
                tmp = pd.read_parquet(p)
                if len(tmp) > 0:
                    dgap = tmp; break
            except Exception as e:
                log.warning(f"  failed to read {fname}: {e}")
    if dgap is None:
        log.warning("news headlines not found — running without sentiment/event features")
    df = build_sentiment(df, dgap); del dgap
    log.info(f"  after sentiment: {df.shape}")

    # Dummies
    df["Month"]   = pd.to_datetime(df["Date"]).dt.strftime("%B")
    df["Weekday"] = pd.to_datetime(df["Date"]).dt.day_name()
    df["disclosure_polarity"] = np.where(df["disclosure_pos"]==1, "pos",
                                 np.where(df["disclosure_neg"]==1, "neg", "none"))
    dum = pd.get_dummies(df[CATEGORICAL], dtype="int8")
    df = pd.concat([df, dum], axis=1); df.drop(columns=CATEGORICAL, inplace=True)

    # Keep only post-2005, finite target
    df = df[df["Date"] >= "2005-01-01"].reset_index(drop=True)
    # Downcast
    for c in df.columns:
        if c == "Date" or c == "Ticker_str": continue
        if df[c].dtype == "float64": df[c] = df[c].astype("float32")
    df.to_parquet(DATA_PRO / "df_combined.parquet", index=False)
    log.info(f"saved df_combined.parquet  shape={df.shape}")
    return df, list(dum.columns)

def get_feature_lists(df, dummies):
    GROWTH = [c for c in df.columns if c.startswith("growth_") and "future" not in c]
    TA_CUSTOM = ["SMA10","SMA20","growing_moving_average","high_minus_low_relative","volatility","ln_volume",
                 "div_payout","stock_split","dist_52w_high","atr_pct","volume_spike","vstoxx_q"]
    TA_TALIB  = [c for c in df.columns if c.startswith(("momentum_","trend_","volatility_","volume_",
                                                       "others_","momentum_rsi","trend_sma","trend_ema"))]
    SENT = ["sentiment_score","events_pos_5d","events_neg_5d","events_all_10d",
            "days_since_pos_event","days_since_neg_event","disclosure_today"]
    MACRO = [c for c in ["us_10y","us_5y","us_1y","fed_funds","ted_spread","vix","usd_eur"] if c in df.columns]
    NUMERICAL = GROWTH + TA_TALIB + TA_CUSTOM + SENT + MACRO
    TO_DROP = [c for c in ["Date"] + TO_DROP_BASE if c in df.columns]
    NUMERICAL = [c for c in NUMERICAL if c in df.columns and c not in TO_DROP]
    return NUMERICAL, dummies, TO_DROP

if __name__ == "__main__":
    build_dataset()
