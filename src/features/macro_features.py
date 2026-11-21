"""Macro-regime features (joined by Date), plus volatility/price-location features."""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=".*DataFrameGroupBy.apply operated on the grouping columns.*")

def build(df, macro_df):
    macro_df = macro_df.copy()
    macro_df["Date"] = pd.to_datetime(macro_df["Date"])
    for c in macro_df.columns:
        if c == "Date": continue
        macro_df[c] = macro_df[c].astype("float32")
    macro_df = macro_df.sort_values("Date").ffill()
    df = df.sort_values("Date").merge(macro_df, on="Date", how="left").sort_values(["Ticker","Date"]).reset_index(drop=True)
    g = df.groupby("Ticker", group_keys=False)
    df["dist_52w_high"] = df["Close"] / g["Close"].transform(lambda x: x.rolling(252, min_periods=60).max()) - 1.0
    df["atr_pct"] = g.apply(lambda s: (s["High"]-s["Low"])/s["Close"]).reset_index(level=0, drop=True).astype("float32")
    df["volume_spike"] = (df["Volume"] / g["Volume"].transform(lambda x: x.rolling(20).mean()) - 1.0).astype("float32")
    # VSTOXX proxy via vix quantile (simple 60d quantile)
    if "vix" in df.columns:
        df["vstoxx_q"] = df["vix"].rolling(60, min_periods=20).apply(
            lambda x: (x.rank(pct=True).iloc[-1]), raw=False).astype("float32")
    else:
        df["vstoxx_q"] = 0.5
    df["dax_ma200_slope"] = np.float32(0.0)  # filled below if DAX series is present; placeholder
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    return df
