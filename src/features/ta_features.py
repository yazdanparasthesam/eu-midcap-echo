"""Technical-analysis features using the `ta` library (pure Python, no TA-Lib needed).

Replicates the key groups used in lectures: momentum, trend, volume, volatility,
plus SMA10/20, growing_moving_average, high_minus_low_relative, volatility,
ln_volume, div_payout, stock_split, growth_lags.
"""
import warnings, numpy as np, pandas as pd
from ..bootstrap import ensure
ensure({"ta": "ta"})
import ta
# Silence pandas 2.x FutureWarning from ta library's groupby.apply patterns
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=".*DataFrameGroupBy.apply operated on the grouping columns.*")

def _add_growth(df, horizons=(1,2,3,5,10,20,30,60,90,180,252,365*3)):
    out = []
    for h in horizons:
        out.append(df.groupby("Ticker")["Close"].pct_change(h).rename(f"growth_{h}d"))
    g5  = df["Close"].pct_change(5).groupby(df["Ticker"]).shift(-5).rename("growth_future_5d")
    g20 = df["Close"].pct_change(20).groupby(df["Ticker"]).shift(-20).rename("growth_future_20d")
    out.append(g5)
    out.append(g20)
    out.append((g20 > 0).astype("int8").rename("is_positive_growth_20d_future"))
    out.append((g5  > 0).astype("int8").rename("is_positive_growth_5d_future"))
    return pd.concat(out, axis=1)

def build(df):
    """df must have columns: Date, Ticker, Open, High, Low, Close, Adj Close, Volume, Dividends, Stock Splits."""
    df = df.sort_values(["Ticker","Date"]).reset_index(drop=True)
    g = df.groupby("Ticker", group_keys=False)

    # Basic custom (from lectures)
    df["ln_volume"]   = np.log(df["Volume"].replace(0, np.nan).fillna(1e-9)).astype("float32")
    df["div_payout"]  = (df["Dividends"].fillna(0) > 0).astype("int8") if "Dividends" in df.columns else 0
    df["stock_split"] = (df["Stock Splits"].fillna(0) != 0).astype("int8") if "Stock Splits" in df.columns else 0

    # ta indicators (applied per group)
    def _apply_ta(sub):
        s = ta.add_all_ta_features(sub, open="Open", high="High", low="Low", close="Close",
                                  volume="Volume", fillna=True, vectorized=True)
        return s
    df = g.apply(_apply_ta).reset_index(drop=True)

    # Our own simple features (matches lecture naming)
    df["SMA10"] = g["Close"].transform(lambda x: x.rolling(10).mean())
    df["SMA20"] = g["Close"].transform(lambda x: x.rolling(20).mean())
    df["growing_moving_average"] = (df["SMA10"] > df["SMA20"]).astype("int8")
    df["high_minus_low_relative"] = (df["High"] - df["Low"]) / df["Close"]
    df["volatility"] = g["Close"].transform(lambda x: x.pct_change().rolling(20).std())

    growth = _add_growth(df)
    df = pd.concat([df, growth], axis=1)
    return df
