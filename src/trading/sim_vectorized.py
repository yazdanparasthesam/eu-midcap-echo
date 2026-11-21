"""Vectorized sim1: $100 per positive prediction, hold `horizon` days, 0.2% fee per trade.

Treats each signal as an independent $100 bet (no reinvestment), records PnL
on the EXIT date (entry_date + horizon), builds an equity curve starting at
$100k, and reports standard performance metrics.
"""
import numpy as np, pandas as pd
from .metrics import summarise

def run(df_test, pred_col, horizon=20, position=100.0, fee=0.002,
        start_capital=100_000.0):
    df = df_test.sort_values(["Ticker","Date"]).reset_index(drop=True)
    p = df[pred_col].fillna(0).astype("int8").values
    # Prefer pre-computed growth_future_XXd; otherwise compute forward return directly
    ret_col = "growth_future_5d" if horizon == 5 else "growth_future_20d"
    if ret_col in df.columns:
        df["_ret"] = df[ret_col]
    else:
        df["_ret"] = df.groupby("Ticker")["Close"].transform(
            lambda x: x.shift(-horizon) / x - 1)
    mask = (p == 1) & np.isfinite(df["_ret"])
    entries = df.loc[mask, ["Date","_ret"]].copy()
    entries["pnl"] = position * entries["_ret"] - position * fee
    n = int(mask.sum())
    net = float(entries["pnl"].sum())

    # Distribute PnL to exit date (entry date + horizon)
    entries["exit_date"] = pd.to_datetime(entries["Date"]) + pd.Timedelta(days=int(horizon*7/5))  # ~calendar days
    # More accurate: use trading-day shift by grouping on ticker
    pnl_by_date = entries.assign(Date=pd.to_datetime(entries["Date"]) + pd.tseries.offsets.BDay(horizon)) \
                         .groupby("Date")["pnl"].sum()
    all_dates = pd.to_datetime(df["Date"]).drop_duplicates().sort_values()
    daily_pnl = pnl_by_date.reindex(all_dates, fill_value=0.0)
    equity = start_capital + daily_pnl.cumsum()

    # This sim bets a FLAT $100/trade (no reinvestment), so "equity" is start_cap + sum(PnL).
    # Sharpe/Sortino aren't meaningful for flat-bet sizing; only report return metrics + win proxy.
    years = max(len(equity)/252.0, 1/252.0)
    final = float(equity.iloc[-1])
    total_return = (final / start_capital) - 1.0
    cagr = (final/start_capital) ** (1/years) - 1 if final > 0 else -1.0
    # Win proxy: fraction of trades with positive PnL
    wins = int((entries["pnl"] > 0).sum())
    win_rate = wins / n if n else np.nan
    # Max drawdown on equity curve (even if flat sizing, this shows peak-to-trough PnL)
    roll_max = equity.cummax()
    maxdd = float((equity / roll_max - 1).min())
    sharpe_v = np.nan   # not meaningful for flat-bet sim
    sortino_v = np.nan

    trades_per_day = entries.groupby("Date").size()
    return dict(strategy=pred_col, n=n, net=net,
                mean_per_day=float(trades_per_day.mean()) if len(trades_per_day) else 0.0,
                q75_per_day=float(trades_per_day.quantile(0.75)) if len(trades_per_day) else 0.0,
                capital=start_capital, CAGR=cagr, Sharpe=sharpe_v, Sortino=sortino_v,
                MaxDD=maxdd, Final=final, n_trades=n, win_rate=win_rate, _equity=equity)
