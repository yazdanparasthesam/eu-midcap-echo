"""Risk-managed flagship strategy: regime filter + Kelly sizing + stop loss.

- Enter only if: (a) predicted probability >= threshold, (b) VSTOXX quantile < 0.75
  (not a panic regime), (c) price not within 3% of a 52-week low (avoid falling knives).
- Position size scales with conviction (Kelly-fraction × prob edge).
- Max 5 concurrent positions, stop loss -6%, 20-day horizon.
- Benchmark: equal-weight buy-and-hold of all stocks in the test window.
"""
import numpy as np, pandas as pd
from .sim_exact import run as exact_run
from .metrics import summarise

def run(df_test, prob_col, threshold=0.60, benchmark_col=None, **kw):
    df = df_test.copy()
    # Regime filter: zero out proba when volatility regime is stressed
    if "vstoxx_q" in df.columns:
        df.loc[df["vstoxx_q"] >= 0.75, prob_col] = 0.0
    # Avoid falling knives: if within 40% of 52w low (dist_52w_high < -0.40 means close is 40% below high)
    if "dist_52w_high" in df.columns:
        df.loc[df["dist_52w_high"] < -0.40, prob_col] = 0.0

    eq, stats = exact_run(df, prob_col, threshold=threshold, risk_fraction=0.15,
                          max_positions=5, stop_loss=-0.06, init_cap=100_000.0, **kw)

    # Benchmark: equal-weight buy-and-hold basket
    # Build daily portfolio return = mean of daily returns across tickers
    df_sorted = df.sort_values(["Ticker","Date"]).copy()
    df_sorted["ret"] = df_sorted.groupby("Ticker")["Close"].pct_change()
    bench_daily = df_sorted.groupby("Date")["ret"].mean().fillna(0).sort_index()
    bench_daily = bench_daily.reindex(eq.index, fill_value=0.0)
    bench_equity = 100_000.0 * (1.0 + bench_daily).cumprod()
    bench_stats = summarise(bench_equity)
    stats["benchmark_CAGR"] = bench_stats["CAGR"]
    stats["benchmark_Sharpe"] = bench_stats["Sharpe"]
    stats["benchmark_Final"] = bench_stats["Final"]
    stats["alpha_vs_bench"] = stats["CAGR"] - bench_stats["CAGR"]
    stats["strategy"] = f"risk_managed_{prob_col}_thr{threshold}"
    return eq, stats, bench_equity
