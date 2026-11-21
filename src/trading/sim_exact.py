"""Exact day-by-day simulation with reinvestment.

Iterates DATE-by-DATE (not row-by-row). On each date, for each ticker with a
signal, allocate a conviction-sized slice of capital. Positions are held for
`horizon` days (or stopped out at -stop_loss) and PnL realized on close.
"""
import numpy as np, pandas as pd
from collections import defaultdict
from .metrics import summarise

def run(df_test, prob_col, threshold=0.60, horizon=20,
        init_cap=10_000.0, risk_fraction=0.10, fee=0.002,
        max_positions=5, stop_loss=-0.06):
    df = df_test.sort_values(["Date","Ticker"]).reset_index(drop=True)
    # Build a date -> ticker -> close/prob lookup
    pivot_close = df.pivot_table(index="Date", columns="Ticker", values="Close").sort_index()
    pivot_prob  = df.pivot_table(index="Date", columns="Ticker", values=prob_col).sort_index()
    dates = pivot_close.index
    tickers = pivot_close.columns
    close = pivot_close.values.astype(float)    # [ndays, ntickers]
    probs = pivot_prob.values.astype(float)
    T = len(dates); K = len(tickers)

    cash = float(init_cap)
    # positions: list of dicts {entry_day, ticker_idx, entry_price, qty}
    positions = []
    equity_curve = np.empty(T)
    n_trades = 0; wins = 0

    for d in range(T):
        prices_d = close[d]
        # 1) Close positions that hit stop or matured today (use today's price for that ticker)
        still_open = []
        for pos in positions:
            ed, ti, ep, qty = pos["entry_day"], pos["ticker_idx"], pos["entry_price"], pos["qty"]
            if not np.isfinite(prices_d[ti]):
                still_open.append(pos); continue
            ret = (prices_d[ti] / ep) - 1.0
            hit_stop = ret <= stop_loss
            matured  = (d - ed) >= horizon
            if hit_stop or matured:
                proceeds = qty * prices_d[ti]
                exit_fee = proceeds * fee
                cash += proceeds - exit_fee
                n_trades += 1
                if prices_d[ti] > ep: wins += 1
            else:
                still_open.append(pos)
        positions = still_open

        # 2) Open new positions on signals (iterate tickers with prob >= threshold)
        if len(positions) < max_positions:
            signal_mask = np.isfinite(probs[d]) & (probs[d] >= threshold)
            candidates = np.where(signal_mask)[0]
            np.random.shuffle(candidates) if False else None
            for ti in candidates:
                if len(positions) >= max_positions: break
                p = probs[d, ti]; price = prices_d[ti]
                if not np.isfinite(p) or not np.isfinite(price) or price <= 0: continue
                # Don't double-up on same ticker
                if any(pp["ticker_idx"]==ti for pp in positions): continue
                conviction = max(0.05, min(1.0, (p - threshold) / (1 - threshold + 1e-9)))
                alloc = cash * risk_fraction * conviction
                entry_fee = alloc * fee
                cost = alloc + entry_fee
                if cost > cash or alloc <= 0: continue
                qty = alloc / price
                cash -= cost
                positions.append({"entry_day": d, "ticker_idx": int(ti),
                                  "entry_price": float(price), "qty": float(qty)})

        # 3) Mark-to-market all open positions (use each position's OWN ticker price)
        mtm = cash
        for pos in positions:
            pr = prices_d[pos["ticker_idx"]]
            if np.isfinite(pr): mtm += pos["qty"] * pr
            else:               mtm += pos["qty"] * pos["entry_price"]
        equity_curve[d] = mtm

    eq = pd.Series(equity_curve, index=dates)
    summ = summarise(eq, n_trades)
    summ["n"] = n_trades  # number of signals that produced trades
    summ["net"] = float(eq.iloc[-1] - init_cap)
    summ["win_rate"] = wins / n_trades if n_trades else np.nan
    summ["capital"] = float(init_cap)
    summ["strategy"] = f"exact_{prob_col}_thr{threshold}"
    return eq, summ
