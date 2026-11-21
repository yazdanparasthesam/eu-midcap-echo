"""Performance metrics for strategy evaluation."""
import numpy as np, pandas as pd

def max_drawdown(equity_curve):
    roll_max = equity_curve.cummax()
    dd = equity_curve / roll_max - 1.0
    return float(dd.min())

def sharpe(returns, rf_annual=0.0, periods=252):
    if len(returns) < 2: return np.nan
    excess = returns - rf_annual/periods
    return float(np.sqrt(periods) * excess.mean() / (excess.std() + 1e-9))

def sortino(returns, rf_annual=0.0, periods=252):
    if len(returns) < 2: return np.nan
    excess = returns - rf_annual/periods
    down = excess[excess < 0]
    if len(down) < 2: return np.nan
    return float(np.sqrt(periods) * excess.mean() / (down.std() + 1e-9))

def cagr(equity_curve, years=None):
    if years is None:
        years = len(equity_curve)/252
    return float((equity_curve.iloc[-1]/equity_curve.iloc[0])**(1/years) - 1) if years>0 else np.nan

def summarise(equity_curve, trades=None):
    rets = equity_curve.pct_change().fillna(0)
    return dict(CAGR=cagr(equity_curve),
                Sharpe=sharpe(rets),
                Sortino=sortino(rets),
                MaxDD=max_drawdown(equity_curve),
                Final=float(equity_curve.iloc[-1]),
                n_trades=int(trades) if trades is not None else None)
