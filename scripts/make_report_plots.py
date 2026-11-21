"""Generate the equity-curve + rolling-Sharpe plot for the README/report.

Reads reports/equity_curve.csv and reports/benchmark_curve.csv (produced by
stage_simulate()) and writes reports/equity_curve.png used in README.md.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import src.config as cfg


def _main():
    reports = Path(cfg.REPORTS)
    eq_path = reports / "equity_curve.csv"
    bm_path = reports / "benchmark_curve.csv"
    if not eq_path.exists():
        raise SystemExit(f"missing {eq_path} — run simulate stage first")

    eq = pd.read_csv(eq_path, index_col=0, parse_dates=True).squeeze("columns")
    bm = (
        pd.read_csv(bm_path, index_col=0, parse_dates=True).squeeze("columns")
        if bm_path.exists()
        else None
    )

    # Normalize both to base=1 for comparability
    eq_n = eq / eq.iloc[0]
    if bm is not None:
        bm = bm.reindex(eq_n.index, method="ffill")
        bm_n = bm / bm.iloc[0]

    # 60-day rolling Sharpe (annualized)
    def rolling_sharpe(s, window=60):
        r = s.pct_change().dropna()
        return (r.rolling(window).mean() / r.rolling(window).std()) * np.sqrt(252)

    rs_eq = rolling_sharpe(eq_n)
    rs_bm = rolling_sharpe(bm_n) if bm is not None else None

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1.4]}
    )

    # Top: equity curves
    ax1.plot(
        eq_n.index, eq_n.values, color="#1f4cea", linewidth=1.4,
        label="Strategy (Risk-managed)",
    )
    if bm is not None:
        ax1.plot(
            bm_n.index, bm_n.values, color="#f2a93b", linewidth=1.2,
            label="Benchmark (Equal-weight EU mid-caps)",
        )
    ax1.axhline(1.0, color="grey", linewidth=0.5, linestyle="--")
    ax1.set_title(
        "Equity Curve: Strategy vs Benchmark (EU mid-caps, equal-weight)",
        fontsize=12,
    )
    ax1.set_ylabel("Equity (Base=1)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)

    # Bottom: rolling Sharpe
    ax2.plot(
        rs_eq.index, rs_eq.values, color="#1f4cea", linewidth=1.0,
        label="Strategy Sharpe (60d)",
    )
    if rs_bm is not None:
        ax2.plot(
            rs_bm.index, rs_bm.values, color="#f2a93b", linewidth=1.0,
            label="Benchmark Sharpe (60d)",
        )
    ax2.axhline(0, color="#cc2936", linewidth=0.8, linestyle="--")
    ax2.set_title("Rolling Sharpe Ratio (60 days)", fontsize=11)
    ax2.set_ylabel("Sharpe")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(-4, 5)

    plt.tight_layout()
    out = reports / "equity_curve.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    _main()

