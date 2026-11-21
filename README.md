# EU Midcap Echo

> Short-horizon directional signals for European mid-cap equities via XGBoost + regime filtering.

## What it does

- Downloads OHLCV for **196 European mid/large caps** across 12 exchanges plus VSTOXX and FRED macro series
- Engineers ~150 features: momentum, vol, distance-from-high, ATR, volume z-scores, sector/country dummies, macro regime
- Trains an **Optuna-tuned XGBoost** (plus RF, DT, LogReg baselines) optimized for precision at threshold 0.55
- Backtests with two simulators: a fast vector sim (flat $100 bets, signal quality) and a realistic risk-managed event loop (position sizing, 5-position cap, 6% stops, VSTOXX regime filter, fees, equal-weight benchmark)
- Runs a **daily incremental pipeline** that scores the universe and writes `reports/latest_trades.csv` with high-conviction longs plus a top-10 watchlist (optional Telegram push)

## Key results (out-of-sample, Jan 2024 – Nov 2026)

**Raw signal (vector sim, $100/trade, 20-day hold, 20 bps fees):**

| Threshold | Trades | Win rate | CAGR | Max DD | Final $ (from $100k) |
|---|---:|---:|---:|---:|---:|
| 0.50 | 21,398 | 54.2% | 7.0% | −5.1% | $125,382 |
| 0.55 | 16,915 | 54.6% | 5.9% | −3.8% | $121,103 |
| 0.60 | 12,939 | 55.0% | 5.2% | −2.8% | $118,298 |
| 0.65 | 9,635 | 55.4% | 4.2% | −2.2% | $114,849 |

Precision is **monotonically calibrated** — higher thresholds deliver higher win rates, exactly as you want from a production signal.

**Risk-managed portfolio** (Kelly sizing, max 5 positions, 6% stops, regime filter, $100k start):
- 181 trades, 53.6% win rate, CAGR 0.4%, Sharpe 0.13, Max DD −5.2%
- Equal-weight benchmark over same window: CAGR 9.9%, Sharpe 0.78
- Alpha vs benchmark: −9.5%

See `reports/final_report.md` for the full discussion (including why the risk-managed version underperforms in this bull-market window — it's an honest result, not a bug).

**Latest signals (2026-11-20):** EN.PA, SGO.PA, ACS.MC (3 actionable longs); watchlist includes KGH.WA, MAERSK-B.CO, GALP.LS, VOE.VI, PKN.WA, HLAG.DE, DHER.DE.

## Quickstart

```bash
# 1. install
pip install -r requirements.txt

# 2. build features, train models, backtest (~5-10 min first run)
python scripts/run_pipeline.py

# 3. explore notebooks in order 01 → 05
jupyter lab notebooks/

# 4. daily run (incremental, ~1 min after first run)
python scripts/daily_predict.py
# writes reports/latest_trades.csv; pushes to Telegram if TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID are set
```

Simulate only (uses saved model bundle):
```bash
python scripts/run_pipeline.py --simulate-only
```

## Notebooks

| # | Notebook | Purpose |
|---|---|---|
| 01 | `01_data_exploration.ipynb` | Universe stats, price histories, data quality |
| 02 | `02_feature_engineering.ipynb` | Build features, correlation heatmap, feature importance preview |
| 03 | `03_modeling.ipynb` | Train 4 models, precision/recall/ROC curves |
| 04 | `04_simulation.ipynb` | Vector + risk-managed backtests, equity curve plot vs benchmark |
| 05 | `05_daily_demo.ipynb` | Run daily pipeline, display latest signals/watchlist |

## Project layout

```
src/
  data/        downloaders (yfinance, FRED, news) + universe.yaml
  features/    build_features.py — momentum, vol, macro, categoricals
  models/      train.py (Optuna RF+XGB), predict.py (robust inference)
  trading/     metrics.py, sim_vectorized.py, sim_exact.py, sim_risk_managed.py
scripts/
  run_pipeline.py   full train+simulate CLI
  daily_predict.py  incremental daily scoring
models/models_bundle.joblib
reports/
  sim_summary.csv, equity_curve.csv, benchmark_curve.csv, latest_trades.csv
tests/         pytest suite (3 tests)
```

## Telegram alerts (optional)

Set environment variables before running `daily_predict.py`:
```bash
export TELEGRAM_BOT_TOKEN="123456:ABC-..."
export TELEGRAM_CHAT_ID="987654321"
```
Without these, the pipeline runs normally and just writes the CSV.

## Caveats / known limitations

- Universe is current constituents (survivorship bias — delisted names excluded).
- Uses close prices, no slippage model; 20 bps commissions.
- Sentiment feature scaffolded but defaults to 0 without a NewsAPI key.
- Risk-managed regime thresholds are heuristic, not walk-forward optimized.

See `reports/final_report.md` for the full write-up, methodology, and discussion.

## Tests

```bash
pytest tests/ -q
```
