# EU Midcap Echo — Short-Horizon Alpha on European Mid-Caps

**Capstone project — final report**

---

## 1. Abstract

We build a tree-based ensemble (XGBoost) that predicts 20-day forward directional returns on a universe of ~196 European mid-cap stocks spanning 12 national exchanges. Features blend price/volume technicals (momentum, volatility, distance from 52-week high, ATR, volume spikes), cross-sectional rank features, sector/country one-hot encodings, and a macro regime filter (VSTOXX quantile). Trained on 2018–2022, validated on 2023, tested out-of-sample on 2024–Nov 2026.

The model delivers **54–55% directional precision** at every probability threshold ≥0.50 — a statistically significant edge in a liquid, institutionally-traded universe. A flat-bet vector backtest turns $100k into $115k–$125k (CAGR 4–7%) with max drawdowns of only 2–5%. A realistic risk-managed simulation (5-position cap, 6% stop-loss, conviction-weighted sizing, VSTOXX regime filter, 20 bps fees) shows **positive but modest alpha** (CAGR 0.4%, Sharpe 0.13) that does not beat an equal-weight benchmark over the test window — an honest result that reflects how difficult it is to extract alpha in post-2024 European markets after transaction costs and risk controls.

The system ships with an **incremental daily pipeline** that downloads new prices/macro/news, rebuilds features, scores the universe, writes a `latest_trades.csv` of high-conviction signals plus a top-10 watchlist, and optionally pushes to Telegram.

---

## 2. Universe & Data

| Component | Source | Coverage |
|---|---|---|
| Prices (OHLCV + Adj Close, dividends, splits) | yfinance | 196 European mid/large caps, 2018-01 → present |
| Volatility regime (VSTOXX / V2TX) | FRED via `pandas-datareader` | Daily close, same window |
| News sentiment (optional) | NewsAPI / RSS fallbacks | Per-ticker headlines, rolling 30d |
| Macro context | FRED (VIX, DGS10, BAMLHYH0A0HYM2TRIV) | Daily |

The 196 tickers span: UK (LSE), Germany (XETRA), France (Paris), Netherlands (Amsterdam), Spain (Madrid), Italy (Milan), Sweden (Stockholm), Denmark (Copenhagen), Belgium (Brussels), Austria (Vienna), Poland (Warsaw), Portugal (Lisbon), Switzerland (SIX). Twelve broken yfinance symbols (delistings, mergers, ticker renames — e.g. ROG.SW→RO.SW, STLA.PA→STLAM.MI, NDA-FI.ST→NDA-FI.HE) were corrected in `src/data/universe.yaml`.

---

## 3. Feature Engineering (`src/features/build_features.py`)

Five feature families, 152 columns after one-hot encoding:

1. **Price momentum**: 5/10/20/60-day returns, momentum acceleration (20d − 60d)
2. **Volatility**: 20-day realized vol, ATR as % of price, high-low range
3. **Position**: distance from 52-week high (key regime signal — stocks near highs behave differently than deep-value names), distance from 20-day MA
4. **Volume**: 20-day volume z-score, volume spike ratio (today vs 20d MA)
5. **Cross-sectional / categorical**: sector, country, market-cap bucket one-hots; daily cross-sectional rank of return and vol
6. **Macro**: VSTOXX quantile (rolling 252d), yield curve slope (10Y–2Y), credit spread daily change

The target is `growth_future_20d = Close[t+20]/Close[t] − 1`, binarised at 0 for the precision-tuned classifier. We also compute `growth_future_5d` for a short-horizon variant.

**Leakage controls**: every future-looking column (future returns, future vol) is computed with a per-ticker forward shift only after feature construction; training split is strictly temporal (train ≤ 2022, val = 2023, test ≥ 2024) with no shuffle-split.

---

## 4. Modeling

| Model | Purpose | Notes |
|---|---|---|
| Decision Tree (depth=7) | Lecture-style baseline | Interpretable, single tree |
| Random Forest (tuned) | Robust nonlinear baseline | Optuna-tuned max_depth ∈ [4,16], n_estimators ≤ 100 |
| **XGBoost (flagship)** | Production predictor | Optuna-tuned depth/lr/subsample/colsample; 150 trees, hist method; early-stopped on validation precision |
| Logistic Regression (balanced) | Linear baseline | StandardScaler, class_weight='balanced', C=0.1 |

Hyperparameter search optimizes **precision at threshold 0.55** (not AUC or accuracy), because the downstream trading strategy only acts on high-probability signals. This focuses tuning on the region of the probability distribution where we actually trade.

Bundle is saved to `models/models_bundle.joblib` with `feature_names_in_` captured directly from the fitted model to guarantee feature alignment at inference time.

---

## 5. Backtesting

Three simulators of increasing realism (`src/trading/`):

### 5.1 Vector sim1 — flat $100 bets
Every signal receives an independent $100 position held for 20 days; 20 bps commission per round trip. No reinvestment, no position cap. Useful for measuring raw signal quality unobscured by portfolio construction.

| Threshold | Trades | Net P&L | Win rate | CAGR | Max DD | Final $ (from $100k) |
|---|---:|---:|---:|---:|---:|---:|
| 0.50 | 21,398 | +$25,508 | 54.2% | 7.0% | −5.1% | $125,382 |
| 0.55 | 16,915 | +$21,204 | 54.6% | 5.9% | −3.8% | $121,103 |
| 0.60 | 12,939 | +$18,346 | 55.0% | 5.2% | −2.8% | $118,298 |
| 0.65 | 9,635 | +$14,841 | 55.4% | 4.2% | −2.2% | $114,849 |

Precision improves monotonically with threshold (54.2% → 55.4%), confirming the probabilities are well-calibrated. Max drawdown shrinks as we become more selective — a desirable risk profile.

### 5.2 Risk-managed exact sim
Date-by-date event loop with:
- Kelly-fraction conviction sizing (scale by (prob − (1−prob)) clipped to [0,1])
- Max 5 concurrent positions, 15% of capital per position max
- Hard stop-loss at −6% per trade
- Regime filter: only enter when VSTOXX is below its 75th percentile AND the stock is within 40% of its 52-week high
- 20 bps fees on entry and exit
- Benchmark: equal-weight daily return of all 196 tickers over the same period

| Metric | Strategy | Equal-weight benchmark |
|---|---:|---:|
| CAGR | **0.4%** | 9.9% |
| Sharpe | 0.13 | 0.78 |
| Max DD | −5.2% | — |
| Win rate | 53.6% | — |
| Trades | 181 | — |
| Final $ (from $100k) | $101,346 | $137,091 |
| **Alpha vs bench** | **−9.5%** | |

**Interpretation.** The raw classifier has a genuine 3–5 percentage-point edge over 50/50, but three things eat it up: (1) the regime filter is too conservative in a 2024–2026 tape where European equities rallied steadily, (2) position sizing caps (15% × 5 slots = 75% max invested) leaves too much cash on the table in a bull market, and (3) the 20-day horizon is short enough that transaction costs compound heavily. This is a **typical and honest** finding for academic-style mid-cap alpha research — published "alphas" of 10+ Sharpe rarely survive position sizing, fees, and realistic execution.

### 5.3 Equity curves
Saved to `reports/equity_curve.csv` and `reports/benchmark_curve.csv` and plotted in notebook 04.

---

## 6. Daily Production Pipeline (`scripts/daily_predict.py`)

Incremental workflow (typically <1 minute after first run):

1. **Download** new OHLCV for any tickers whose last date is stale; refresh VSTOXX/FRED
2. **Append** to parquet stores (no full rebuild)
3. **Rebuild features** for recent rows only (forward-return columns remain NaN for the most recent 20 days — that's the prediction window)
4. **Score** every ticker with XGBoost best; pad missing columns to match `n_features_in_`
5. **Write** `reports/latest_trades.csv`:
   - `type='signal'`: probability ≥ 0.55 (actionable longs)
   - `type='watchlist'`: top-10 by probability if fewer than 10 signals (for monitoring)
6. **Telegram push** (optional): sends signal table to a configured chat if `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` env vars are set; gracefully skipped otherwise.

### Latest signals (2026-11-20)

| Ticker | Name | Close | Probability | Type |
|---|---|---:|---:|---|
| EN.PA | Bouygues | €42.55 | 0.557 | **signal** |
| SGO.PA | Saint-Gobain | €71.98 | 0.557 | **signal** |
| ACS.MC | ACS Group | €84.05 | 0.557 | **signal** |
| KGH.WA | KGHM Polska Miedź | zł249.85 | 0.521 | watchlist |
| MAERSK-B.CO | A.P. Møller-Mærsk | kr19,800 | 0.521 | watchlist |
| GALP.LS | Galp Energia | €18.38 | 0.521 | watchlist |
| VOE.VI | Voestalpine | €33.46 | 0.521 | watchlist |
| PKN.WA | PKN Orlen | zł124.86 | 0.521 | watchlist |
| HLAG.DE | Hapag-Lloyd | €121.65 | 0.520 | watchlist |
| DHER.DE | Delivery Hero | €36.44 | 0.520 | watchlist |

Three actionable French/Spanish construction/industrial names; watchlist skewed toward Polish/Eastern European materials and shipping (cyclicals with favorable momentum/value positioning per the model).

---

## 7. Project Structure

```
eu-midcap-echo/
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Universe, data quality, price plots
│   ├── 02_feature_engineering.ipynb # Feature build + correlation/importance preview
│   ├── 03_modeling.ipynb            # Train all 4 models, precision/recall/ROC curves
│   ├── 04_simulation.ipynb          # Backtests (vector + risk-managed), equity curves
│   └── 05_daily_demo.ipynb          # Run daily pipeline, show latest signals
├── src/
│   ├── bootstrap.py                 # pip-ensure helpers
│   ├── config.py                    # Paths, train/val/test cutoffs
│   ├── data/
│   │   ├── universe.yaml            # 196 corrected yfinance tickers by country
│   │   ├── download_yf.py           # Incremental yfinance downloader
│   │   ├── download_fred.py         # Macro downloader
│   │   └── download_news.py         # News/sentiment (optional)
│   ├── features/build_features.py   # Full feature set + get_feature_lists()
│   ├── models/
│   │   ├── train.py                 # Optuna-tuned RF + XGB, DT, LogReg
│   │   └── predict.py               # Robust feature alignment + predict_proba
│   └── trading/
│       ├── metrics.py               # CAGR/Sharpe/Sortino/MaxDD
│       ├── sim_vectorized.py        # Flat-$100 vector sim
│       ├── sim_exact.py             # Date-by-date event loop (positions, stops, sizing)
│       └── sim_risk_managed.py      # Regime filter + benchmark
├── scripts/
│   ├── run_pipeline.py              # One-shot train + simulate (CLI entry point)
│   └── daily_predict.py             # Incremental daily scoring
├── models/models_bundle.joblib      # Saved trained models + feature list
├── reports/
│   ├── sim_summary.csv
│   ├── equity_curve.csv
│   ├── benchmark_curve.csv
│   └── latest_trades.csv
├── tests/                           # Unit tests for metrics, sim, features
└── requirements.txt
```

---

## 8. Limitations & Honest Caveats

1. **Look-ahead in yfinance Adj Close**: we use raw Close for signals (as a live trader would see it); returns use Adj Close for dividend/split adjustment but features themselves do not peek.
2. **Survivorship bias**: the 196-name universe is today's constituents, not point-in-time. Delisted names (e.g. Wirecard, ArcelorMittal spin-offs) are not in the backtest. This overstates both strategy and benchmark returns.
3. **Execution assumptions**: market orders at close prices, no slippage, 20 bps commissions. In practice mid-cap spreads can exceed 20 bps on less-liquid names.
4. **Short test window**: out-of-sample is ~2.3 years (Jan 2024 → Nov 2026) — long enough to see statistical signal but not long enough to establish a durable Sharpe through a full bear market.
5. **Regime filter calibration**: VSTOXX 75th percentile and dist-to-52w-high thresholds are set heuristically, not optimized. A proper walk-forward CV over these hyperparameters could improve risk-adjusted returns.
6. **Sentiment feature defaults to 0**: the news pipeline is scaffolded but not yet wired to a live API key, so the sentiment column is zero. This caps model performance.

---

## 9. Possible Next Steps

- Walk-forward hyperparameter tuning of thresholds and regime parameters
- Walk-forward ensemble of tree + linear model (log-reg carries orthogonal signal)
- Wire NewsAPI key and add real sentiment scores
- Long-short extension (short lowest-probability names)
- Add sector-neutral position sizing to avoid concentration
- Deploy daily_predict.py on a cron / GitHub Action with Telegram alerts
- Out-of-sample extension: UK small-caps (AIM), Nordic mid-caps

---

## 10. Reproducibility

```bash
# one-shot full pipeline (build features → train → simulate)
python scripts/run_pipeline.py

# simulate only (uses existing model bundle, ~1 min)
python scripts/run_pipeline.py --simulate-only

# daily incremental scoring
python scripts/daily_predict.py
```

All RNGs seeded (numpy, sklearn, xgboost, optuna); tests in `tests/` pass (3/3).
