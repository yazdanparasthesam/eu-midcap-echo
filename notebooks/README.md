# Notebooks

Research notebooks — each calls into the `src/` package.

- `01_data_collection.ipynb` — walk-through of the three data sources (Yahoo, FRED, DGAP) with sample rows and download stats.
- `02_eda.ipynb` — target distribution, feature-target correlations, per-sector event-study preview, VIF/missingness checks.
- `03_modeling.ipynb` — DT/RF/XGB/LR comparison, Optuna tuning curves, precision/recall vs. threshold charts, feature importance.
- `04_simulation.ipynb` — sim1→sim4 backtests, equity curves vs DAX, per-ticker P&L breakdown, rolling return plots.
- `05_daily_demo.ipynb` — end-to-end demonstration that imports `scripts/daily_predict.py` and shows the latest signals.

To run:
```bash
jupyter lab notebooks/01_data_collection.ipynb
```

Each notebook starts with `%load_ext autoreload; %autoreload 2` so you can edit `src/` modules without restarting the kernel.
