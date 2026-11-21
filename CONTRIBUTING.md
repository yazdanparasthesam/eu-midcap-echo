# Contributing

1. Fork, clone, `make install`.
2. Branch off `main`: `git checkout -b feat/your-idea`.
3. Keep modules small; run `make test && make lint` before pushing.
4. PRs welcome — especially:
   - better German sentiment model (fine-tuned FinBERT on DGAP corpus),
   - additional EU mid-cap exchanges (BME, LSE mid-cap),
   - IBKR / Alpaca live-trading integration,
   - additional benchmark strategies.

Please do not commit `data/`, `models/*.joblib`, or `.env`.
