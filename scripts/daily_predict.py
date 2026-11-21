"""Incremental daily prediction: fetch latest prices, rebuild features,
score with saved model, write today's trades + optional Telegram notification.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd, numpy as np
from src.config import log, DATA_RAW, DATA_PRO, REPORTS, MODELS, load_universe
from src.data.download_yf import download as yf_download
from src.data.download_fred import download as fred_download
from src.data.download_dgap import download as dgap_download
from src.features.build_features import build_dataset
from src.models.predict import load_bundle, predict_proba
from src.monitoring.telegram_bot import send_trades

def main():
    log.info("=== daily incremental run ===")
    yf_download()       # incremental by ticker
    fred_download()
    try:
        dgap_download()
    except Exception as e:
        log.warning(f"news download skipped: {e}")
    df, _dummies = build_dataset()

    # Auto-retrain if model missing or incompatible
    bundle_path = MODELS / "models_bundle.joblib"
    retrain = not bundle_path.exists()
    if not retrain:
        try:
            bundle = load_bundle()
            sample_model = bundle["models"].get("xgb_best", next(iter(bundle["models"].values())))
            expected = sample_model.n_features_in_
            have = len(bundle["features"])
            if have != expected:
                log.warning(f"stale model ({have} vs {expected} feats) — retraining")
                retrain = True
        except Exception as e:
            log.warning(f"could not load model: {e} — retraining"); retrain = True
    if retrain:
        from src.models.train import train
        train(); bundle = load_bundle()

    flag_model = "xgb_best" if "xgb_best" in bundle["models"] else (
                 "rf_best" if "rf_best" in bundle["models"] else next(iter(bundle["models"])))
    proba_col = f"prob_{flag_model}"
    # score only the last 60 days (speed)
    last_date = df["Date"].max()
    window = df[df["Date"] >= last_date - pd.Timedelta(days=60)].copy()
    window[proba_col] = predict_proba(window, bundle, flag_model)
    # Latest date rows
    last = window[window["Date"] == last_date].copy()
    keep_cols = ["Date","Ticker","Close", proba_col, "sentiment_score"]
    for extra in ("vstoxx_q","atr_pct","dist_52w_high","volume_spike"):
        if extra in last.columns: keep_cols.append(extra)
    # Signals at a moderate threshold (0.55 catches more trades for the demo)
    THRESHOLD = 0.55
    signals = last[last[proba_col] >= THRESHOLD][keep_cols].copy()
    signals = signals.sort_values(proba_col, ascending=False)
    # Also always include the top-10 highest-probability tickers as a watchlist
    watchlist = last.sort_values(proba_col, ascending=False).head(10)[keep_cols].copy()
    watchlist["type"] = "watchlist"
    if len(signals) == 0:
        # No high-conviction signals; emit the top-10 watchlist so demo has output
        out = watchlist
    else:
        signals["type"] = "signal"
        out = pd.concat([signals, watchlist], ignore_index=True).drop_duplicates(subset=["Ticker"])
    out.to_csv(REPORTS / "latest_trades.csv", index=False)
    log.info(f"signals (p>={THRESHOLD}): {len(signals)}; watchlist: {len(watchlist)}; total rows: {len(out)}")
    log.info(f"top 10:\n{out.head(10).to_string(index=False)}")
    try:
        send_trades(str(REPORTS / "latest_trades.csv"))
    except Exception as e:
        log.warning(f"telegram send skipped: {e}")

if __name__ == "__main__":
    main()
