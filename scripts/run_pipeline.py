"""Run the full pipeline (or a stage via flags)."""
import argparse, sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import log, DATA_RAW, DATA_PRO, REPORTS, MODELS

def stage_download():
    from src.data.download_yf import download as dl_yf
    from src.data.download_fred import download as dl_fred
    from src.data.download_dgap import download as dl_dgap
    dl_yf(); dl_fred(); dl_dgap()

def stage_from_drive():
    log.info("downloading pre-built parquet from Google Drive...")
    import gdown
    url = "https://drive.google.com/uc?id=1REPLACE_WITH_REAL_ID"
    out = DATA_PRO / "df_combined.parquet"
    if not out.exists():
        gdown.download(url, str(out), quiet=False, fuzzy=True)

def stage_features():
    from src.features.build_features import build_dataset
    return build_dataset()

def stage_train():
    from src.models.train import train
    train()

def stage_simulate():
    import numpy as np, pandas as pd, joblib
    from src.models.predict import predict_proba, load_bundle
    from src.trading.sim_vectorized import run as sim1
    from src.trading.sim_exact import run as sim_exact
    from src.trading.sim_risk_managed import run as sim_risk
    from src.trading.metrics import summarise
    df = pd.read_parquet(DATA_PRO / "df_combined.parquet")
    from src.models.train import temporal_split
    split = temporal_split(df["Date"])
    df_te = df[split == 2].reset_index(drop=True)

    # Load existing model (do NOT auto-retrain — predict.py handles feature alignment)
    bundle_path = MODELS / "models_bundle.joblib"
    if not bundle_path.exists():
        log.info("no model bundle found — training now (fast mode, 5 Optuna trials each)")
        stage_train()
    bundle = load_bundle()
    log.info(f"loaded model bundle: {len(bundle['models'])} models, {len(bundle['features'])} features")

    for mname in ["dt_depth7","rf_best","xgb_best","lr_balanced"]:
        if mname not in bundle["models"]: continue
        try:
            df_te[f"prob_{mname}"] = predict_proba(df_te, bundle, mname)
        except Exception as e:
            log.warning(f"  predict failed for {mname}: {e}")
            df_te[f"prob_{mname}"] = 0.0
    flag_model = "xgb_best" if "xgb_best" in bundle["models"] else "rf_best"
    proba_col = f"prob_{flag_model}"
    if proba_col not in df_te.columns:
        proba_col = [c for c in df_te.columns if c.startswith("prob_")][0]
    rows = []
    eq_vec = None
    for thr in (0.50, 0.55, 0.60, 0.65):
        pred_col = f"pred_{flag_model}_{thr}"
        df_te[pred_col] = (df_te[proba_col] >= thr).astype("int8")
        r = sim1(df_te, pred_col)
        eq_vec = r.pop("_equity", eq_vec)
        rows.append(r)
    eq_r, bench = None, None
    try:
        eq_r, stats_r, bench = sim_risk(df_te, proba_col, threshold=0.55)
        rows.append(stats_r)
    except Exception as e:
        import traceback; traceback.print_exc()
        log.warning(f"risk-managed sim failed: {e}")
    # Write equity curves
    out_eq = eq_r if eq_r is not None else eq_vec
    if out_eq is not None:
        out_eq.to_csv(REPORTS / "equity_curve.csv", header=["equity"])
    if bench is not None:
        bench.to_csv(REPORTS / "benchmark_curve.csv", header=["benchmark"])
    cols = ["strategy","n","net","mean_per_day","q75_per_day","capital",
            "CAGR","Sharpe","Sortino","MaxDD","Final","n_trades",
            "win_rate","benchmark_CAGR","benchmark_Sharpe","benchmark_Final","alpha_vs_bench"]
    summary_df = pd.DataFrame(rows).reindex(columns=cols)
    summary_df.to_csv(REPORTS / "sim_summary.csv", index=False)
    log.info(f"sim summary:\n{summary_df.to_string(index=False)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--from-drive", action="store_true")
    ap.add_argument("--features-only", action="store_true")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--simulate-only", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    if args.from_drive:
        stage_from_drive()
    elif args.download_only:
        stage_download()
    else:
        if not args.features_only and not args.train_only and not args.simulate_only or args.full:
            stage_download()
        if not args.train_only and not args.simulate_only or args.full or args.features_only:
            if not (DATA_PRO / "df_combined.parquet").exists() or args.full:
                stage_features()
        if not args.features_only and not args.simulate_only or args.full or args.train_only:
            stage_train()
        if not args.features_only and not args.train_only or args.full or args.simulate_only:
            stage_simulate()
            # Auto-generate equity-curve plot for README/report
            try:
                from scripts.make_report_plots import _main as _make_plots
                _make_plots()
            except Exception as e:
                log.warning(f"could not generate report plot: {e}")
    log.info(f"done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
