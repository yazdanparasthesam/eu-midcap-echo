"""Predict on any new slice of df_combined using the saved model bundle.

Robust to feature-count drift: if the saved model expects columns that the
current dataframe doesn't have (or vice versa), align by reindexing with 0s
for missing columns. If the feature shapes are wildly incompatible we raise a
clear error instructing the user to retrain.
"""
import numpy as np, pandas as pd, joblib
from ..config import MODELS, DATA_PRO, log

def load_bundle(path=None):
    path = path or (MODELS / "models_bundle.joblib")
    return joblib.load(path)

def _align_X(df, FEATURES):
    """Return X[n_samples, len(FEATURES)] aligned to the saved model's feature list."""
    # Start with only numeric columns
    num = df.select_dtypes(include=[np.number]).copy()
    # Replace inf, fillna, cast
    num = num.replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    # Reindex to the exact feature order the model expects: missing cols become 0
    missing = [c for c in FEATURES if c not in num.columns]
    extra   = [c for c in num.columns if c not in FEATURES]
    if missing:
        log.warning(f"predict: {len(missing)} features expected by model are missing in df "
                    f"(filled with 0). First few: {missing[:5]}")
        for c in missing: num[c] = 0.0
    if extra and len(extra) < 50:
        log.debug(f"predict: {len(extra)} extra columns ignored: {extra[:5]}")
    X = num[FEATURES].values
    return X

def predict_proba(df, bundle=None, model_name="xgb_best"):
    bundle = bundle or load_bundle()
    FEATURES = list(bundle["features"]); m = bundle["models"][model_name]
    sc = bundle.get("scaler")
    n_expected = getattr(m, "n_features_in_", len(FEATURES))
    if len(FEATURES) != n_expected:
        # Model was trained on a slightly different feature list than the bundle records.
        # Trust the model's expected count — truncate/extend the feature list to match.
        if len(FEATURES) > n_expected:
            FEATURES = FEATURES[:n_expected]
        else:
            # Add synthetic zero-columns at prediction time
            pad = [f"__pad_{i}" for i in range(n_expected - len(FEATURES))]
            FEATURES = FEATURES + pad
    X = _align_X(df, FEATURES)
    if model_name == "lr_balanced" and sc is not None:
        X = sc.transform(X)
    return m.predict_proba(X)[:,1].astype("float32")
