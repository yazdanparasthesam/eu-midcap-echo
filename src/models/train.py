"""Train + tune DT / RF / XGBoost / LogReg models with Optuna; save best model + ONNX."""
import json, gc, os
import numpy as np, pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
from ..bootstrap import ensure
ensure({"xgboost": "xgboost", "optuna": "optuna"})
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from ..config import DATA_PRO, MODELS, log

TARGET = "is_positive_growth_20d_future"

def temporal_split(dates):
    s = pd.Series(dates); mn,mx = s.min(), s.max()
    tr_end = mn + pd.Timedelta(days=int((mx-mn).days*0.70))
    v_end  = tr_end + pd.Timedelta(days=int((mx-mn).days*0.15))
    return np.where(s<=np.datetime64(tr_end),0,
                    np.where(s<=np.datetime64(v_end),1,2)).astype("int8")

def _build_Xy(df, features, mask):
    sub = df.loc[mask, features + [TARGET, "growth_future_20d"]].replace([np.inf,-np.inf], np.nan).fillna(0)
    X = sub[features].astype("float32").values
    y = sub[TARGET].values.astype("int8")
    return X, y

def _tune_rf(X_tr, y_tr, X_v, y_v):
    def obj(t):
        md = t.suggest_int("max_depth", 4, 16)
        ne = t.suggest_int("n_estimators", 50, 300, step=50)
        m = RandomForestClassifier(n_estimators=ne, max_depth=md, random_state=42, n_jobs=-1)
        m.fit(X_tr, y_tr)
        p = m.predict(X_v)
        return precision_score(y_v, p, zero_division=0)
    s = optuna.create_study(direction="maximize"); s.optimize(obj, n_trials=5, show_progress_bar=False)
    return s.best_params, s.best_value

def _tune_xgb(X_tr, y_tr, X_v, y_v):
    if not HAS_XGB: return None, None
    def obj(t):
        md = t.suggest_int("max_depth", 3, 10)
        lr = t.suggest_float("lr", 0.02, 0.3, log=True)
        ss = t.suggest_float("subsample", 0.6, 1.0)
        cs = t.suggest_float("colsample", 0.6, 1.0)
        m = xgb.XGBClassifier(n_estimators=200, max_depth=md, learning_rate=lr,
                              subsample=ss, colsample_bytree=cs,
                              eval_metric="logloss", random_state=42, n_jobs=-1,
                              tree_method="hist")
        m.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
        p = (m.predict_proba(X_v)[:,1] >= 0.55).astype("int8")
        return precision_score(y_v, p, zero_division=0)
    s = optuna.create_study(direction="maximize"); s.optimize(obj, n_trials=5, show_progress_bar=False)
    return s.best_params, s.best_value

def train(df=None):
    from ..features.build_features import get_feature_lists, build_dataset
    if df is None:
        df = pd.read_parquet(DATA_PRO / "df_combined.parquet")
    dummies = [c for c in df.columns if c.startswith(("Month_","Weekday_","ticker_type_","sector_","disclosure_polarity_"))]
    NUMERICAL, DUMMIES, _ = get_feature_lists(df, dummies)
    FEATURES = NUMERICAL + DUMMIES
    log.info(f"training with {len(FEATURES)} features")

    split = temporal_split(df["Date"])
    m_tr = split == 0; m_v = split == 1; m_te = split == 2
    X_tr, y_tr = _build_Xy(df, FEATURES, m_tr)
    X_v,  y_v  = _build_Xy(df, FEATURES, m_v)
    X_te, y_te = _build_Xy(df, FEATURES, m_te)
    log.info(f"splits: train={len(y_tr)}, val={len(y_v)}, test={len(y_te)}")

    models = {}
    # Manual baseline DT depth=7 (lecture-style)
    dt = DecisionTreeClassifier(max_depth=7, random_state=42).fit(X_tr, y_tr)
    models["dt_depth7"] = dt
    # Tuned RF (fewer estimators for speed)
    rf_p, _ = _tune_rf(X_tr, y_tr, X_v, y_v)
    rf = RandomForestClassifier(n_estimators=min(rf_p["n_estimators"], 100),
                                max_depth=rf_p["max_depth"],
                                random_state=42, n_jobs=-1).fit(X_tr, y_tr)
    models["rf_best"] = rf
    # Tuned XGBoost (flagship; fewer estimators for speed)
    if HAS_XGB:
        x_p, _ = _tune_xgb(X_tr, y_tr, X_v, y_v)
        xm = xgb.XGBClassifier(n_estimators=150, max_depth=x_p["max_depth"], learning_rate=x_p["lr"],
                               subsample=x_p["subsample"], colsample_bytree=x_p["colsample"],
                               eval_metric="logloss", random_state=42, tree_method="hist", n_jobs=-1)
        xm.fit(X_tr, y_tr, eval_set=[(X_v,y_v)], verbose=False)
        models["xgb_best"] = xm
    # LogReg on scaled data
    sc = StandardScaler(); X_trs = sc.fit_transform(X_tr); X_vs = sc.transform(X_v)
    lr = LogisticRegression(max_iter=500, C=0.1, class_weight="balanced").fit(X_trs, y_tr)
    models["lr_balanced"] = lr
    # Use the actual feature list the models were fit on (from sklearn's feature_names_in_)
    first_mdl = models.get("xgb_best") or models.get("rf_best") or next(iter(models.values()))
    actual_features = list(first_mdl.feature_names_in_) if hasattr(first_mdl, "feature_names_in_") else FEATURES
    joblib.dump({"models": models, "features": actual_features, "scaler": sc,
                 "rf_params": rf_p},
                MODELS / "models_bundle.joblib")
    log.info(f"saved bundle with {len(actual_features)} features, {len(models)} models")

    # Evaluate on test set for each model at various thresholds
    results = []
    for name, m in models.items():
        if name == "lr_balanced":
            prob = m.predict_proba(sc.transform(X_te))[:,1]
        else:
            prob = m.predict_proba(X_te)[:,1]
        for thr in (0.50, 0.55, 0.60, 0.65, 0.70):
            p = (prob >= thr).astype("int8")
            if p.sum() < 50: continue
            results.append(dict(model=name, threshold=thr, n=int(p.sum()),
                                precision=precision_score(y_te,p,zero_division=0),
                                mean_proba=float(prob[p].mean())))
    res = pd.DataFrame(results).sort_values("precision", ascending=False)
    res.to_csv(MODELS / "test_threshold_results.csv", index=False)
    log.info("best models:\n" + res.head(8).to_string(index=False))
    del X_tr,X_v,X_te; gc.collect()
    return models, FEATURES, res

if __name__ == "__main__":
    train()
