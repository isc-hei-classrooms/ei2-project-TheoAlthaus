"""
Entrainement modeles — version 4
===================================
"""

import pickle
import datetime
import polars as pl
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import xgboost as xgb

from config import DATA_DIR, FILE_OIKEN_CLEAN

MODELS_DIR = DATA_DIR / "models"
FILE_FEAT  = DATA_DIR / "features_v4b.parquet"

WF_STEP = 30 * 96  # 30 jours pour tous les modeles

RF_PARAMS = dict(
    n_estimators=200, max_depth=15,
    min_samples_leaf=5, n_jobs=-1, random_state=42,
)
LGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, max_depth=8,
    num_leaves=63, min_child_samples=10,
    n_jobs=-1, random_state=42, verbose=-1,
)
XGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    n_jobs=-1, random_state=42, verbosity=0,
)


def build_oiken_mask(timestamps: list) -> np.ndarray:
    bug_start = datetime.datetime(2025, 9, 13,  2, 15, 0,
                                  tzinfo=datetime.timezone.utc)
    bug_end   = datetime.datetime(2025, 9, 17,  2,  0, 0,
                                  tzinfo=datetime.timezone.utc)
    mask = []
    for ts in timestamps:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=datetime.timezone.utc)
        mask.append(not (bug_start <= ts <= bug_end))
    return np.array(mask)


def compute_metrics(y_true, y_pred, label, mask=None):
    m = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask is not None:
        m = m & mask
    mae  = mean_absolute_error(y_true[m], y_pred[m])
    rmse = np.sqrt(mean_squared_error(y_true[m], y_pred[m]))
    print(f"  {label:<30} MAE={mae:.5f}  RMSE={rmse:.5f}  (n={m.sum():,})")
    return {"model": label, "mae": mae, "rmse": rmse}


def load_data():
    df        = pl.read_parquet(FILE_FEAT).sort("timestamp")
    feat_cols = [c for c in df.columns
                 if c not in ("timestamp", "date_locale", "target", "split")]

    train = df.filter(pl.col("split") == "train")
    val   = df.filter(pl.col("split") == "val")
    test  = df.filter(pl.col("split") == "test")

    X_train = train.select(feat_cols).to_numpy()
    y_train = train["target"].to_numpy()
    X_val   = val.select(feat_cols).to_numpy()
    y_val   = val["target"].to_numpy()
    X_test  = test.select(feat_cols).to_numpy()
    y_test  = test["target"].to_numpy()
    ts_val  = val["timestamp"].to_list()
    ts_test = test["timestamp"].to_list()

    print(f"Train : {X_train.shape[0]:,} pts ({X_train.shape[0]//96} jours)")
    print(f"Val   : {X_val.shape[0]:,} pts ({X_val.shape[0]//96} jours)")
    print(f"Test  : {X_test.shape[0]:,} pts ({X_test.shape[0]//96} jours)")
    print(f"Features : {X_train.shape[1]}")

    oiken  = pl.read_parquet(FILE_OIKEN_CLEAN).sort("timestamp")
    fc_map = {ts: fc for ts, fc in zip(
        oiken["timestamp"].to_list(),
        oiken["forecast_load"].to_list()
    )}
    y_fc_val  = np.array([fc_map.get(ts, np.nan) for ts in ts_val])
    y_fc_test = np.array([fc_map.get(ts, np.nan) for ts in ts_test])
    mask_val  = build_oiken_mask(ts_val)
    mask_test = build_oiken_mask(ts_test)

    print(f"  Points exclus periode buguee : {(~mask_test).sum():,}")

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            ts_val, ts_test, feat_cols,
            y_fc_val, y_fc_test, mask_val, mask_test, df)


def run_walkforward(name, model_class, params,
                    X_train_init, y_train_init,
                    X_wf, y_wf, ts_wf, splits_wf,
                    fc_map, mask_val, mask_test):

    print(f"\n{'='*60}")
    print(f"  WALK-FORWARD {name.upper()} (pas={WF_STEP//96}j)")
    print(f"{'='*60}")

    n_wf    = X_wf.shape[0]
    n_iters = (n_wf + WF_STEP - 1) // WF_STEP
    X_tr    = X_train_init.copy()
    y_tr    = y_train_init.copy()

    ts_all, sp_all, true_all, pred_all, fc_all = [], [], [], [], []

    for i in range(0, n_wf, WF_STEP):
        iter_num = i // WF_STEP + 1
        i_end    = min(i + WF_STEP, n_wf)

        model = model_class(**params)
        model.fit(X_tr, y_tr)

        X_pred = X_wf[i:i_end]
        y_true = y_wf[i:i_end]
        y_pred = model.predict(X_pred)
        ts_seg = ts_wf[i:i_end]
        sp_seg = splits_wf[i:i_end]
        fc_seg = np.array([fc_map.get(ts, np.nan) for ts in ts_seg])

        ts_all.extend(ts_seg)
        sp_all.extend(sp_seg)
        true_all.extend(y_true.tolist())
        pred_all.extend(y_pred.tolist())
        fc_all.extend(fc_seg.tolist())

        X_tr = np.vstack([X_tr, X_pred])
        y_tr = np.concatenate([y_tr, y_true])

        if iter_num % 5 == 0 or iter_num == n_iters:
            print(f"  Iter {iter_num:>3}/{n_iters}  "
                  f"(train={X_tr.shape[0]:,} pts)")

    metrics = []
    for split, mask in [("val", mask_val), ("test", mask_test)]:
        idx    = [i for i, s in enumerate(sp_all) if s == split]
        y_t    = np.array(true_all)[idx]
        y_p    = np.array(pred_all)[idx]
        y_f    = np.array(fc_all)[idx]
        m_mask = mask if split == "test" else None

        print(f"\n  {split} :")
        m  = compute_metrics(y_t, y_p, name,            m_mask)
        mf = compute_metrics(y_t, y_f, "oiken_forecast", m_mask)
        m["split"]  = split
        mf["split"] = split
        metrics.extend([m, mf])

    df_pred = pl.DataFrame({
        "timestamp": ts_all,
        "split":     sp_all,
        "model":     [name] * len(ts_all),
        "true":      true_all,
        "pred":      pred_all,
        "forecast":  fc_all,
    }).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))

    return df_pred, metrics


if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Chargement features v4b...")
    (X_train, y_train, X_val, y_val, X_test, y_test,
     ts_val, ts_test, feat_cols,
     y_fc_val, y_fc_test, mask_val, mask_test, df_full) = load_data()

    oiken  = pl.read_parquet(FILE_OIKEN_CLEAN).sort("timestamp")
    fc_map = {ts: fc for ts, fc in zip(
        oiken["timestamp"].to_list(),
        oiken["forecast_load"].to_list()
    )}

    # ── Baseline Oiken ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  BASELINE OIKEN FORECAST")
    print(f"{'='*60}")
    fc_val_m  = compute_metrics(y_val,  y_fc_val,  "val")
    fc_test_m = compute_metrics(y_test, y_fc_test, "test", mask_test)
    fc_val_m["split"]  = "val"
    fc_test_m["split"] = "test"

    # ── Validation simple ─────────────────────────────────────────────────
    simple_metrics = [
        {**fc_val_m,  "model": "oiken_forecast"},
        {**fc_test_m, "model": "oiken_forecast"},
    ]
    simple_preds = []

    for name, model in [
        ("random_forest", RandomForestRegressor(**RF_PARAMS)),
        ("lightgbm",      lgb.LGBMRegressor(**LGB_PARAMS)),
        ("xgboost",       xgb.XGBRegressor(**XGB_PARAMS)),
    ]:
        print(f"\n{'='*60}")
        print(f"  SIMPLE — {name.upper()}")
        print(f"{'='*60}")
        print("  Entrainement...")
        model.fit(X_train, y_train)
        print("  Evaluation :")
        mv = compute_metrics(y_val,  model.predict(X_val),  "val")
        mt = compute_metrics(y_test, model.predict(X_test), "test", mask_test)
        mv["split"] = "val"
        mt["split"] = "test"
        simple_metrics.extend([{**mv, "model": name}, {**mt, "model": name}])

        with open(MODELS_DIR / f"{name}_v4b.pkl", "wb") as f:
            pickle.dump(model, f)

        for split, X, y_true, ts, y_fc in [
            ("val",  X_val,  y_val,  ts_val,  y_fc_val),
            ("test", X_test, y_test, ts_test, y_fc_test),
        ]:
            y_pred = model.predict(X)
            simple_preds.append(pl.DataFrame({
                "timestamp": ts, "split": [split]*len(ts),
                "model": [name]*len(ts),
                "true": y_true.tolist(), "pred": y_pred.tolist(),
                "forecast": y_fc.tolist(),
            }).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC"))))

    pl.concat(simple_preds).write_parquet(MODELS_DIR / "predictions_v4b.parquet")
    pl.DataFrame(simple_metrics).write_parquet(MODELS_DIR / "results_v4b.parquet")

    # ── Walk-forward ──────────────────────────────────────────────────────
    df_wf     = df_full.filter(pl.col("split").is_in(["val", "test"]))
    X_wf_all  = df_wf.select(feat_cols).to_numpy()
    y_wf_all  = df_wf["target"].to_numpy()
    ts_wf     = df_wf["timestamp"].to_list()
    splits_wf = df_wf["split"].to_list()

    wf_preds, wf_metrics = [], []
    for name, model_class, params in [
        ("random_forest", RandomForestRegressor, RF_PARAMS),
        ("lightgbm",      lgb.LGBMRegressor,    LGB_PARAMS),
        ("xgboost",       xgb.XGBRegressor,     XGB_PARAMS),
    ]:
        df_p, m = run_walkforward(
            name, model_class, params,
            X_train, y_train,
            X_wf_all, y_wf_all,
            ts_wf, splits_wf, fc_map,
            mask_val, mask_test,
        )
        wf_preds.append(df_p)
        wf_metrics.extend(m)

    pl.concat(wf_preds).write_parquet(MODELS_DIR / "wf_predictions_v4b.parquet")
    pl.DataFrame(wf_metrics).write_parquet(MODELS_DIR / "wf_results_v4b.parquet")

    # ── Resume ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESUME FINAL — v4b")
    print(f"{'='*60}")
    print("  (periode Oiken buguee exclue du calcul test)")

    print("\n  Simple :")
    print(pl.DataFrame(simple_metrics)
            .select(["model", "split", "mae", "rmse"])
            .sort(["split", "mae"]))

    print("\n  Walk-forward :")
    wf_df = pl.DataFrame(wf_metrics)
    print(wf_df.filter(pl.col("model") != "oiken_forecast")
               .select(["model", "split", "mae", "rmse"])
               .sort(["split", "mae"]))

    print("\n  Oiken forecast (reference) :")
    print(wf_df.filter(pl.col("model") == "oiken_forecast")
               .unique(subset=["split"])
               .select(["model", "split", "mae", "rmse"])
               .sort("split"))

    print(f"\nFichiers sauvegardes dans {MODELS_DIR} :")
    for p in sorted(MODELS_DIR.glob("*v4*")):
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  {p.name:<45} {size_mb:.1f} MB")