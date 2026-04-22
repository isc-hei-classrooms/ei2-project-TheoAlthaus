"""
Walk-forward validation — LightGBM et XGBoost
===============================================
Lit  : data/features_v1.parquet
       data/oiken_clean.parquet
Ecrit: data/models/wf_results_lgb_xgb_v1.parquet
       data/models/wf_predictions_lgb_xgb_v1.parquet

Principe :
  - Fenetre expansive, pas de 30 jours
  - Modeles : LightGBM et XGBoost (MultiOutputRegressor)
  - Comparaison avec Oiken forecast et WF Random Forest
"""

import pickle
from pathlib import Path

import polars as pl
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import xgboost as xgb

from config import DATA_DIR, FILE_OIKEN_CLEAN

MODELS_DIR = DATA_DIR / "models"
FILE_FEAT  = DATA_DIR / "features_v1.parquet"
WF_STEP    = 30

LGB_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=63,
    min_child_samples=10,
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)

XGB_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    verbosity=0,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    label: str) -> dict:
    mask_rows = ~(np.isnan(y_true).any(axis=1) |
                  np.isnan(y_pred).any(axis=1))
    y_true = y_true[mask_rows]
    y_pred = y_pred[mask_rows]

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"  {label:<30} MAE={mae:.5f}  RMSE={rmse:.5f}")
    return {"model": label, "mae": mae, "rmse": rmse}


def run_walkforward(model_name: str, base_estimator,
                    X_train_init, y_train_init,
                    X_wf_all, y_wf_all,
                    days_wf, splits_wf,
                    extract_forecast_fn) -> pl.DataFrame:

    print(f"\n{'='*60}")
    print(f"  WALK-FORWARD {model_name.upper()}")
    print(f"{'='*60}")

    n_wf    = X_wf_all.shape[0]
    n_iters = (n_wf + WF_STEP - 1) // WF_STEP

    X_train = X_train_init.copy()
    y_train = y_train_init.copy()

    all_preds = []

    for i in range(0, n_wf, WF_STEP):
        iter_num = i // WF_STEP + 1
        i_end    = min(i + WF_STEP, n_wf)

        # Entrainement
        model = MultiOutputRegressor(base_estimator.__class__(**base_estimator.get_params()),
                                     n_jobs=-1)
        model.fit(X_train, y_train)

        # Prediction
        X_pred  = X_wf_all[i:i_end]
        y_true  = y_wf_all[i:i_end]
        y_pred  = model.predict(X_pred)
        ts_pred = days_wf[i:i_end]
        sp      = splits_wf[i:i_end]
        y_fc    = extract_forecast_fn(ts_pred)

        for j in range(len(ts_pred)):
            row = {
                "timestamp_day": ts_pred[j],
                "split":         sp[j],
                "model":         model_name,
                "iter":          iter_num,
            }
            for q in range(y_true.shape[1]):
                q_label = f"q{q+1:02d}"
                row[f"true_{q_label}"]     = float(y_true[j, q])
                row[f"pred_{q_label}"]     = float(y_pred[j, q])
                row[f"forecast_{q_label}"] = float(y_fc[j, q]) \
                    if not np.isnan(y_fc[j, q]) else None
            all_preds.append(row)

        # Etendre le train
        X_train = np.vstack([X_train, X_pred])
        y_train = np.vstack([y_train, y_true])

        print(f"  Iteration {iter_num:>2}/{n_iters}  "
              f"(train={X_train.shape[0]:,} jours, "
              f"predit jours {i+1}-{i_end})")

    # Construire le DataFrame
    n_targets = y_wf_all.shape[1]
    schema    = {
        "timestamp_day": pl.Datetime("us", "UTC"),
        "split":         pl.Utf8,
        "model":         pl.Utf8,
        "iter":          pl.Int32,
    }
    for q in range(1, n_targets + 1):
        q_label = f"q{q:02d}"
        schema[f"true_{q_label}"]     = pl.Float64
        schema[f"pred_{q_label}"]     = pl.Float64
        schema[f"forecast_{q_label}"] = pl.Float64

    return pl.DataFrame(all_preds, schema=schema)


if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Chargement ────────────────────────────────────────────────────────
    print("Chargement des features...")
    df = pl.read_parquet(FILE_FEAT).sort("timestamp_day")

    feat_cols   = [c for c in df.columns if c.startswith("feat_")]
    target_cols = sorted([c for c in df.columns if c.startswith("target_")])

    oiken     = pl.read_parquet(FILE_OIKEN_CLEAN).sort("timestamp")
    fc_values = oiken["forecast_load"].to_list()
    fc_ts     = {ts: i for i, ts in enumerate(oiken["timestamp"].to_list())}

    def extract_forecast(ts_days: list) -> np.ndarray:
        rows = []
        for ts_day in ts_days:
            if ts_day not in fc_ts:
                rows.append([np.nan] * 96)
                continue
            idx = fc_ts[ts_day]
            rows.append(fc_values[idx:idx + 96])
        return np.array(rows, dtype=float)

    df_train_init = df.filter(pl.col("split") == "train")
    df_wf         = df.filter(pl.col("split").is_in(["val", "test"]))

    X_train_init = df_train_init.select(feat_cols).to_numpy()
    y_train_init = df_train_init.select(target_cols).to_numpy()
    X_wf_all     = df_wf.select(feat_cols).to_numpy()
    y_wf_all     = df_wf.select(target_cols).to_numpy()
    days_wf      = df_wf["timestamp_day"].to_list()
    splits_wf    = df_wf["split"].to_list()

    n_wf = df_wf.shape[0]
    print(f"Train initial : {df_train_init.shape[0]:,} jours")
    print(f"Walk-forward  : {n_wf:,} jours | pas={WF_STEP} jours "
          f"| {(n_wf + WF_STEP - 1) // WF_STEP} iterations")

    # ── LightGBM ──────────────────────────────────────────────────────────
    lgb_base = lgb.LGBMRegressor(**LGB_PARAMS)
    df_lgb   = run_walkforward(
        "lightgbm", lgb_base,
        X_train_init, y_train_init,
        X_wf_all, y_wf_all,
        days_wf, splits_wf, extract_forecast,
    )

    # ── XGBoost ───────────────────────────────────────────────────────────
    xgb_base = xgb.XGBRegressor(**XGB_PARAMS)
    df_xgb   = run_walkforward(
        "xgboost", xgb_base,
        X_train_init, y_train_init,
        X_wf_all, y_wf_all,
        days_wf, splits_wf, extract_forecast,
    )

    # ── Concatener et sauvegarder ─────────────────────────────────────────
    df_all = pl.concat([df_lgb, df_xgb])
    df_all.write_parquet(MODELS_DIR / "wf_predictions_lgb_xgb_v1.parquet")

    # ── Metriques ─────────────────────────────────────────────────────────
    true_cols     = [c for c in df_all.columns if c.startswith("true_")]
    pred_cols     = [c for c in df_all.columns if c.startswith("pred_")]
    forecast_cols = [c for c in df_all.columns if c.startswith("forecast_")]

    all_metrics = []
    print(f"\n{'='*60}")
    print("RESULTATS WALK-FORWARD LGB + XGB")
    print(f"{'='*60}")

    for model_name in ["lightgbm", "xgboost"]:
        for split in ["val", "test"]:
            sub    = df_all.filter(
                (pl.col("model") == model_name) &
                (pl.col("split") == split)
            )
            y_true = sub.select(true_cols).to_numpy()
            y_pred = sub.select(pred_cols).to_numpy()
            y_fc   = sub.select(forecast_cols).to_numpy()

            print(f"\n  {model_name} — {split} :")
            m = compute_metrics(y_true, y_pred, model_name)
            m["split"] = split
            all_metrics.append(m)

            if model_name == "lightgbm":
                mfc = compute_metrics(y_true, y_fc, "oiken_forecast")
                mfc["split"] = split

    # ── Comparaison finale ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("COMPARAISON COMPLETE")
    print(f"{'='*60}")

    wf_rf = pl.read_parquet(MODELS_DIR / "wf_results_v1.parquet")
    simple = pl.read_parquet(MODELS_DIR / "results_v1.parquet").drop("label") \
               if "label" in pl.read_parquet(MODELS_DIR / "results_v1.parquet").columns \
               else pl.read_parquet(MODELS_DIR / "results_v1.parquet")

    df_metrics = pl.DataFrame(all_metrics)
    df_metrics.write_parquet(MODELS_DIR / "wf_results_lgb_xgb_v1.parquet")

    print("\n  Simple validation :")
    print(simple.select(["model", "split", "mae", "rmse"])
                .sort(["split", "mae"]))

    print("\n  Walk-forward (RF pas=6, LGB/XGB pas=30) :")
    combined = pl.concat([
        wf_rf.select(["model", "split", "mae", "rmse"]),
        df_metrics.select(["model", "split", "mae", "rmse"]),
    ])
    print(combined.sort(["split", "mae"]))