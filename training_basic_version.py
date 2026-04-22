"""
Entrainement modeles — version basique
========================================
Lit  : data/features_v1.parquet
       data/oiken_clean.parquet
Ecrit: data/models/results_v1.parquet
       data/models/predictions_v1.parquet

Modeles : RandomForest, LightGBM, XGBoost (multi-output)
Baseline: forecast_load Oiken
Metriques : MAE, RMSE, MAPE par quart d'heure et global
"""

import pickle
from pathlib import Path

import polars as pl
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import xgboost as xgb

from config import DATA_DIR, FILE_OIKEN_CLEAN

MODELS_DIR = DATA_DIR / "models"
FILE_FEAT  = DATA_DIR / "features_v1.parquet"


def load_data() -> tuple:
    df      = pl.read_parquet(FILE_FEAT)
    oiken   = pl.read_parquet(FILE_OIKEN_CLEAN).sort("timestamp")

    feat_cols   = [c for c in df.columns if c.startswith("feat_")]
    target_cols = sorted([c for c in df.columns if c.startswith("target_")])

    train = df.filter(pl.col("split") == "train")
    val   = df.filter(pl.col("split") == "val")
    test  = df.filter(pl.col("split") == "test")

    X_train = train.select(feat_cols).to_numpy()
    y_train = train.select(target_cols).to_numpy()
    X_val   = val.select(feat_cols).to_numpy()
    y_val   = val.select(target_cols).to_numpy()
    X_test  = test.select(feat_cols).to_numpy()
    y_test  = test.select(target_cols).to_numpy()

    print(f"Train : {X_train.shape[0]:,} jours")
    print(f"Val   : {X_val.shape[0]:,} jours")
    print(f"Test  : {X_test.shape[0]:,} jours")
    print(f"Features  : {X_train.shape[1]}")
    print(f"Targets   : {y_train.shape[1]}")

    ts_val  = val["timestamp_day"].to_list()
    ts_test = test["timestamp_day"].to_list()

    # ── Baseline Oiken : forecast_load aligne sur les jours ───────────────
    # Pour chaque jour J du split, extraire les 96 valeurs de forecast_load
    forecast_vals = oiken["forecast_load"].to_list()
    timestamps    = oiken["timestamp"].to_list()
    ts_to_idx     = {ts: i for i, ts in enumerate(timestamps)}

    def extract_forecast(ts_days: list) -> np.ndarray:
        rows = []
        for ts_day in ts_days:
            # Trouver l'index du premier quart d'heure du jour J
            if ts_day not in ts_to_idx:
                rows.append([np.nan] * 96)
                continue
            idx = ts_to_idx[ts_day]
            rows.append(forecast_vals[idx:idx + 96])
        return np.array(rows, dtype=float)

    y_forecast_val  = extract_forecast(ts_val)
    y_forecast_test = extract_forecast(ts_test)

    print(f"\nBaseline Oiken forecast :")
    print(f"  NaN val  : {np.isnan(y_forecast_val).sum()}")
    print(f"  NaN test : {np.isnan(y_forecast_test).sum()}")

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            ts_val, ts_test, target_cols,
            y_forecast_val, y_forecast_test)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    label: str) -> dict:
    # Exclure les lignes avec NaN dans y_pred ou y_true
    mask_rows = ~(np.isnan(y_true).any(axis=1) |
                  np.isnan(y_pred).any(axis=1))
    y_true = y_true[mask_rows]
    y_pred = y_pred[mask_rows]

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAPE en evitant la division par zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    print(f"  {label:<25} MAE={mae:.5f}  RMSE={rmse:.5f}  MAPE={mape:.2f}%")
    return {"label": label, "mae": mae, "rmse": rmse, "mape": mape}


def train_random_forest(X_train, y_train, X_val, y_val,
                        X_test, y_test) -> tuple:
    print("\n" + "="*60)
    print("  RANDOM FOREST")
    print("="*60)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )

    print("  Entrainement...")
    model.fit(X_train, y_train)

    print("  Evaluation :")
    metrics_val  = compute_metrics(y_val,  model.predict(X_val),  "val")
    metrics_test = compute_metrics(y_test, model.predict(X_test), "test")

    return model, metrics_val, metrics_test


def train_lightgbm(X_train, y_train, X_val, y_val,
                   X_test, y_test) -> tuple:
    print("\n" + "="*60)
    print("  LIGHTGBM")
    print("="*60)

    base = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=63,
        min_child_samples=10,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    model = MultiOutputRegressor(base, n_jobs=-1)

    print("  Entrainement...")
    model.fit(X_train, y_train)

    print("  Evaluation :")
    metrics_val  = compute_metrics(y_val,  model.predict(X_val),  "val")
    metrics_test = compute_metrics(y_test, model.predict(X_test), "test")

    return model, metrics_val, metrics_test


def train_xgboost(X_train, y_train, X_val, y_val,
                  X_test, y_test) -> tuple:
    print("\n" + "="*60)
    print("  XGBOOST")
    print("="*60)

    base = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )
    model = MultiOutputRegressor(base, n_jobs=-1)

    print("  Entrainement...")
    model.fit(X_train, y_train)

    print("  Evaluation :")
    metrics_val  = compute_metrics(y_val,  model.predict(X_val),  "val")
    metrics_test = compute_metrics(y_test, model.predict(X_test), "test")

    return model, metrics_val, metrics_test


def save_predictions(model, model_name: str,
                     X_val, y_val, ts_val,
                     X_test, y_test, ts_test,
                     y_forecast_val, y_forecast_test,
                     target_cols: list) -> pl.DataFrame:
    records = []
    for split, X, y_true, y_fc, timestamps in [
        ("val",  X_val,  y_val,  y_forecast_val,  ts_val),
        ("test", X_test, y_test, y_forecast_test, ts_test),
    ]:
        y_pred = model.predict(X)
        for i, ts in enumerate(timestamps):
            row = {
                "timestamp_day": ts,
                "split":         split,
                "model":         model_name,
            }
            for q, col in enumerate(target_cols):
                q_label = col.replace("target_", "")
                row[f"true_{q_label}"]     = float(y_true[i, q])
                row[f"pred_{q_label}"]     = float(y_pred[i, q])
                row[f"forecast_{q_label}"] = float(y_fc[i, q]) \
                    if not np.isnan(y_fc[i, q]) else None
            records.append(row)

    schema = {
        "timestamp_day": pl.Datetime("us", "UTC"),
        "split":         pl.Utf8,
        "model":         pl.Utf8,
    }
    for col in target_cols:
        q_label = col.replace("target_", "")
        schema[f"true_{q_label}"]     = pl.Float64
        schema[f"pred_{q_label}"]     = pl.Float64
        schema[f"forecast_{q_label}"] = pl.Float64

    return pl.DataFrame(records, schema=schema)


if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Chargement ────────────────────────────────────────────────────────
    print("Chargement des features...")
    (X_train, y_train, X_val, y_val, X_test, y_test,
     ts_val, ts_test, target_cols,
     y_forecast_val, y_forecast_test) = load_data()

    # ── Baseline Oiken ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  BASELINE OIKEN FORECAST")
    print("="*60)
    print("  Evaluation :")
    fc_val  = compute_metrics(y_val,  y_forecast_val,  "val")
    fc_test = compute_metrics(y_test, y_forecast_test, "test")

    all_metrics = [
        {"model": "oiken_forecast", "split": "val",  **fc_val},
        {"model": "oiken_forecast", "split": "test", **fc_test},
    ]
    all_preds = []

    # ── Random Forest ─────────────────────────────────────────────────────
    rf_model, rf_val, rf_test = train_random_forest(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    with open(MODELS_DIR / "random_forest_v1.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    all_metrics.append({"model": "random_forest", "split": "val",  **rf_val})
    all_metrics.append({"model": "random_forest", "split": "test", **rf_test})
    all_preds.append(save_predictions(
        rf_model, "random_forest",
        X_val, y_val, ts_val,
        X_test, y_test, ts_test,
        y_forecast_val, y_forecast_test, target_cols
    ))

    # ── LightGBM ──────────────────────────────────────────────────────────
    lgb_model, lgb_val, lgb_test = train_lightgbm(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    with open(MODELS_DIR / "lightgbm_v1.pkl", "wb") as f:
        pickle.dump(lgb_model, f)
    all_metrics.append({"model": "lightgbm", "split": "val",  **lgb_val})
    all_metrics.append({"model": "lightgbm", "split": "test", **lgb_test})
    all_preds.append(save_predictions(
        lgb_model, "lightgbm",
        X_val, y_val, ts_val,
        X_test, y_test, ts_test,
        y_forecast_val, y_forecast_test, target_cols
    ))

    # ── XGBoost ───────────────────────────────────────────────────────────
    xgb_model, xgb_val, xgb_test = train_xgboost(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    with open(MODELS_DIR / "xgboost_v1.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
    all_metrics.append({"model": "xgboost", "split": "val",  **xgb_val})
    all_metrics.append({"model": "xgboost", "split": "test", **xgb_test})
    all_preds.append(save_predictions(
        xgb_model, "xgboost",
        X_val, y_val, ts_val,
        X_test, y_test, ts_test,
        y_forecast_val, y_forecast_test, target_cols
    ))

    # ── Sauvegarde resultats ──────────────────────────────────────────────
    df_metrics = pl.DataFrame(all_metrics)
    df_metrics.write_parquet(MODELS_DIR / "results_v1.parquet")

    df_preds = pl.concat(all_preds)
    df_preds.write_parquet(MODELS_DIR / "predictions_v1.parquet")

    # ── Resume final ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESUME FINAL")
    print(f"{'='*60}")
    print(df_metrics.select(["model", "split", "mae", "rmse", "mape"])
                    .sort(["split", "mae"]))

    print(f"\nFichiers sauvegardes dans {MODELS_DIR} :")
    for p in sorted(MODELS_DIR.glob("*")):
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  {p.name:<40} {size_mb:.1f} MB")