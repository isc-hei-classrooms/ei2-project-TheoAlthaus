"""
Walk-forward validation — version basique
==========================================
Lit  : data/features_v1.parquet
       data/oiken_clean.parquet
Ecrit: data/models/wf_results_v1.parquet
       data/models/wf_predictions_v1.parquet

Principe :
  - Fenetre expansive : on ajoute 6 jours au train a chaque iteration
  - On predit les 6 jours suivants
  - Modele : Random Forest uniquement (multi-output natif)
  - Comparaison avec Oiken forecast
"""

import pickle
from pathlib import Path

import polars as pl
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import DATA_DIR, FILE_OIKEN_CLEAN

MODELS_DIR  = DATA_DIR / "models"
FILE_FEAT   = DATA_DIR / "features_v1.parquet"
WF_STEP     = 6   # reintrainement tous les 6 jours

RF_PARAMS = dict(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42,
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
    return {"label": label, "mae": mae, "rmse": rmse}


if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Chargement ────────────────────────────────────────────────────────
    print("Chargement des features...")
    df = pl.read_parquet(FILE_FEAT).sort("timestamp_day")

    feat_cols   = [c for c in df.columns if c.startswith("feat_")]
    target_cols = sorted([c for c in df.columns if c.startswith("target_")])

    # Oiken forecast
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

    # Indices train initial et walk-forward
    df_train_init = df.filter(pl.col("split") == "train")
    df_wf         = df.filter(pl.col("split").is_in(["val", "test"]))

    n_wf     = df_wf.shape[0]
    days_wf  = df_wf["timestamp_day"].to_list()
    X_wf_all = df_wf.select(feat_cols).to_numpy()
    y_wf_all = df_wf.select(target_cols).to_numpy()

    print(f"Train initial : {df_train_init.shape[0]:,} jours")
    print(f"Walk-forward  : {n_wf:,} jours ({n_wf // WF_STEP} iterations)")

    # ── Walk-forward ──────────────────────────────────────────────────────
    print(f"\nDemarrage walk-forward (pas={WF_STEP} jours)...")

    X_train_wf = df_train_init.select(feat_cols).to_numpy()
    y_train_wf = df_train_init.select(target_cols).to_numpy()

    all_preds   = []
    n_iters     = (n_wf + WF_STEP - 1) // WF_STEP

    for i in range(0, n_wf, WF_STEP):
        iter_num = i // WF_STEP + 1
        i_end    = min(i + WF_STEP, n_wf)

        # Entrainer sur train initial + tout ce qu'on a vu jusqu'ici
        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(X_train_wf, y_train_wf)

        # Predire les WF_STEP prochains jours
        X_pred  = X_wf_all[i:i_end]
        y_true  = y_wf_all[i:i_end]
        y_pred  = model.predict(X_pred)
        ts_pred = days_wf[i:i_end]

        # Split (val ou test)
        splits = df_wf["split"].to_list()[i:i_end]

        # Forecast Oiken pour ces jours
        y_fc = extract_forecast(ts_pred)

        # Sauvegarder les predictions
        for j in range(len(ts_pred)):
            row = {
                "timestamp_day": ts_pred[j],
                "split":         splits[j],
                "iter":          iter_num,
            }
            for q, col in enumerate(target_cols):
                q_label = col.replace("target_", "")
                row[f"true_{q_label}"]     = float(y_true[j, q])
                row[f"pred_{q_label}"]     = float(y_pred[j, q])
                row[f"forecast_{q_label}"] = float(y_fc[j, q]) \
                    if not np.isnan(y_fc[j, q]) else None
            all_preds.append(row)

        # Etendre le train avec les jours predits
        X_train_wf = np.vstack([X_train_wf, X_pred])
        y_train_wf = np.vstack([y_train_wf, y_true])

        if iter_num % 10 == 0 or iter_num == n_iters:
            print(f"  Iteration {iter_num:>3}/{n_iters}  "
                  f"(train={X_train_wf.shape[0]:,} jours, "
                  f"predit jours {i+1}-{i_end})")

    # ── Construire le DataFrame de predictions ────────────────────────────
    print("\nConstruction du DataFrame de predictions...")
    schema = {
        "timestamp_day": pl.Datetime("us", "UTC"),
        "split":         pl.Utf8,
        "iter":          pl.Int32,
    }
    for col in target_cols:
        q_label = col.replace("target_", "")
        schema[f"true_{q_label}"]     = pl.Float64
        schema[f"pred_{q_label}"]     = pl.Float64
        schema[f"forecast_{q_label}"] = pl.Float64

    df_preds = pl.DataFrame(all_preds, schema=schema)
    df_preds.write_parquet(MODELS_DIR / "wf_predictions_v1.parquet")

    # ── Metriques ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESULTATS WALK-FORWARD")
    print(f"{'='*60}")

    true_cols     = [c for c in df_preds.columns if c.startswith("true_")]
    pred_cols     = [c for c in df_preds.columns if c.startswith("pred_")]
    forecast_cols = [c for c in df_preds.columns if c.startswith("forecast_")]

    all_metrics = []
    for split in ["val", "test"]:
        sub = df_preds.filter(pl.col("split") == split)
        y_true = sub.select(true_cols).to_numpy()
        y_pred = sub.select(pred_cols).to_numpy()
        y_fc   = sub.select(forecast_cols).to_numpy()

        print(f"\n  Split : {split}")
        m_wf = compute_metrics(y_true, y_pred, "walk-forward RF")
        m_fc = compute_metrics(y_true, y_fc,   "oiken forecast")

        all_metrics.append({"model": "wf_random_forest",
                             "split": split, **m_wf})
        all_metrics.append({"model": "oiken_forecast",
                             "split": split, **m_fc})

    # ── Comparaison avec validation simple ────────────────────────────────
    print(f"\n{'='*60}")
    print("COMPARAISON SIMPLE vs WALK-FORWARD")
    print(f"{'='*60}")

    results_simple = pl.read_parquet(MODELS_DIR / "results_v1.parquet")
    results_simple = results_simple.filter(
        pl.col("model").is_in(["random_forest", "oiken_forecast"])
    ).select(["model", "split", "mae", "rmse"])

    df_wf_metrics = pl.DataFrame(all_metrics).drop("label")

    print("\n  Validation simple :")
    print(results_simple.sort(["split", "mae"]))

    print("\n  Walk-forward :")
    print(df_wf_metrics.select(["model", "split", "mae", "rmse"])
                       .sort(["split", "mae"]))

    # Sauvegarde
    df_wf_metrics.write_parquet(MODELS_DIR / "wf_results_v1.parquet")

    print(f"\nFichiers sauvegardes dans {MODELS_DIR}")