"""
Inference — v6 golden dataset
================================
Lit  : golden_data/features_v6_golden.parquet
       data/models/lightgbm_v6.pkl
       data/models/xgboost_v6.pkl
Ecrit: golden_data/predictions_v6_golden.parquet
       golden_data/results_v6_golden.parquet

Evalue LightGBM et XGBoost sur les golden data
Comparaison avec Oiken forecast
"""

import pickle
import datetime
import polars as pl
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import DATA_DIR

GOLDEN_DIR = DATA_DIR.parent / "golden_data"
MODELS_DIR = DATA_DIR / "models"

FILE_FEAT  = GOLDEN_DIR / "features_v6_golden.parquet"
FILE_PREDS = GOLDEN_DIR / "predictions_v6_golden.parquet"
FILE_RES   = GOLDEN_DIR / "results_v6_golden.parquet"

TZ_LOCAL = "Europe/Zurich"

# Periode buguee Oiken (dans les donnees d'entrainement, pas dans golden)
# On la garde au cas ou elle apparaitrait dans les golden
BUG_START = datetime.datetime(2025, 9, 13,  2, 15, 0,
                               tzinfo=datetime.timezone.utc)
BUG_END   = datetime.datetime(2025, 9, 17,  2,  0, 0,
                               tzinfo=datetime.timezone.utc)


def build_oiken_mask(timestamps: list) -> np.ndarray:
    mask = []
    for ts in timestamps:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=datetime.timezone.utc)
        mask.append(not (BUG_START <= ts <= bug_end))
    return np.array(mask)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    label: str, mask: np.ndarray = None) -> dict:
    m = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask is not None:
        m = m & mask
    y_true = y_true[m]
    y_pred = y_pred[m]
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"  {label:<30} MAE={mae:.5f}  RMSE={rmse:.5f}  (n={len(y_true):,})")
    return {"model": label, "mae": mae, "rmse": rmse}


def compute_monthly_metrics(df: pl.DataFrame,
                             model_name: str) -> pl.DataFrame:
    """Calcule MAE par mois pour un modele donne."""
    return (
        df.filter(pl.col("model") == model_name)
          .with_columns([
              pl.col("timestamp")
                .dt.convert_time_zone(TZ_LOCAL)
                .dt.strftime("%Y-%m")
                .alias("month"),
              (pl.col("true") - pl.col("pred")).abs().alias("err_model"),
              (pl.col("true") - pl.col("forecast")).abs().alias("err_oiken"),
          ])
          .group_by("month")
          .agg([
              pl.col("err_model").mean().alias("mae_model"),
              pl.col("err_oiken").mean().alias("mae_oiken"),
              pl.len().alias("n_pts"),
          ])
          .with_columns([
              ((pl.col("mae_model") - pl.col("mae_oiken")) /
               pl.col("mae_oiken") * 100).alias("diff_pct")
          ])
          .sort("month")
    )


if __name__ == "__main__":
    GOLDEN_DIR.mkdir(exist_ok=True)

    # ── Chargement features ───────────────────────────────────────────────
    print("Chargement features golden...")
    df = pl.read_parquet(FILE_FEAT).sort("timestamp")

    feat_cols = [c for c in df.columns
                 if c not in ("timestamp", "date_locale", "target")]

    X      = df.select(feat_cols).to_numpy()
    y_true = df["target"].to_numpy()
    ts     = df["timestamp"].to_list()

    print(f"  {df.shape[0]:,} points")
    print(f"  {len(feat_cols)} features")
    print(f"  Periode : {df['date_locale'].min()} -> "
          f"{df['date_locale'].max()}")

    # Forecast Oiken depuis les features golden
    oiken_golden = pl.read_parquet(
        GOLDEN_DIR / "oiken_golden_clean.parquet"
    ).sort("timestamp")

    fc_map = {t: fc for t, fc in zip(
        oiken_golden["timestamp"].to_list(),
        oiken_golden["forecast_load"].to_list()
    )}
    y_fc = np.array([fc_map.get(t, np.nan) for t in ts])

    # ── Chargement modeles ────────────────────────────────────────────────
    print("\nChargement des modeles v6...")
    models = {}
    for name in ["lightgbm", "xgboost"]:
        path = MODELS_DIR / f"{name}_v6.pkl"
        if path.exists():
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
            print(f"  {name} charge")
        else:
            print(f"  ATTENTION : {path} introuvable")

    # ── Baseline Oiken ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  BASELINE OIKEN FORECAST — GOLDEN")
    print(f"{'='*60}")
    fc_metrics = compute_metrics(y_true, y_fc, "oiken_forecast")
    fc_metrics["model"] = "oiken_forecast"

    # ── Inference ─────────────────────────────────────────────────────────
    all_metrics = [fc_metrics]
    all_preds   = []

    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"  {name.upper()} — GOLDEN")
        print(f"{'='*60}")

        y_pred = model.predict(X)
        m      = compute_metrics(y_true, y_pred, name)
        m["model"] = name
        all_metrics.append(m)

        # Comparaison vs Oiken
        diff = (m["mae"] - fc_metrics["mae"]) / fc_metrics["mae"] * 100
        sign = "▼" if diff < 0 else "▲"
        print(f"  vs Oiken : {sign} {abs(diff):.1f}%")

        all_preds.append(pl.DataFrame({
            "timestamp": ts,
            "model":     [name] * len(ts),
            "true":      y_true.tolist(),
            "pred":      y_pred.tolist(),
            "forecast":  y_fc.tolist(),
        }).with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
        ))

    # ── Metriques par mois ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  METRIQUES PAR MOIS")
    print(f"{'='*60}")

    df_preds = pl.concat(all_preds)

    for name in models:
        print(f"\n  {name.upper()} :")
        monthly = compute_monthly_metrics(df_preds, name)
        print(f"  {'Mois':<10} {'MAE model':>10} {'MAE Oiken':>10} "
              f"{'Diff':>8} {'N pts':>8}")
        print(f"  {'-'*50}")
        for row in monthly.iter_rows(named=True):
            sign = "▼" if row["diff_pct"] < 0 else "▲"
            print(f"  {row['month']:<10} "
                  f"{row['mae_model']:>10.5f} "
                  f"{row['mae_oiken']:>10.5f} "
                  f"{sign}{abs(row['diff_pct']):>6.1f}% "
                  f"{row['n_pts']:>8,}")

        # Moyenne globale
        mae_m = monthly["mae_model"].mean()
        mae_o = monthly["mae_oiken"].mean()
        diff  = (mae_m - mae_o) / mae_o * 100
        sign  = "▼" if diff < 0 else "▲"
        print(f"  {'MOYENNE':<10} "
              f"{mae_m:>10.5f} "
              f"{mae_o:>10.5f} "
              f"{sign}{abs(diff):>6.1f}%")

        n_better = (monthly["diff_pct"] < 0).sum()
        n_total  = monthly.shape[0]
        print(f"  Mois meilleurs qu'Oiken : {n_better}/{n_total}")

    # ── Sauvegarde ────────────────────────────────────────────────────────
    df_preds.write_parquet(FILE_PREDS)

    df_results = pl.DataFrame(all_metrics)
    df_results.write_parquet(FILE_RES)

    # ── Resume final ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESUME FINAL — GOLDEN")
    print(f"{'='*60}")
    print(df_results.select(["model", "mae", "rmse"]).sort("mae"))

    print(f"\nFichiers sauvegardes :")
    for p in [FILE_PREDS, FILE_RES]:
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  {p.name:<45} {size_mb:.1f} MB")