"""
Feature engineering — version 4
===================================
"""

import datetime
import polars as pl
from zoneinfo import ZoneInfo

from config import (
    DATA_DIR, FILE_OIKEN_CLEAN, FILE_CALENDAR,
    FILE_METEO_HIST_CLEAN, FILE_METEO_PRED_CLEAN,
)

TZ_LOCAL        = "Europe/Zurich"
TRAIN_END_LOCAL = datetime.date(2024,  6, 30)
VAL_END_LOCAL   = datetime.date(2024, 12, 31)
FILE_OUT        = DATA_DIR / "features_v4b.parquet"

CAL_COLS = [
    "hour_sin", "hour_cos",
    "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos",
    "day_of_year_sin", "day_of_year_cos",
    "is_weekend", "is_holiday", "is_school_holiday",
]

METEO_HIST_COLS = [
    "hist_temperature", "hist_radiation",
    "hist_sunshine",    "hist_humidity",
]

METEO_PRED_VARS = [
    "pred_temperature", "pred_radiation",
    "pred_sunshine",    "pred_humidity",
]

SUBTYPES  = ["ctrl", "q10", "q90", "stde"]
PRED_LAGS = list(range(1, 34))


def add_local_time(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("timestamp").dt.convert_time_zone(TZ_LOCAL).alias("ts_local")
    ).with_columns([
        pl.col("ts_local").dt.date().alias("date_locale"),
        pl.col("ts_local").dt.hour().alias("hour_local"),
        pl.col("ts_local").dt.minute().alias("minute_local"),
    ])


def min_available_lag(hour_local: int, minute_local: int) -> float:
    """
    lag_min = 13 + heure_locale_cible
    Lags disponibles : lag >= lag_min
    """
    return (24 - 11) + hour_local + minute_local / 60


def get_normal_days(df: pl.DataFrame) -> list:
    pts_per_day = (
        df.group_by("date_locale")
          .agg(pl.len().alias("n_pts"))
          .sort("date_locale")
    )
    normal   = pts_per_day.filter(pl.col("n_pts") == 96)["date_locale"].to_list()
    abnormal = pts_per_day.filter(pl.col("n_pts") != 96)
    print(f"  Jours normaux   : {len(normal):,}")
    print(f"  Jours ignores   :")
    for row in abnormal.iter_rows(named=True):
        print(f"    {row['date_locale']}  ->  {row['n_pts']} pts")
    return normal


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)

    # ── Chargement ────────────────────────────────────────────────────────
    print("Chargement des donnees...")
    oiken      = pl.read_parquet(FILE_OIKEN_CLEAN).sort("timestamp")
    cal        = pl.read_parquet(FILE_CALENDAR).sort("timestamp")
    meteo_hist = pl.read_parquet(FILE_METEO_HIST_CLEAN("sion")).sort("timestamp")
    meteo_pred = pl.read_parquet(FILE_METEO_PRED_CLEAN("sion")) \
                   .sort("timestamp_target") \
                   .rename({"timestamp_target": "timestamp"})

    print(f"  Oiken      : {oiken.shape[0]:,} lignes")
    print(f"  Calendrier : {cal.shape[0]:,} lignes")
    print(f"  Meteo hist : {meteo_hist.shape[0]:,} lignes")
    print(f"  Meteo pred : {meteo_pred.shape[0]:,} lignes x {meteo_pred.shape[1]} col")

    # ── Heure locale ──────────────────────────────────────────────────────
    print("\nAjout heure locale...")
    oiken = add_local_time(oiken)

    print("\nAnalyse des jours normaux :")
    normal_days   = get_normal_days(oiken)
    normal_days_s = set(normal_days)
    oiken         = oiken.filter(pl.col("date_locale").is_in(normal_days))

    # ── Index load ────────────────────────────────────────────────────────
    print("\nConstruction index load...")
    load_index = {
        (row["date_locale"], row["hour_local"], row["minute_local"]): row["load"]
        for row in oiken.select([
            "date_locale", "hour_local", "minute_local", "load"
        ]).iter_rows(named=True)
    }
    print(f"  {len(load_index):,} entrees")

    # ── Index meteo historique ────────────────────────────────────────────
    print("Construction index meteo historique...")
    meteo_hist    = add_local_time(meteo_hist)
    meteo_hist_idx = {}
    for row in meteo_hist.select(
        ["date_locale", "hour_local", "minute_local"] + METEO_HIST_COLS
    ).iter_rows(named=True):
        key = (row["date_locale"], row["hour_local"], row["minute_local"])
        meteo_hist_idx[key] = {col: row[col] for col in METEO_HIST_COLS}
    print(f"  {len(meteo_hist_idx):,} entrees")

    # ── Previsions meteo ──────────────────────────────────────────────────
    print("\nPreparation previsions meteo...")
    pred_cols_to_use = [
        f"lag_{lag:02d}h_{var}_{subtype}"
        for lag in PRED_LAGS
        for var in METEO_PRED_VARS
        for subtype in SUBTYPES
    ]
    pred_cols_to_use = [c for c in pred_cols_to_use if c in meteo_pred.columns]
    print(f"  Colonnes prevision : {len(pred_cols_to_use)}")

    print("  Interpolation 1h -> 15min...")
    meteo_pred_interp = meteo_pred.select(
        ["timestamp"] + pred_cols_to_use
    ).with_columns([
        pl.col(c).interpolate() for c in pred_cols_to_use
    ])

    pred_index = {
        row["timestamp"]: {c: row[c] for c in pred_cols_to_use}
        for row in meteo_pred_interp.iter_rows(named=True)
    }
    print(f"  {len(pred_index):,} entrees")

    # ── Index calendrier ──────────────────────────────────────────────────
    print("Construction index calendrier...")
    cal_index = {
        row["timestamp"]: {c: row[c] for c in CAL_COLS}
        for row in cal.select(["timestamp"] + CAL_COLS).iter_rows(named=True)
    }

    # ── Construction des features ─────────────────────────────────────────
    print("\nConstruction des features...")

    records   = []
    n_skipped = 0

    for row in oiken.select([
        "timestamp", "date_locale", "hour_local", "minute_local", "load"
    ]).iter_rows(named=True):

        ts_utc  = row["timestamp"]
        date_j  = row["date_locale"]
        h       = row["hour_local"]
        m       = row["minute_local"]

        date_j2 = date_j - datetime.timedelta(days=2)
        date_j7 = date_j - datetime.timedelta(days=7)

        if date_j2 not in normal_days_s or date_j7 not in normal_days_s:
            n_skipped += 1
            continue

        key_j2 = (date_j2, h, m)
        key_j7 = (date_j7, h, m)

        if key_j2 not in load_index or key_j7 not in load_index:
            n_skipped += 1
            continue

        rec = {
            "timestamp":   ts_utc,
            "date_locale": date_j,
            "target":      row["load"],
        }

        # Calendaire
        cal_vals = cal_index.get(ts_utc, {})
        for col in CAL_COLS:
            rec[col] = cal_vals.get(col, None)

        # Load lags
        rec["feat_load_lag2d"] = load_index[key_j2]
        rec["feat_load_lag7d"] = load_index[key_j7]

        # Meteo hist lags
        meteo_j2 = meteo_hist_idx.get(key_j2, {})
        meteo_j7 = meteo_hist_idx.get(key_j7, {})
        for col in METEO_HIST_COLS:
            short = col.replace("hist_", "")
            rec[f"feat_{short}_lag2d"] = meteo_j2.get(col, None)
            rec[f"feat_{short}_lag7d"] = meteo_j7.get(col, None)

        # Previsions meteo avec contrainte causalite
        lag_min   = min_available_lag(h, m)
        pred_vals = pred_index.get(ts_utc, {})

        for lag in PRED_LAGS:
            for var in METEO_PRED_VARS:
                for subtype in SUBTYPES:
                    col = f"lag_{lag:02d}h_{var}_{subtype}"
                    if col not in pred_cols_to_use:
                        continue
                    rec[col] = pred_vals.get(col, None) \
                               if lag >= lag_min else None

        records.append(rec)

    print(f"  Points construits : {len(records):,}")
    print(f"  Points ignores    : {n_skipped:,}")

    # ── DataFrame ─────────────────────────────────────────────────────────
    print("\nConstruction du DataFrame...")
    feat_cols_all = (
        CAL_COLS
        + ["feat_load_lag2d", "feat_load_lag7d"]
        + [f"feat_{c.replace('hist_', '')}_lag2d" for c in METEO_HIST_COLS]
        + [f"feat_{c.replace('hist_', '')}_lag7d" for c in METEO_HIST_COLS]
        + pred_cols_to_use
    )

    schema = {
        "timestamp":   pl.Datetime("us", "UTC"),
        "date_locale": pl.Date,
        "target":      pl.Float64,
    }
    for col in CAL_COLS:
        schema[col] = pl.Float64
    schema["feat_load_lag2d"] = pl.Float64
    schema["feat_load_lag7d"] = pl.Float64
    for col in METEO_HIST_COLS:
        short = col.replace("hist_", "")
        schema[f"feat_{short}_lag2d"] = pl.Float64
        schema[f"feat_{short}_lag7d"] = pl.Float64
    for col in pred_cols_to_use:
        schema[col] = pl.Float64

    df_feat = pl.DataFrame(records, schema=schema)

    # ── Split ─────────────────────────────────────────────────────────────
    df_feat = df_feat.with_columns(
        pl.when(pl.col("date_locale") <= pl.lit(TRAIN_END_LOCAL))
          .then(pl.lit("train"))
          .when(pl.col("date_locale") <= pl.lit(VAL_END_LOCAL))
          .then(pl.lit("val"))
          .otherwise(pl.lit("test"))
          .alias("split")
    )

    # ── Verification ──────────────────────────────────────────────────────
    feat_cols = [c for c in df_feat.columns
                 if c not in ("timestamp", "date_locale", "target", "split")]

    print(f"\nShape finale : {df_feat.shape[0]:,} lignes x {df_feat.shape[1]} col")
    print(f"Periode      : {df_feat['date_locale'].min()} -> "
          f"{df_feat['date_locale'].max()}")
    print(f"Features     : {len(feat_cols)}")

    print(f"\nRepartition split :")
    print(df_feat.group_by("split")
                 .agg(pl.len().alias("n_points"))
                 .sort("n_points", descending=True))

    print(f"\nVerification NaN (features cles) :")
    for col in ["feat_load_lag2d", "feat_load_lag7d",
                "feat_temperature_lag2d",
                "lag_13h_pred_temperature_ctrl",
                "lag_20h_pred_temperature_ctrl",
                "lag_33h_pred_temperature_ctrl"]:
        if col in df_feat.columns:
            n   = df_feat[col].is_null().sum()
            pct = n / df_feat.shape[0] * 100
            print(f"  {col:<45} {n:>7,} NaN ({pct:.1f}%)")

    # ── Sauvegarde ────────────────────────────────────────────────────────
    df_feat.write_parquet(FILE_OUT)
    size_mb = FILE_OUT.stat().st_size / 1024 / 1024
    print(f"\n-> {FILE_OUT.name}  "
          f"({df_feat.shape[0]:,} lignes x {df_feat.shape[1]} col, "
          f"{size_mb:.1f} MB)")