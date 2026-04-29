"""
Feature engineering — version 6 golden dataset
=================================================
Lit  : golden_data/oiken_golden_clean.parquet
       golden_data/meteo_sion_hist_golden_clean.parquet
       golden_data/meteo_sion_pred_golden_clean.parquet
       data/calendar.parquet  (calendrier deja calcule sur toute la periode)
Ecrit: golden_data/features_v6_golden.parquet

Identique a feature_v6.py mais :
  - Lit les fichiers golden
  - Pas de split train/val/test
  - Pas de filtre sur jours normaux strict (on garde tous les jours)
  - Periode : 2025-09-30 -> 2026-04-22
"""

import datetime
import polars as pl
from zoneinfo import ZoneInfo

from config import (
    DATA_DIR,
    FILE_OIKEN_CLEAN,
    FILE_CALENDAR,
    FILE_METEO_HIST_CLEAN,
)

GOLDEN_DIR = DATA_DIR.parent / "golden_data"
FILE_OUT   = GOLDEN_DIR / "features_v6_golden.parquet"

TZ_LOCAL = "Europe/Zurich"

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

SUBTYPES  = ["ctrl", "stde"]
PRED_LAGS = list(range(13, 34))

PV_COLS = [
    "pv_central_valais",
    "pv_sierre",
    "pv_sion",
]


def add_local_time(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("timestamp").dt.convert_time_zone(TZ_LOCAL).alias("ts_local")
    ).with_columns([
        pl.col("ts_local").dt.date().alias("date_locale"),
        pl.col("ts_local").dt.hour().alias("hour_local"),
        pl.col("ts_local").dt.minute().alias("minute_local"),
    ])


def min_available_lag(hour_local: int, minute_local: int) -> float:
    return 13 + hour_local + minute_local / 60


def get_normal_days(df: pl.DataFrame) -> tuple[list, list]:
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
    return normal, abnormal["date_locale"].to_list() \
           if abnormal.shape[0] > 0 else []


if __name__ == "__main__":
    GOLDEN_DIR.mkdir(exist_ok=True)

    # ── Chargement ────────────────────────────────────────────────────────
    print("Chargement des donnees...")

    # Oiken golden
    oiken = pl.read_parquet(
        GOLDEN_DIR / "oiken_golden_clean.parquet"
    ).sort("timestamp")

    # Calendrier — deja calcule sur toute la periode dans data/
    # Si le calendrier ne couvre pas la periode golden, il faudra le regenerer
    cal = pl.read_parquet(FILE_CALENDAR).sort("timestamp")

    # Meteo historique golden
    meteo_hist = pl.read_parquet(
        GOLDEN_DIR / "meteo_sion_hist_golden_clean.parquet"
    ).sort("timestamp")

    # Previsions golden
    meteo_pred = pl.read_parquet(
        GOLDEN_DIR / "meteo_sion_pred_golden_clean.parquet"
    ).rename({"timestamp_target": "timestamp"})

    print(f"  Oiken golden : {oiken.shape[0]:,} lignes")
    print(f"  Calendrier   : {cal.shape[0]:,} lignes")
    print(f"  Meteo hist   : {meteo_hist.shape[0]:,} lignes")
    print(f"  Meteo pred   : {meteo_pred.shape[0]:,} lignes x "
          f"{meteo_pred.shape[1]} col")

    # Verifier que le calendrier couvre la periode golden
    cal_max = cal["timestamp"].max()
    oiken_max = oiken["timestamp"].max()
    if cal_max < oiken_max:
        print(f"\n  ATTENTION : calendrier finit a {cal_max}, "
              f"oiken golden finit a {oiken_max}")
        print(f"  Il faudra regenerer le calendrier pour couvrir "
              f"la periode golden.")

    # ── Heure locale ──────────────────────────────────────────────────────
    print("\nAjout heure locale...")
    oiken = add_local_time(oiken)

    # On utilise les donnees d'entrainement pour les lags J-2 et J-7
    # Il faut charger oiken_clean.parquet pour avoir les valeurs historiques
    print("\nChargement oiken_clean pour lags historiques...")
    oiken_train = pl.read_parquet(FILE_OIKEN_CLEAN).sort("timestamp")
    oiken_train = add_local_time(oiken_train)

    # Combiner oiken_train + oiken golden pour avoir acces aux lags
    # On a besoin d'au moins 7 jours avant le debut golden pour les lags J-7
    oiken_full = pl.concat([
        oiken_train,
        oiken.select(oiken_train.columns)
    ]).sort("timestamp").unique(subset=["timestamp"], keep="first")

    print(f"  Oiken complet : {oiken_full.shape[0]:,} lignes")

    # Jours normaux sur oiken_full
    print("\nAnalyse des jours normaux :")
    normal_days, _ = get_normal_days(oiken_full)
    normal_days_s  = set(normal_days)

    # ── Index load + PV (depuis oiken_full) ───────────────────────────────
    print("\nConstruction index load + PV...")
    load_index = {}
    pv_index   = {}

    oiken_full_local = add_local_time(
        oiken_full.select(["timestamp"] + ["load"] + PV_COLS)
    ) if "date_locale" not in oiken_full.columns else oiken_full

    for row in oiken_full.select(
        ["date_locale", "hour_local", "minute_local", "load"] + PV_COLS
    ).iter_rows(named=True):
        key             = (row["date_locale"], row["hour_local"],
                           row["minute_local"])
        load_index[key] = row["load"]
        pv_index[key]   = {col: row[col] for col in PV_COLS}
    print(f"  {len(load_index):,} entrees")

    # ── Index meteo historique ────────────────────────────────────────────
    print("Construction index meteo historique...")

    # Charger aussi meteo_hist d'entrainement pour les lags
    meteo_hist_train = pl.read_parquet(
        FILE_METEO_HIST_CLEAN("sion")
    ).sort("timestamp")

    # Garder uniquement les colonnes communes
    common_cols = [c for c in meteo_hist_train.columns
                   if c in meteo_hist.columns]

    meteo_hist_full = pl.concat([
        meteo_hist_train.select(common_cols),
        meteo_hist.select(common_cols),
    ]).sort("timestamp").unique(subset=["timestamp"], keep="first")

    meteo_hist_full = add_local_time(meteo_hist_full)

    meteo_hist_idx = {}
    for row in meteo_hist_full.select(
        ["date_locale", "hour_local", "minute_local"] + METEO_HIST_COLS
    ).iter_rows(named=True):
        key                 = (row["date_locale"], row["hour_local"],
                               row["minute_local"])
        meteo_hist_idx[key] = {col: row[col] for col in METEO_HIST_COLS}
    print(f"  {len(meteo_hist_idx):,} entrees")

    # ── Previsions meteo golden ───────────────────────────────────────────
    print("\nPreparation previsions meteo golden...")
    pred_cols_to_use = [
        f"lag_{lag:02d}h_{var}_{subtype}"
        for lag in PRED_LAGS
        for var in METEO_PRED_VARS
        for subtype in SUBTYPES
    ]
    pred_cols_to_use = [c for c in pred_cols_to_use
                        if c in meteo_pred.columns]
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
    print("\nConstruction des features golden...")

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

        # PV lags
        pv_j2 = pv_index.get(key_j2, {})
        pv_j7 = pv_index.get(key_j7, {})
        for col in PV_COLS:
            rec[f"feat_{col}_lag2d"] = pv_j2.get(col, None)
            rec[f"feat_{col}_lag7d"] = pv_j7.get(col, None)

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

    schema = {
        "timestamp":   pl.Datetime("us", "UTC"),
        "date_locale": pl.Date,
        "target":      pl.Float64,
    }
    for col in CAL_COLS:
        schema[col] = pl.Float64
    schema["feat_load_lag2d"] = pl.Float64
    schema["feat_load_lag7d"] = pl.Float64
    for col in PV_COLS:
        schema[f"feat_{col}_lag2d"] = pl.Float64
        schema[f"feat_{col}_lag7d"] = pl.Float64
    for col in METEO_HIST_COLS:
        short = col.replace("hist_", "")
        schema[f"feat_{short}_lag2d"] = pl.Float64
        schema[f"feat_{short}_lag7d"] = pl.Float64
    for col in pred_cols_to_use:
        schema[col] = pl.Float64

    df_feat = pl.DataFrame(records, schema=schema)

    # ── Verification ──────────────────────────────────────────────────────
    feat_cols = [c for c in df_feat.columns
                 if c not in ("timestamp", "date_locale", "target")]

    print(f"\nShape finale : {df_feat.shape[0]:,} lignes x "
          f"{df_feat.shape[1]} col")
    print(f"Periode      : {df_feat['date_locale'].min()} -> "
          f"{df_feat['date_locale'].max()}")
    print(f"Features     : {len(feat_cols)}")

    print(f"\nVerification NaN (features cles) :")
    for col in [
        "feat_load_lag2d",
        "feat_pv_central_valais_lag2d",
        "feat_temperature_lag2d",
        "lag_13h_pred_temperature_ctrl",
        "lag_20h_pred_temperature_ctrl",
        "lag_33h_pred_temperature_ctrl",
    ]:
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