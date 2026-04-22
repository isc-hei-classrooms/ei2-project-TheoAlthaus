"""
Feature engineering — version 5
==================================
Lit  : data/oiken_clean.parquet
       data/calendar.parquet
       data/meteo_sion_pred_clean.parquet
Ecrit: data/features_v5.parquet

Features :
  - Calendaire : day_of_year_sin/cos, hour_sin/cos, is_weekend (5)
  - Load lags  : feat_load_lag2d, feat_load_lag7d (2)
  - Previsions : temperature ctrl+stde, sunshine ctrl+stde x lags 13h-33h (84)
  Total : 91 features

Correction causalite :
  lag_Nh disponible uniquement si emission (timestamp_target - Nh) <= heure decision
  Heure decision = 10h UTC (hiver, UTC+1) ou 09h UTC (ete, UTC+2)
  => NaN pour les lags non disponibles selon l'heure cible locale
"""

import datetime
import polars as pl
from zoneinfo import ZoneInfo

from config import DATA_DIR, FILE_OIKEN_CLEAN, FILE_CALENDAR, FILE_METEO_PRED_CLEAN

TZ_LOCAL        = "Europe/Zurich"
TRAIN_END_LOCAL = datetime.date(2024,  6, 30)
VAL_END_LOCAL   = datetime.date(2024, 12, 31)
FILE_OUT        = DATA_DIR / "features_v5.parquet"

CAL_COLS = [
    "day_of_year_sin", "day_of_year_cos",
    "hour_sin", "hour_cos",
    "is_weekend",
]

# Lags de prevision disponibles (contrainte causalite)
PRED_LAGS = list(range(13, 34))  # 13h -> 33h

# Variables et sous-types de prevision a garder
PRED_VARS_SUBTYPES = [
    ("pred_temperature", "ctrl"),
    ("pred_temperature", "stde"),
    ("pred_sunshine",    "ctrl"),
    ("pred_sunshine",    "stde"),
]


def add_local_time(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("timestamp").dt.convert_time_zone(TZ_LOCAL).alias("ts_local")
    ).with_columns([
        pl.col("ts_local").dt.date().alias("date_locale"),
        pl.col("ts_local").dt.hour().alias("hour_local"),
        pl.col("ts_local").dt.minute().alias("minute_local"),
    ])


def get_decision_hour_utc(ts_utc) -> int:
    """
    Retourne l'heure de decision en UTC selon la saison :
    - Hiver (UTC+1) : decision 10h locale = 09h UTC
    - Ete  (UTC+2) : decision 11h locale = 09h UTC
    Attention : hiver = UTC+1 => 11h locale - 1 = 10h UTC
                ete   = UTC+2 => 11h locale - 2 = 09h UTC
    """
    tz    = ZoneInfo(TZ_LOCAL)
    dt    = ts_utc.replace(tzinfo=datetime.timezone.utc).astimezone(tz)
    offset_h = int(dt.utcoffset().total_seconds() / 3600)
    return 11 - offset_h  # 10h UTC en hiver, 09h UTC en ete


def max_available_lag(ts_utc) -> float:
    tz         = ZoneInfo(TZ_LOCAL)
    dt_local   = ts_utc.replace(tzinfo=datetime.timezone.utc).astimezone(tz)
    offset_h   = int(dt_local.utcoffset().total_seconds() / 3600)
    decision_h_utc = 11 - offset_h  # 10h UTC hiver, 09h UTC ete

    # Heure locale du target (0-23.xx)
    target_h_local = dt_local.hour + dt_local.minute / 60

    # Lag minimum = heures entre emission max et target
    # Emission max = decision_h_utc le jour J local
    # Target = target_h_local le jour J+1 local
    # Ecart = (24 - decision_h_utc) + (target_h_local en UTC)
    #       = 24 - decision_h_utc + target_h_local + offset_h - 24  (si target > minuit UTC)
    # Plus simplement : lag = (24 - decision_h_local) + target_h_local
    # car decision_h_local = 11h toujours
    lag_min = (24 - 11) + target_h_local  # = 13 + target_h_local
    return lag_min


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)

    # ── Chargement ────────────────────────────────────────────────────────
    print("Chargement des donnees...")
    oiken = pl.read_parquet(FILE_OIKEN_CLEAN).sort("timestamp")
    cal   = pl.read_parquet(FILE_CALENDAR).sort("timestamp")
    pred  = pl.read_parquet(FILE_METEO_PRED_CLEAN("sion")).sort("timestamp_target") \
              .rename({"timestamp_target": "timestamp"})

    print(f"  Oiken      : {oiken.shape[0]:,} lignes")
    print(f"  Calendrier : {cal.shape[0]:,} lignes")
    print(f"  Meteo pred : {pred.shape[0]:,} lignes x {pred.shape[1]} col")

    # ── Ajouter heure locale ──────────────────────────────────────────────
    oiken = add_local_time(oiken)

    # ── Jours normaux ─────────────────────────────────────────────────────
    pts_per_day   = (oiken.group_by("date_locale")
                         .agg(pl.len().alias("n_pts"))
                         .sort("date_locale"))
    normal_days   = pts_per_day.filter(pl.col("n_pts") == 96)["date_locale"].to_list()
    normal_days_s = set(normal_days)

    abnormal = pts_per_day.filter(pl.col("n_pts") != 96)
    print(f"\n  Jours normaux   : {len(normal_days):,}")
    print(f"  Jours ignores   :")
    for row in abnormal.iter_rows(named=True):
        print(f"    {row['date_locale']}  ->  {row['n_pts']} pts")

    oiken = oiken.filter(pl.col("date_locale").is_in(normal_days))

    # ── Index load ────────────────────────────────────────────────────────
    print("\nConstruction index load...")
    load_index = {
        (row["date_locale"], row["hour_local"], row["minute_local"]): row["load"]
        for row in oiken.select([
            "date_locale", "hour_local", "minute_local", "load"
        ]).iter_rows(named=True)
    }

    # ── Index calendrier ──────────────────────────────────────────────────
    print("Construction index calendrier...")
    cal_index = {
        row["timestamp"]: {c: row[c] for c in CAL_COLS}
        for row in cal.select(["timestamp"] + CAL_COLS).iter_rows(named=True)
    }

    # ── Colonnes de prevision a extraire ──────────────────────────────────
    pred_col_names = [
        f"lag_{lag:02d}h_{var}_{subtype}"
        for lag in PRED_LAGS
        for var, subtype in PRED_VARS_SUBTYPES
    ]
    pred_col_names = [c for c in pred_col_names if c in pred.columns]
    print(f"\nColonnes prevision : {len(pred_col_names)}")

    # ── Interpolation 1h -> 15min ─────────────────────────────────────────
    print("Interpolation previsions 1h -> 15min...")
    pred_interp = pred.select(["timestamp"] + pred_col_names).with_columns([
        pl.col(c).interpolate() for c in pred_col_names
    ])

    # Index previsions (timestamp_utc -> dict)
    pred_index = {
        row["timestamp"]: {c: row[c] for c in pred_col_names}
        for row in pred_interp.iter_rows(named=True)
    }
    print(f"  {len(pred_index):,} entrees")

    # ── Construction des features ─────────────────────────────────────────
    print("\nConstruction des features...")

    records   = []
    n_skipped = 0

# ── Diagnostic ────────────────────────────────────────────────────────
    print("\nDiagnostic causalite sur 5 exemples :")
    for row in oiken.select([
        "timestamp", "date_locale", "hour_local", "minute_local"
    ]).head(20).iter_rows(named=True):
        if row["hour_local"] == 0 and row["minute_local"] == 0:
            ts_utc   = row["timestamp"]
            lag_min  = max_available_lag(ts_utc)
            tz       = ZoneInfo(TZ_LOCAL)
            dt_local = ts_utc.replace(tzinfo=datetime.timezone.utc).astimezone(tz)
            offset_h = int(dt_local.utcoffset().total_seconds() / 3600)
            print(f"  ts_utc      : {ts_utc}")
            print(f"  heure_locale: {row['hour_local']}h{row['minute_local']}")
            print(f"  offset_h    : UTC+{offset_h}")
            print(f"  lag_min     : {lag_min:.2f}h")
            print(f"  lags dispo  : {[l for l in PRED_LAGS if l >= lag_min]}")
            print()
            break

    for row in oiken.select([
        "timestamp", "date_locale", "hour_local", "minute_local", "load"
    ]).iter_rows(named=True):

        ts_utc  = row["timestamp"]
        date_j  = row["date_locale"]
        h       = row["hour_local"]
        m       = row["minute_local"]

        # J-2 et J-7 en jours locaux
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

        # ── Assembler la ligne ────────────────────────────────────────────
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

        # Previsions avec contrainte causalite
        lag_min    = max_available_lag(ts_utc)
        pred_vals  = pred_index.get(ts_utc, {})

        for lag in PRED_LAGS:
            for var, subtype in PRED_VARS_SUBTYPES:
                col = f"lag_{lag:02d}h_{var}_{subtype}"
                # NaN si ce lag n'est pas disponible au moment de la decision
                if lag >= lag_min:
                    rec[col] = pred_vals.get(col, None)
                else:
                    rec[col] = None

        records.append(rec)

    print(f"  Points construits : {len(records):,}")
    print(f"  Points ignores    : {n_skipped:,}")

    # ── Schema et DataFrame ───────────────────────────────────────────────
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
    for lag in PRED_LAGS:
        for var, subtype in PRED_VARS_SUBTYPES:
            schema[f"lag_{lag:02d}h_{var}_{subtype}"] = pl.Float64

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