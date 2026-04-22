"""
Feature engineering — version 1 corrigee (jours locaux)
=========================================================
Lit  : data/oiken_clean.parquet
Ecrit: data/features_v1_local.parquet

Structure :
  Une ligne = un jour local J (Europe/Zurich)
  Features  : 96 valeurs load J-2 + 96 valeurs load J-7 (heure locale)
  Targets   : 96 valeurs load jour J (00h00 -> 23h45 heure locale)
  Split     : train / val / test

Corrections vs v1 :
  - Groupement par jour local (Europe/Zurich) au lieu de jour UTC
  - Lags calcules en jours calendaires locaux
  - Jours de changement d'heure (92 ou 100 points) ignores
"""

import polars as pl
from zoneinfo import ZoneInfo

from config import DATA_DIR, FILE_OIKEN_CLEAN

TZ_LOCAL  = "Europe/Zurich"

TRAIN_END_LOCAL = "2024-06-30"
VAL_END_LOCAL   = "2024-12-31"

FILE_OUT = DATA_DIR / "features_v1_local.parquet"


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)

    # ── Chargement ────────────────────────────────────────────────────────
    print("Chargement Oiken clean...")
    df = pl.read_parquet(FILE_OIKEN_CLEAN).sort("timestamp")

    print(f"  {df.shape[0]:,} lignes")

    # ── Ajouter date et heure locales ─────────────────────────────────────
    df = df.with_columns([
        pl.col("timestamp")
          .dt.convert_time_zone(TZ_LOCAL)
          .alias("ts_local"),
    ]).with_columns([
        pl.col("ts_local").dt.date().alias("date_locale"),
        pl.col("ts_local").dt.hour().alias("hour_local"),
        pl.col("ts_local").dt.minute().alias("minute_local"),
    ])

    # ── Compter les points par jour local ─────────────────────────────────
    pts_per_day = (
        df.group_by("date_locale")
          .agg(pl.len().alias("n_pts"))
          .sort("date_locale")
    )

    # Identifier les jours normaux (96 pts) et les jours exceptionnels
    days_normal   = pts_per_day.filter(pl.col("n_pts") == 96)["date_locale"].to_list()
    days_abnormal = pts_per_day.filter(pl.col("n_pts") != 96)

    print(f"\n  Jours totaux       : {pts_per_day.shape[0]:,}")
    print(f"  Jours normaux (96) : {len(days_normal):,}")
    print(f"  Jours exceptionnels (ignores) :")
    for row in days_abnormal.iter_rows(named=True):
        print(f"    {row['date_locale']}  ->  {row['n_pts']} pts")

    # Garder uniquement les jours normaux
    df = df.filter(pl.col("date_locale").is_in(days_normal))

    # ── Construire un index (date_locale, heure_locale) -> load ───────────
    # Pour acceder rapidement aux valeurs lag
    print(f"\nConstruction de l'index date_locale x heure_locale...")

    # Cle : (date_locale, hour_local, minute_local) -> load
    load_index = {
        (row["date_locale"], row["hour_local"], row["minute_local"]): row["load"]
        for row in df.select([
            "date_locale", "hour_local", "minute_local", "load"
        ]).iter_rows(named=True)
    }

    # Liste des jours normaux tries
    import datetime
    days_sorted = sorted(days_normal)

    # ── Construire les features jour par jour ─────────────────────────────
    print(f"Construction des features...")

    # Quarts d'heure d'une journee normale (96 pts)
    quarters = [
        (h, m)
        for h in range(24)
        for m in [0, 15, 30, 45]
    ]

    records   = []
    n_skipped = 0

    for i, day_j in enumerate(days_sorted):
        # J-2 et J-7 en jours calendaires
        day_j2 = day_j - datetime.timedelta(days=2)
        day_j7 = day_j - datetime.timedelta(days=7)

        # Verifier que J-2 et J-7 sont des jours normaux
        if day_j2 not in set(days_normal) or day_j7 not in set(days_normal):
            n_skipped += 1
            continue

        # Extraire les 96 valeurs pour J, J-2, J-7
        targets = []
        lag2    = []
        lag7    = []
        valid   = True

        for h, m in quarters:
            key_j  = (day_j,  h, m)
            key_j2 = (day_j2, h, m)
            key_j7 = (day_j7, h, m)

            if key_j  not in load_index or \
               key_j2 not in load_index or \
               key_j7 not in load_index:
                valid = False
                break

            targets.append(load_index[key_j])
            lag2.append(load_index[key_j2])
            lag7.append(load_index[key_j7])

        if not valid:
            n_skipped += 1
            continue

        # Assembler la ligne
        row = {"date_locale": day_j}
        for q, val in enumerate(lag2, 1):
            row[f"feat_lag2d_q{q:02d}"] = val
        for q, val in enumerate(lag7, 1):
            row[f"feat_lag7d_q{q:02d}"] = val
        for q, val in enumerate(targets, 1):
            row[f"target_q{q:02d}"] = val

        records.append(row)

    print(f"  Jours construits : {len(records):,}")
    print(f"  Jours ignores    : {n_skipped:,}")

    # ── Construire le DataFrame ───────────────────────────────────────────
    schema = {"date_locale": pl.Date}
    for q in range(1, 97):
        schema[f"feat_lag2d_q{q:02d}"] = pl.Float64
        schema[f"feat_lag7d_q{q:02d}"] = pl.Float64
    for q in range(1, 97):
        schema[f"target_q{q:02d}"] = pl.Float64

    df_feat = pl.DataFrame(records, schema=schema)

    # ── Split train / val / test ──────────────────────────────────────────
    df_feat = df_feat.with_columns(
        pl.when(pl.col("date_locale") <= pl.lit(TRAIN_END_LOCAL).str.to_date())
          .then(pl.lit("train"))
          .when(pl.col("date_locale") <= pl.lit(VAL_END_LOCAL).str.to_date())
          .then(pl.lit("val"))
          .otherwise(pl.lit("test"))
          .alias("split")
    )

    # ── Verification ──────────────────────────────────────────────────────
    print(f"\nShape finale : {df_feat.shape[0]:,} lignes x {df_feat.shape[1]} col")
    print(f"Periode      : {df_feat['date_locale'].min()} -> "
          f"{df_feat['date_locale'].max()}")

    print(f"\nRepartition split :")
    print(df_feat.group_by("split")
                 .agg(pl.len().alias("n_jours"))
                 .sort("n_jours", descending=True))

    print(f"\nVerification NaN :")
    feat_cols   = [c for c in df_feat.columns if c.startswith("feat_")]
    target_cols = [c for c in df_feat.columns if c.startswith("target_")]
    n_nan_feat   = sum(df_feat[c].is_null().sum() for c in feat_cols)
    n_nan_target = sum(df_feat[c].is_null().sum() for c in target_cols)
    print(f"  NaN features : {n_nan_feat:,}")
    print(f"  NaN targets  : {n_nan_target:,}")

    print(f"\nApercu (premieres lignes, quelques colonnes) :")
    print(df_feat.select([
        "date_locale",
        "feat_lag2d_q01", "feat_lag2d_q48", "feat_lag2d_q96",
        "feat_lag7d_q01", "feat_lag7d_q48", "feat_lag7d_q96",
        "target_q01",     "target_q48",     "target_q96",
        "split",
    ]).head(5))

    # ── Sauvegarde ────────────────────────────────────────────────────────
    df_feat.write_parquet(FILE_OUT)
    size_mb = FILE_OUT.stat().st_size / 1024 / 1024
    print(f"\n-> {FILE_OUT.name}  "
          f"({df_feat.shape[0]:,} lignes x {df_feat.shape[1]} col, "
          f"{size_mb:.1f} MB)")