"""
Feature engineering — version 3 (quart d'heure + meteo)
=========================================================
Lit  : data/oiken_clean.parquet
       data/calendar.parquet
       data/meteo_sion_hist_clean.parquet
Ecrit: data/features_v3.parquet

Structure :
  Une ligne = un quart d'heure
  Features  : lag J-2 et J-7 (load + meteo Sion) + calendrier
  Target    : load du quart d'heure courant
  Split     : train / val / test
"""

import polars as pl

from config import DATA_DIR, FILE_OIKEN_CLEAN, FILE_CALENDAR, FILE_METEO_HIST_CLEAN

PERIOD_START = pl.datetime(2022, 10,  1,  0, 15, 0, time_unit="us", time_zone="UTC")
PERIOD_END   = pl.datetime(2025,  9, 30,  0,  0, 0, time_unit="us", time_zone="UTC")

LAG_2D = 2 * 96   # 192 points = J-2
LAG_7D = 7 * 96   # 672 points = J-7

TRAIN_END = pl.datetime(2024,  6, 30, 23, 45, 0, time_unit="us", time_zone="UTC")
VAL_END   = pl.datetime(2024, 12, 31, 23, 45, 0, time_unit="us", time_zone="UTC")

FILE_OUT = DATA_DIR / "features_v3.parquet"

CAL_COLS = [
    "hour_sin", "hour_cos",
    "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos",
    "day_of_year_sin", "day_of_year_cos",
    "is_weekend", "is_holiday", "is_school_holiday",
]

METEO_COLS = [
    "hist_temperature",
    "hist_radiation",
    "hist_sunshine",
    "hist_humidity",
]


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)

    # ── Chargement ────────────────────────────────────────────────────────
    print("Chargement des donnees...")
    oiken = pl.read_parquet(FILE_OIKEN_CLEAN).sort("timestamp")
    cal   = pl.read_parquet(FILE_CALENDAR).sort("timestamp")
    meteo = pl.read_parquet(FILE_METEO_HIST_CLEAN("sion")).sort("timestamp")

    print(f"  Oiken      : {oiken.shape[0]:,} lignes")
    print(f"  Calendrier : {cal.shape[0]:,} lignes")
    print(f"  Meteo Sion : {meteo.shape[0]:,} lignes")

    # ── Jointures ─────────────────────────────────────────────────────────
    df = (
        oiken.select(["timestamp", "load"])
        .join(cal.select(["timestamp"] + CAL_COLS),
              on="timestamp", how="left")
        .join(meteo.select(["timestamp"] + METEO_COLS),
              on="timestamp", how="left")
    )

    print(f"  Apres join : {df.shape[0]:,} lignes x {df.shape[1]} col")

    # ── Features lag load ─────────────────────────────────────────────────
    print("\nConstruction des features lag...")

    lag_exprs = [
        pl.col("load").shift(LAG_2D).alias("feat_load_lag2d"),
        pl.col("load").shift(LAG_7D).alias("feat_load_lag7d"),
    ]

    # ── Features lag meteo ────────────────────────────────────────────────
    for col in METEO_COLS:
        short = col.replace("hist_", "")
        lag_exprs.append(
            pl.col(col).shift(LAG_2D).alias(f"feat_{short}_lag2d")
        )
        lag_exprs.append(
            pl.col(col).shift(LAG_7D).alias(f"feat_{short}_lag7d")
        )

    df = df.with_columns(lag_exprs)

    # ── Split ─────────────────────────────────────────────────────────────
    df = df.with_columns(
        pl.when(pl.col("timestamp") <= TRAIN_END)
          .then(pl.lit("train"))
          .when(pl.col("timestamp") <= VAL_END)
          .then(pl.lit("val"))
          .otherwise(pl.lit("test"))
          .alias("split")
    )

    # ── Renommer load en target ───────────────────────────────────────────
    df = df.rename({"load": "target"})

    # ── Supprimer les colonnes meteo brutes (on garde uniquement les lags) ─
    df = df.drop(METEO_COLS)

    # ── Supprimer les lignes avec NaN ─────────────────────────────────────
    n_before = df.shape[0]
    df = df.drop_nulls()
    n_after  = df.shape[0]
    print(f"  Lignes supprimees (NaN) : {n_before - n_after:,}")

    # ── Verification ──────────────────────────────────────────────────────
    feat_cols = [c for c in df.columns
                 if c not in ("timestamp", "target", "split")]

    print(f"\nShape finale : {df.shape[0]:,} lignes x {df.shape[1]} col")
    print(f"Periode      : {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"Features     : {len(feat_cols)}")
    for c in feat_cols:
        print(f"  {c}")

    print(f"\nRepartition split :")
    print(df.group_by("split")
            .agg(pl.len().alias("n_points"))
            .sort("n_points", descending=True))

    print(f"\nVerification NaN :")
    n_nan_total = sum(df[c].is_null().sum() for c in feat_cols + ["target"])
    print(f"  Total NaN : {n_nan_total:,}")

    print(f"\nApercu :")
    print(df.select(["timestamp", "feat_load_lag2d", "feat_load_lag7d",
                     "feat_temperature_lag2d", "feat_radiation_lag2d",
                     "hour_sin", "is_weekend", "target", "split"]).head(5))

    # ── Sauvegarde ────────────────────────────────────────────────────────
    df.write_parquet(FILE_OUT)
    size_mb = FILE_OUT.stat().st_size / 1024 / 1024
    print(f"\n-> {FILE_OUT.name}  "
          f"({df.shape[0]:,} lignes x {df.shape[1]} col, {size_mb:.1f} MB)")