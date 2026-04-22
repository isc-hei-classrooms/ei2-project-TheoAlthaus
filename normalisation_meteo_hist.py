"""
Normalisation meteorologie historique
=======================================
Lit  : data/meteo_{station}_hist_raw.parquet
Ecrit: data/meteo_{station}_hist_clean.parquet

Etapes :
  1. Alignement sur grille 10min UTC
  2. Interpolation lineaire des trous <= 1h (6 points a 10min)
  3. Affichage des grands trous restants
  4. Resampling 10min -> 15min
  5. Check des NaN restants
  6. Sauvegarde
"""

import polars as pl
from datetime import datetime

from config import (
    DATA_DIR,
    STATIONS,
    METEO_HIST_STEP_MIN,
    TARGET_STEP_MIN,
    FILE_METEO_HIST_RAW,
    FILE_METEO_HIST_CLEAN,
)

PERIOD_START = pl.datetime(2022, 10,  1,  0, 15, 0, time_unit="us", time_zone="UTC")
PERIOD_END   = pl.datetime(2025,  9, 30,  0,  0, 0, time_unit="us", time_zone="UTC")

INTERP_MAX_PTS_10MIN = 6   # 6 x 10min = 1h


def make_grid_15() -> pl.DataFrame:
    return pl.DataFrame({
        "timestamp": pl.datetime_range(
            PERIOD_START, PERIOD_END,
            interval=f"{TARGET_STEP_MIN}m",
            time_unit="us", time_zone="UTC", eager=True,
        )
    })


def interpolate_short_gaps(df: pl.DataFrame, numeric_cols: list[str],
                            max_pts: int, label: str) -> pl.DataFrame:
    for col in numeric_cols:
        if col not in df.columns:
            continue

        n_nan = df[col].is_null().sum()
        if n_nan == 0:
            continue

        runs = (
            df.with_columns(pl.col(col).is_null().cast(pl.Int8).alias("_is_null"))
              .with_columns(
                  (pl.col("_is_null") != pl.col("_is_null").shift(1))
                    .fill_null(True).cum_sum().alias("_run_id")
              )
              .group_by("_run_id", maintain_order=True)
              .agg([
                  pl.col("_is_null").first().alias("is_null"),
                  pl.len().alias("run_len"),
                  pl.first("timestamp").alias("run_start"),
              ])
              .filter(pl.col("is_null") == 1)
        )

        df = df.with_columns(
            pl.col(col).interpolate().alias(f"_{col}_interp")
        )

        long_runs      = runs.filter(pl.col("run_len") > max_pts)
        large_gap_mask = pl.Series("mask", [False] * df.shape[0])

        if long_runs.shape[0] > 0:
            for row in long_runs.iter_rows(named=True):
                idx_mask       = (df["timestamp"] >= row["run_start"]) & df[col].is_null()
                large_gap_mask = large_gap_mask | idx_mask

            df = df.with_columns(
                pl.when(large_gap_mask)
                  .then(None)
                  .otherwise(pl.col(f"_{col}_interp"))
                  .alias(f"_{col}_interp")
            )

        df = df.with_columns(
            pl.col(f"_{col}_interp").alias(col)
        ).drop(f"_{col}_interp")

        n_nan_after = df[col].is_null().sum()
        print(f"  [{label}] {col:<28} NaN avant={n_nan:>6,}  "
              f"interpoles={n_nan - n_nan_after:>6,}  "
              f"restants={n_nan_after:>6,}")

    return df


def get_gap_ranges(df: pl.DataFrame, col: str) -> list[tuple]:
    """
    Retourne la liste des plages (t_start, t_end) des NaN restants
    sous forme de tuples de datetime.
    """
    runs = (
        df.with_columns(pl.col(col).is_null().cast(pl.Int8).alias("_is_null"))
          .with_columns(
              (pl.col("_is_null") != pl.col("_is_null").shift(1))
                .fill_null(True).cum_sum().alias("_run_id")
          )
          .group_by("_run_id", maintain_order=True)
          .agg([
              pl.col("_is_null").first().alias("is_null"),
              pl.first("timestamp").alias("run_start"),
              pl.last("timestamp").alias("run_end"),
          ])
          .filter(pl.col("is_null") == 1)
    )
    return [(row["run_start"], row["run_end"])
            for row in runs.iter_rows(named=True)]


def normalise_hist(station: str, file_name: str) -> None:
    print(f"\n{'='*65}")
    print(f"  {station}")
    print(f"{'='*65}")

    path_in   = FILE_METEO_HIST_RAW(file_name)
    df        = pl.read_parquet(path_in)
    hist_cols = [c for c in df.columns if c.startswith("hist_")]

    print(f"  Shape brute : {df.shape[0]:,} lignes x {df.shape[1]} col")
    print(f"  Periode brute : {df['timestamp'].min()} -> {df['timestamp'].max()}")

    # ── 1. Alignement sur grille 10min UTC ────────────────────────────────
    df = df.with_columns(
        pl.col("timestamp").dt.truncate("10m").alias("timestamp")
    )
    df = df.unique(subset=["timestamp"], keep="first").sort("timestamp")

    t_min   = df["timestamp"].min()
    t_max   = df["timestamp"].max()
    grid_10 = pl.DataFrame({
        "timestamp": pl.datetime_range(
            t_min, t_max,
            interval=f"{METEO_HIST_STEP_MIN}m",
            time_unit="us", time_zone="UTC", eager=True,
        )
    })

    n_expected = grid_10.shape[0]
    n_actual   = df.shape[0]
    print(f"\n  Grille 10min : {n_expected:,} attendus, "
          f"{n_actual:,} presents, {n_expected - n_actual:,} manquants")

    df = grid_10.join(df, on="timestamp", how="left")

    # ── 2. Interpolation sur grille 10min (max 6 pts = 1h) ────────────────
    print(f"\n  Interpolation 10min (max {INTERP_MAX_PTS_10MIN} pts = 1h) :")
    df = interpolate_short_gaps(df, hist_cols, INTERP_MAX_PTS_10MIN, station)

    # ── 3. Affichage des grands trous restants ────────────────────────────
    print(f"\n  Grands trous restants (> 1h) :")
    any_gap = False
    for col in hist_cols:
        runs = (
            df.with_columns(pl.col(col).is_null().cast(pl.Int8).alias("_is_null"))
              .with_columns(
                  (pl.col("_is_null") != pl.col("_is_null").shift(1))
                    .fill_null(True).cum_sum().alias("_run_id")
              )
              .group_by("_run_id", maintain_order=True)
              .agg([
                  pl.col("_is_null").first().alias("is_null"),
                  pl.len().alias("run_len"),
                  pl.first("timestamp").alias("run_start"),
                  pl.last("timestamp").alias("run_end"),
              ])
              .filter((pl.col("is_null") == 1) &
                      (pl.col("run_len") > INTERP_MAX_PTS_10MIN))
        )

        if runs.shape[0] == 0:
            continue

        any_gap = True
        total   = runs["run_len"].sum()
        print(f"\n  {col:<30} {runs.shape[0]} trous, {total:,} pts manquants")
        for row in runs.iter_rows(named=True):
            duree_h = row["run_len"] * METEO_HIST_STEP_MIN / 60
            print(f"    {row['run_start']}  ->  {row['run_end']}"
                  f"  ({row['run_len']} pts = {duree_h:.1f}h)")

    if not any_gap:
        print("  Aucun grand trou detecte")

    # ── 4. Resampling 10min -> 15min ──────────────────────────────────────
    # Sauvegarder les plages temporelles des grands trous par colonne
    # (plages et non timestamps exacts pour couvrir aussi les 15min intercales)
    gap_ranges: dict[str, list[tuple]] = {}
    for col in hist_cols:
        gap_ranges[col] = get_gap_ranges(df, col)

    grid_15 = make_grid_15()

    # Fusionner grilles 10min et 15min
    ts_10min   = set(df["timestamp"].to_list())
    ts_15_only = [t for t in grid_15["timestamp"].to_list() if t not in ts_10min]

    grid_15_only = pl.DataFrame({
        "timestamp": pl.Series(ts_15_only, dtype=pl.Datetime("us", "UTC"))
    }).with_columns([
        pl.lit(None).cast(pl.Float64).alias(c) for c in hist_cols
    ])

    grid_combined = pl.concat([df, grid_15_only]).sort("timestamp")

    # Interpoler tout sans limite
    grid_combined = grid_combined.with_columns([
        pl.col(c).interpolate() for c in hist_cols
    ])

    # Re-appliquer NaN aux grands trous par plage temporelle
    # Couvre les timestamps 10min ET 15min dans les trous
    ts_list = grid_combined["timestamp"].to_list()
    for col in hist_cols:
        if not gap_ranges[col]:
            continue

        mask = pl.Series([
            any(t_start <= t <= t_end for t_start, t_end in gap_ranges[col])
            for t in ts_list
        ])

        grid_combined = grid_combined.with_columns(
            pl.when(mask)
              .then(None)
              .otherwise(pl.col(col))
              .alias(col)
        )

    # Garder uniquement les timestamps 15min de la plage Oiken
    ts_15_set = set(grid_15["timestamp"].to_list())
    df_15 = grid_combined.filter(
        pl.col("timestamp").is_in(list(ts_15_set))
    ).sort("timestamp")

    # ── 5. Check des NaN restants ─────────────────────────────────────────
    print(f"\n  NaN restants apres resampling :")
    for col in hist_cols:
        n_null = df_15[col].is_null().sum()
        if n_null > 0:
            print(f"  {col:<30} {n_null:>6,} NaN")
        else:
            print(f"  {col:<30} aucun NaN")

    # ── 6. Sauvegarde ─────────────────────────────────────────────────────
    path_out = FILE_METEO_HIST_CLEAN(file_name)
    df_15.write_parquet(path_out)
    size_mb = path_out.stat().st_size / 1024 / 1024
    print(f"\n  -> {path_out.name}  "
          f"({df_15.shape[0]:,} lignes x {df_15.shape[1]} col, {size_mb:.1f} MB)")
    print(f"  Periode : {df_15['timestamp'].min()} -> {df_15['timestamp'].max()}")


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)

    for station, file_name in STATIONS.items():
        normalise_hist(station, file_name)

    print(f"\n{'='*65}")
    print("FICHIERS GENERES")
    print(f"{'='*65}")
    for p in sorted(DATA_DIR.glob("meteo_*_hist_clean.parquet")):
        df      = pl.read_parquet(p)
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  {p.name:<50} {df.shape[0]:>9,} lignes  "
              f"{df.shape[1]:>5} col  {size_mb:>6.1f} MB")