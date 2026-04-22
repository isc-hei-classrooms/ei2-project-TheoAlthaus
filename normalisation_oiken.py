"""
Normalisation Oiken
====================
Lit  : data/oiken_raw.parquet
Ecrit: data/oiken_clean.parquet

Etapes :
  1. Assignation directe sur grille 15min UTC (plage Oiken)
  2. Correction offset nocturne pv_remote
"""

from pathlib import Path
import polars as pl

from config import (
    DATA_DIR,
    FILE_OIKEN_RAW,
    FILE_OIKEN_CLEAN,
    TARGET_STEP_MIN,
    OIKEN_INTERP_MAX_PTS,
)

# Plage temporelle basee sur Oiken
PERIOD_START = pl.datetime(2022, 10,  1,  0, 15, 0, time_unit="us", time_zone="UTC")
PERIOD_END   = pl.datetime(2025,  9, 30,  0,  0, 0, time_unit="us", time_zone="UTC")


def make_grid() -> pl.DataFrame:
    return pl.DataFrame({
        "timestamp": pl.datetime_range(
            PERIOD_START, PERIOD_END,
            interval=f"{TARGET_STEP_MIN}m",
            time_unit="us", time_zone="UTC", eager=True,
        )
    })

def interpolate_short_gaps(df: pl.DataFrame, numeric_cols: list[str],
                            max_pts: int) -> pl.DataFrame:
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
        print(f"  {col:<30} NaN avant={n_nan:>6,}  "
              f"interpoles={n_nan - n_nan_after:>6,}  "
              f"restants={n_nan_after:>6,}")

    return df

def normalise_oiken() -> None:
    print("="*65)
    print("  NORMALISATION OIKEN")
    print("="*65)

    df = pl.read_parquet(FILE_OIKEN_RAW)
    df = df.rename({"timestamp": "timestamp_local"})

    num_cols = ["load", "forecast_load",
                "pv_central_valais", "pv_sion", "pv_sierre", "pv_remote"]
    num_cols = [c for c in num_cols if c in df.columns]

    # ── Verification du nombre de lignes ──────────────────────────────────
    grid       = make_grid()
    n_expected = grid.shape[0]
    n_actual   = df.shape[0]

    print(f"\n  Lignes CSV        : {n_actual:,}")
    print(f"  Lignes attendues  : {n_expected:,}")
    print(f"  Difference        : {n_actual - n_expected:+,}")
    print(f"  CSV debut         : {df['timestamp_local'].min()}")
    print(f"  CSV fin           : {df['timestamp_local'].max()}")
    print(f"  Grille UTC debut  : {grid['timestamp'].min()}")
    print(f"  Grille UTC fin    : {grid['timestamp'].max()}")

    if n_actual != n_expected:
        print(f"\  ATTENTION : le nombre de lignes ne correspond pas.")
        print(f"  Assignation positionnelle — verifier les donnees brutes.")

    # ── Assignation directe sur la grille UTC ─────────────────────────────
    records = {"timestamp_local": df["timestamp_local"].to_list()}
    for col in num_cols:
        records[col] = df[col].to_list()

    df_out = grid.with_columns([
        pl.Series("timestamp_local", records["timestamp_local"],
                  dtype=pl.Datetime("us")),
        *[pl.Series(c, records[c], dtype=pl.Float64) for c in num_cols]
    ])

    print(f"\n  Assignation complete : {n_actual:,} lignes placees sur la grille UTC")

    # ── Correction offset nocturne pv_remote ──────────────────────────────
    # Nuit : 21h-5h UTC pour couvrir la nuit en heure locale
    # (UTC+1 hiver : 22h locale = 21h UTC)
    # (UTC+2 ete   : 22h locale = 20h UTC)
    night_mask = (
        (pl.col("timestamp").dt.hour() >= 21) |
        (pl.col("timestamp").dt.hour() <= 5)
    )
    offset = df_out.filter(night_mask)["pv_remote"].drop_nulls().median()
    print(f"\n  Offset nocturne pv_remote detecte : {offset:.4f} kWh")

    df_out = df_out.with_columns(
        (pl.col("pv_remote") - offset)
          .clip(lower_bound=0)
          .alias("pv_remote")
    )
    print(f"  pv_remote : offset soustrait et clippe a 0")

    # ── Interpolation lineaire des trous <= 2h ────────────────────────────
    print(f"\n  Interpolation (max {OIKEN_INTERP_MAX_PTS} points = 2h) :")
    df_out = interpolate_short_gaps(df_out, num_cols, OIKEN_INTERP_MAX_PTS)

    # ── Sauvegarde ────────────────────────────────────────────────────────
    df_out.write_parquet(FILE_OIKEN_CLEAN)
    size_mb = FILE_OIKEN_CLEAN.stat().st_size / 1024 / 1024
    print(f"\n  -> {FILE_OIKEN_CLEAN.name}  "
          f"({df_out.shape[0]:,} lignes x {df_out.shape[1]} col, {size_mb:.1f} MB)")
    print(f"  Periode : {df_out['timestamp'].min()} -> {df_out['timestamp'].max()}")


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    normalise_oiken()