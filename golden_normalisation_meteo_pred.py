"""
Normalisation meteorologie previsions — golden dataset
=======================================================
Lit  : golden_data/meteo_sion_pred_golden_raw.parquet
Ecrit: golden_data/meteo_sion_pred_golden_clean.parquet

Structure de sortie :
  timestamp_target | lag_13h_{var}_{subtype} | ... | lag_33h_{var}_{subtype}

  lag_Nh_{var}_{subtype} = prevision de {var} emise N heures avant timestamp_target
  Uniquement lags 13h -> 33h (contrainte causalite)
  Sous-types : ctrl + stde uniquement
"""

import polars as pl
from pathlib import Path

from config import TARGET_STEP_MIN

GOLDEN_DIR = Path(__file__).parent / "golden_data"
FILE_IN    = GOLDEN_DIR / "meteo_sion_pred_golden_raw.parquet"
FILE_OUT   = GOLDEN_DIR / "meteo_sion_pred_golden_clean.parquet"

PERIOD_START = pl.datetime(2025,  9, 30,  0,  0, 0,
                            time_unit="us", time_zone="UTC")
PERIOD_END   = pl.datetime(2026,  4, 22,  0,  0, 0,
                            time_unit="us", time_zone="UTC")

# Uniquement lags 13h -> 33h
LAGS = list(range(13, 34))

# Variables et sous-types
PRED_VARS_GOLDEN = {
    "Temperature": "pred_temperature",
    "Radiation":   "pred_radiation",
    "Sunshine":    "pred_sunshine",
    "Humidity":    "pred_humidity",
}
SUBTYPES_GOLDEN = ["ctrl", "stde"]


def make_grid_15() -> pl.DataFrame:
    return pl.DataFrame({
        "timestamp_target": pl.datetime_range(
            PERIOD_START, PERIOD_END,
            interval=f"{TARGET_STEP_MIN}m",
            time_unit="us", time_zone="UTC", eager=True,
        )
    })


def normalise_pred_golden() -> None:
    print("="*65)
    print("  NORMALISATION METEO PREVISIONS — GOLDEN (Sion)")
    print("="*65)

    df = pl.read_parquet(FILE_IN)
    print(f"  Shape brute : {df.shape[0]:,} lignes x {df.shape[1]} col")

    # ── Colonnes de sortie ────────────────────────────────────────────────
    out_cols = [
        f"lag_{lag:02d}h_{var_col}_{subtype}"
        for lag in LAGS
        for var_col in PRED_VARS_GOLDEN.values()
        for subtype in SUBTYPES_GOLDEN
    ]
    print(f"  Colonnes de sortie : {len(out_cols)}")
    print(f"  Lags              : {LAGS[0]}h -> {LAGS[-1]}h")
    print(f"  Sous-types        : {SUBTYPES_GOLDEN}")

    # ── Grille 15min ──────────────────────────────────────────────────────
    grid    = make_grid_15()
    ts_list = grid["timestamp_target"].to_list()
    ts_set  = set(ts_list)
    print(f"  Grille 15min      : {len(ts_list):,} points")

    # ── Construction du mapping timestamp_target -> valeurs ───────────────
    print(f"\n  Construction du mapping...")

    target_data: dict[object, dict[str, float]] = {}

    for row in df.iter_rows(named=True):
        ts_target = row["timestamp_target"]

        if ts_target not in ts_set:
            continue

        ts_emit = row["timestamp_emission"]
        diff_s  = (ts_target - ts_emit).total_seconds()
        lag_h   = int(round(diff_s / 3600))

        if lag_h not in LAGS:
            continue

        if ts_target not in target_data:
            target_data[ts_target] = {}

        for var_col in PRED_VARS_GOLDEN.values():
            for subtype in SUBTYPES_GOLDEN:
                src_col = f"{var_col}_{subtype}_run{lag_h:02d}"
                dst_col = f"lag_{lag_h:02d}h_{var_col}_{subtype}"
                try:
                    val = row[src_col]
                    if val is not None:
                        target_data[ts_target][dst_col] = float(val)
                except KeyError:
                    pass

    print(f"  Timestamps avec donnees : {len(target_data):,}")

    # ── Diagnostic ────────────────────────────────────────────────────────
    if target_data:
        sample_ts  = next(iter(target_data))
        sample_val = target_data[sample_ts]
        print(f"\n  Exemple mapping pour {sample_ts} :")
        print(f"  Nombre de cles : {len(sample_val)}")
        for k, v in list(sample_val.items())[:5]:
            print(f"    '{k}' = {v:.3f}")

    # ── Construire le DataFrame final ─────────────────────────────────────
    print(f"\n  Construction du DataFrame final...")

    columns = {"timestamp_target": ts_list}
    for col in out_cols:
        columns[col] = [
            target_data.get(ts, {}).get(col, None)
            for ts in ts_list
        ]

    schema = {"timestamp_target": pl.Datetime("us", "UTC")}
    for col in out_cols:
        schema[col] = pl.Float64

    df_out = pl.DataFrame(columns, schema=schema)

    # ── Verification ──────────────────────────────────────────────────────
    print(f"\n  Verification :")
    for lag in [13, 20, 25, 33]:
        col = f"lag_{lag:02d}h_pred_temperature_ctrl"
        if col not in df_out.columns:
            continue
        n_null  = df_out[col].is_null().sum()
        n_total = df_out.shape[0]
        pct     = (n_total - n_null) / n_total * 100
        print(f"  {col:<45} {n_total - n_null:>8,} valeurs  ({pct:.1f}% rempli)")

    # ── Sauvegarde ────────────────────────────────────────────────────────
    df_out.write_parquet(FILE_OUT)
    size_mb = FILE_OUT.stat().st_size / 1024 / 1024
    print(f"\n  -> {FILE_OUT.name}  "
          f"({df_out.shape[0]:,} lignes x {df_out.shape[1]} col, "
          f"{size_mb:.1f} MB)")
    print(f"  Periode : {df_out['timestamp_target'].min()} -> "
          f"{df_out['timestamp_target'].max()}")


if __name__ == "__main__":
    GOLDEN_DIR.mkdir(exist_ok=True)
    normalise_pred_golden()