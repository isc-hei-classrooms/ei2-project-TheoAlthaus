"""
Normalisation meteorologie previsions
=======================================
Lit  : data/meteo_{station}_pred_raw.parquet
Ecrit: data/meteo_{station}_pred_clean.parquet

Structure de sortie :
  timestamp_target | lag_01h_{var}_{subtype} | ... | lag_33h_{var}_{subtype}

  lag_Nh_{var}_{subtype} = prevision de {var} emise N heures avant le timestamp_target
  Les timestamps 15min non alignes sur 1h sont laisses vides (null).
"""

import polars as pl

from config import (
    DATA_DIR,
    STATIONS,
    PRED_VARS,
    SUBTYPES,
    TARGET_STEP_MIN,
    FILE_METEO_PRED_RAW,
    FILE_METEO_PRED_CLEAN,
)

PERIOD_START = pl.datetime(2022, 10,  1,  0, 15, 0, time_unit="us", time_zone="UTC")
PERIOD_END   = pl.datetime(2025,  9, 30,  0,  0, 0, time_unit="us", time_zone="UTC")

LAGS = list(range(1, 34))  # 1h -> 33h


def make_grid_15() -> pl.DataFrame:
    return pl.DataFrame({
        "timestamp_target": pl.datetime_range(
            PERIOD_START, PERIOD_END,
            interval=f"{TARGET_STEP_MIN}m",
            time_unit="us", time_zone="UTC", eager=True,
        )
    })


def normalise_pred(station: str, file_name: str) -> None:
    print(f"\n{'='*65}")
    print(f"  {station}")
    print(f"{'='*65}")

    path_in = FILE_METEO_PRED_RAW(file_name)
    df      = pl.read_parquet(path_in)

    print(f"  Shape brute : {df.shape[0]:,} lignes x {df.shape[1]} col")

    # ── Noms des colonnes de sortie ───────────────────────────────────────
    # lag_01h_pred_radiation_ctrl, lag_01h_pred_radiation_q10, ...
    out_cols = [
        f"lag_{lag:02d}h_{var_col}_{subtype}"
        for lag in LAGS
        for var_col in PRED_VARS.values()
        for subtype in SUBTYPES
    ]
    print(f"  Colonnes de sortie : {len(out_cols)}")

    # ── Grille 15min ──────────────────────────────────────────────────────
    grid    = make_grid_15()
    ts_list = grid["timestamp_target"].to_list()
    ts_set  = set(ts_list)
    print(f"  Grille 15min : {len(ts_list):,} points")

    # ── Construction du mapping timestamp_target -> valeurs ───────────────
    # Pour chaque ligne du raw :
    #   lag = (timestamp_target - timestamp_emission) en heures
    #   src_col = {var_col}_{subtype}_run{lag:02d}
    #             ex: pred_radiation_ctrl_run31
    #   dst_col = lag_{lag:02d}h_{var_col}_{subtype}
    #             ex: lag_31h_pred_radiation_ctrl
    print(f"\n  Construction du mapping...")

    target_data: dict[object, dict[str, float]] = {}

    for row in df.iter_rows(named=True):
        ts_target = row["timestamp_target"]

        if ts_target not in ts_set:
            continue

        ts_emit = row["timestamp_emission"]
        diff_s  = (ts_target - ts_emit).total_seconds()
        lag_h   = int(round(diff_s / 3600))

        if lag_h < 1 or lag_h > 33:
            continue

        if ts_target not in target_data:
            target_data[ts_target] = {}

        for var_col in PRED_VARS.values():
            for subtype in SUBTYPES:
                # Nom de la colonne dans le fichier raw
                # ex: pred_radiation_ctrl_run31
                src_col = f"{var_col}_{subtype}_run{lag_h:02d}"
                # Nom de la colonne dans le fichier clean
                # ex: lag_31h_pred_radiation_ctrl
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
    for lag in [1, 6, 12, 24, 33]:
        col = f"lag_{lag:02d}h_pred_temperature_ctrl"
        if col not in df_out.columns:
            continue
        n_null  = df_out[col].is_null().sum()
        n_total = df_out.shape[0]
        pct     = (n_total - n_null) / n_total * 100
        print(f"  {col:<45} {n_total - n_null:>8,} valeurs  ({pct:.1f}% rempli)")

    # ── Sauvegarde ────────────────────────────────────────────────────────
    path_out = FILE_METEO_PRED_CLEAN(file_name)
    df_out.write_parquet(path_out)
    size_mb = path_out.stat().st_size / 1024 / 1024
    print(f"\n  -> {path_out.name}  "
          f"({df_out.shape[0]:,} lignes x {df_out.shape[1]} col, "
          f"{size_mb:.1f} MB)")
    print(f"  Periode : {df_out['timestamp_target'].min()} -> "
          f"{df_out['timestamp_target'].max()}")


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)

    for station, file_name in STATIONS.items():
        normalise_pred(station, file_name)

    print(f"\n{'='*65}")
    print("FICHIERS GENERES")
    print(f"{'='*65}")
    for p in sorted(DATA_DIR.glob("meteo_*_pred_clean.parquet")):
        df      = pl.read_parquet(p)
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  {p.name:<55} {df.shape[0]:>9,} lignes  "
              f"{df.shape[1]:>5} col  {size_mb:>6.1f} MB")