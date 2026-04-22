"""
Acquisition des donnees brutes
================================
Lit  : InfluxDB MeteoSuisse + Data_Oiken.csv
Ecrit: data/oiken_raw.parquet
       data/meteo_{station}_hist_raw.parquet
       data/meteo_{station}_pred_raw.parquet

Aucune transformation des donnees — sauvegarde brute.
"""

from pathlib import Path
import polars as pl
import certifi
from influxdb_client import InfluxDBClient
from datetime import timedelta

from config import (
    URL, TOKEN, ORG, BUCKET,
    START, STOP,
    STATIONS, HISTORICAL, PRED_VARS, SUBTYPES, RUNS,
    DATA_DIR,
    FILE_OIKEN_RAW,
    FILE_METEO_HIST_RAW,
    FILE_METEO_PRED_RAW,
)

# ════════════════════════════════════════════════════════════════════════════
# HELPERS INFLUXDB
# ════════════════════════════════════════════════════════════════════════════
def make_client() -> InfluxDBClient:
    return InfluxDBClient(
        url=URL, token=TOKEN, org=ORG,
        ssl_ca_cert=certifi.where(),
        timeout=1_000_000,
    )


def build_query(measurement: str, site: str, run_filter: str = "") -> str:
    q = (
        f'from(bucket: "{BUCKET}")'
        f' |> range(start: {START}, stop: {STOP})'
        f' |> filter(fn: (r) => r["_measurement"] == "{measurement}")'
        f' |> filter(fn: (r) => r["Site"] == "{site}")'
    )
    if run_filter:
        q += f' |> filter(fn: (r) => {run_filter})'
    q += ' |> keep(columns: ["_time", "_value", "Prediction"])'
    return q


def query_to_df(api, measurement: str, site: str,
                col_name: str, run_filter: str = "") -> pl.DataFrame:
    q      = build_query(measurement, site, run_filter)
    tables = api.query(org=ORG, query=q)

    times, values, preds = [], [], []
    for table in tables:
        for record in table.records:
            times.append(record["_time"])
            values.append(record["_value"])
            preds.append(record.values.get("Prediction", None))

    if not times:
        return pl.DataFrame(schema={
            "timestamp": pl.Datetime("us", "UTC"),
            col_name:    pl.Float64,
            "run_id":    pl.Utf8,
        })

    return pl.DataFrame({
        "timestamp": pl.Series(times).cast(pl.Datetime("us", "UTC")),
        col_name:    pl.Series(values, dtype=pl.Float64),
        "run_id":    pl.Series(preds,  dtype=pl.Utf8),
    })


# ════════════════════════════════════════════════════════════════════════════
# ACQUISITION OIKEN
# ════════════════════════════════════════════════════════════════════════════
def acquire_oiken(csv_path: str = "Data_Oiken.csv") -> None:
    """
    Charge le CSV Oiken et sauvegarde en Parquet brut.
    Seul le parsing du timestamp et le renommage des colonnes sont effectues.
    Aucune correction de valeurs.
    """
    print("\n" + "="*65)
    print("  OIKEN")
    print("="*65)

    df = pl.read_csv(
        csv_path,
        separator=",",
        try_parse_dates=False,
        null_values=["#N/A", "N/A", "NA", "", "null", "NULL"],
        schema_overrides={
            "standardised load [-]":                 pl.Float64,
            "standardised forecast load [-]":        pl.Float64,
            "central valais solar production [kWh]": pl.Float64,
            "sion area solar production [kWh]":      pl.Float64,
            "sierre area production [kWh]":          pl.Float64,
            "remote solar production [kWh]":         pl.Float64,
        }
    )

    df = df.with_columns(
        pl.col("timestamp")
          .str.strptime(pl.Datetime("us"), "%d/%m/%Y %H:%M")
          .dt.replace_time_zone("UTC")
          .alias("timestamp")
    ).rename({
        "standardised load [-]":                 "load",
        "standardised forecast load [-]":        "forecast_load",
        "central valais solar production [kWh]": "pv_central_valais",
        "sion area solar production [kWh]":      "pv_sion",
        "sierre area production [kWh]":          "pv_sierre",
        "remote solar production [kWh]":         "pv_remote",
    }).sort("timestamp")

    df.write_parquet(FILE_OIKEN_RAW)
    size_mb = FILE_OIKEN_RAW.stat().st_size / 1024 / 1024
    print(f"  {FILE_OIKEN_RAW.name}  "
          f"({df.shape[0]:,} lignes x {df.shape[1]} col, {size_mb:.1f} MB)")
    print(f"  Periode : {df['timestamp'].min()} -> {df['timestamp'].max()}")


# ════════════════════════════════════════════════════════════════════════════
# ACQUISITION HISTORIQUE
# ════════════════════════════════════════════════════════════════════════════
def acquire_historical(api, site: str, file_name: str) -> None:
    """
    Telecharge les donnees historiques d'une station et sauvegarde en Parquet.
    Format : une ligne par timestamp, une colonne par variable.
    """
    print(f"\n  [HIST] {site}")

    dfs = []
    for measurement, col_name in HISTORICAL.items():
        df = query_to_df(api, measurement, site, col_name)
        n  = df.shape[0]
        print(f"    {col_name:<28} {n:>9,} points")

        if n == 0:
            continue

        dfs.append(df.drop("run_id"))

    if not dfs:
        print(f"    Aucune donnee pour {site} — fichier non cree")
        return

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.join(df, on="timestamp", how="full", coalesce=True)
    merged = merged.sort("timestamp")

    path = FILE_METEO_HIST_RAW(file_name)
    merged.write_parquet(path)
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"    -> {path.name}  "
          f"({merged.shape[0]:,} lignes x {merged.shape[1]} col, {size_mb:.1f} MB)")


# ════════════════════════════════════════════════════════════════════════════
# ACQUISITION PREDICTIONS
# ════════════════════════════════════════════════════════════════════════════
def acquire_predictions(api, site: str, file_name: str) -> None:
    """
    Telecharge les predictions d'une station et sauvegarde en Parquet.

    Format long d'abord : (timestamp_emission, timestamp_target, run_id, variable, valeur)
    Puis pivot vers format wide :
      une ligne par (timestamp_emission, timestamp_target)
      une colonne par (variable_subtype_runXX)

    timestamp_emission : quand la prediction a ete emise (toutes les 3h)
    timestamp_target   : l'heure predite
    """
    print(f"\n  [PRED] {site}")

    runs_filter = " or ".join([f'r["Prediction"] == "{r}"' for r in RUNS])

    # Collecte en format long :
    # {(timestamp_emission, timestamp_target, run_id) -> {col: value}}
    # Dans InfluxDB :
    #   _time      = timestamp_target  (l'heure predite)
    #   Prediction = run_id
    # Le timestamp d'emission est deduit : emission = target - run_id * 1h
    # (le run 01 predit dans 1h, le run 02 dans 2h, etc.)
    long_data: dict[tuple, dict[str, float]] = {}

    for var_key, col_base in PRED_VARS.items():
        for subtype in SUBTYPES:
            measurement = f"{var_key}_{subtype}"
            col         = f"{col_base}_{subtype}"
            print(f"    {col:<35} ...", end=" ", flush=True)

            df = query_to_df(api, measurement, site,
                             col_name=col,
                             run_filter=runs_filter)
            n = df.shape[0]
            print(f"{n:>9,} points")

            if n == 0:
                continue

            for row in df.iter_rows(named=True):
                ts_target = row["timestamp"]
                run_id    = row["run_id"]

                # Calculer le timestamp d'emission :
                # run_id "01" -> target - 1h, "02" -> target - 2h, etc.
                run_num   = int(run_id)
                ts_emit = ts_target - timedelta(hours=run_num)

                key = (ts_emit, ts_target, run_id)
                if key not in long_data:
                    long_data[key] = {}
                long_data[key][col] = row[col]

    if not long_data:
        print(f"    Aucune donnee pour {site} — fichier non cree")
        return

    # Construire toutes les colonnes wide attendues
    all_wide_cols = [
        f"{col_base}_{subtype}_run{run}"
        for col_base in PRED_VARS.values()
        for subtype   in SUBTYPES
        for run       in RUNS
    ]

    # Pivot : une ligne par (ts_emit, ts_target)
    wide_data: dict[tuple, dict[str, float]] = {}
    for (ts_emit, ts_target, run_id), values in long_data.items():
        key = (ts_emit, ts_target)
        if key not in wide_data:
            wide_data[key] = {}
        for col, val in values.items():
            wide_col = f"{col}_run{run_id}"
            wide_data[key][wide_col] = val

    # Construire le DataFrame
    keys      = sorted(wide_data.keys())
    records   = []
    for ts_emit, ts_target in keys:
        row = {
            "timestamp_emission": ts_emit,
            "timestamp_target":   ts_target,
        }
        for col in all_wide_cols:
            row[col] = wide_data[(ts_emit, ts_target)].get(col, None)
        records.append(row)

    schema = {
        "timestamp_emission": pl.Datetime("us", "UTC"),
        "timestamp_target":   pl.Datetime("us", "UTC"),
    }
    for col in all_wide_cols:
        schema[col] = pl.Float64

    df_wide = pl.DataFrame(records, schema=schema)
    df_wide = df_wide.sort(["timestamp_emission", "timestamp_target"])

    path = FILE_METEO_PRED_RAW(file_name)
    df_wide.write_parquet(path)
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"\n    -> {path.name}  "
          f"({df_wide.shape[0]:,} lignes x {df_wide.shape[1]} col, {size_mb:.1f} MB)")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)

    # ── Oiken ─────────────────────────────────────────────────────────────
    acquire_oiken("Data_Oiken.csv")

    # ── MeteoSuisse ───────────────────────────────────────────────────────
    print("\nConnexion a InfluxDB...")
    client = make_client()
    api    = client.query_api()

    for site, file_name in STATIONS.items():
        print(f"\n{'='*65}")
        print(f"  Station : {site}")
        print(f"{'='*65}")
        acquire_historical(api, site, file_name)
        acquire_predictions(api, site, file_name)

    client.close()

    # ── Resume ────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("FICHIERS GENERES")
    print(f"{'='*65}")
    for p in sorted(DATA_DIR.glob("*_raw.parquet")):
        df      = pl.read_parquet(p)
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  {p.name:<50} {df.shape[0]:>9,} lignes  "
              f"{df.shape[1]:>5} col  {size_mb:>6.1f} MB")