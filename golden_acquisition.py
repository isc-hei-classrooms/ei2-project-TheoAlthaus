"""
Acquisition golden dataset
===========================
Lit  : golden_data/oiken-golden-dataset.csv
       InfluxDB MeteoSuisse (station Sion uniquement)
Ecrit: golden_data/oiken_golden_raw.parquet
       golden_data/meteo_sion_hist_golden_raw.parquet
       golden_data/meteo_sion_pred_golden_raw.parquet

Periode : 2025-09-30 00:00 UTC -> fin des donnees
Variables meteo : temperature, radiation, sunshine, humidity
Predictions     : ctrl + stde uniquement
"""

from pathlib import Path
import polars as pl
import certifi
from influxdb_client import InfluxDBClient
from datetime import timedelta

from config import URL, TOKEN, ORG, BUCKET, RUNS

# ── Parametres golden dataset ─────────────────────────────────────────────
GOLDEN_DIR   = Path(__file__).parent / "golden_data"
GOLDEN_START = "2025-09-30T00:00:00Z"
GOLDEN_STOP  = "2026-04-22T00:15:00Z"

GOLDEN_CSV   = GOLDEN_DIR / "oiken-golden-dataset.csv"

FILE_OIKEN_GOLDEN    = GOLDEN_DIR / "oiken_golden_raw.parquet"
FILE_METEO_HIST_GOLDEN = GOLDEN_DIR / "meteo_sion_hist_golden_raw.parquet"
FILE_METEO_PRED_GOLDEN = GOLDEN_DIR / "meteo_sion_pred_golden_raw.parquet"

# Station unique
SITE      = "Sion"
FILE_NAME = "sion"

# Variables historiques a acquerir
HISTORICAL_GOLDEN = {
    "Air temperature 2m above ground (current value)":       "hist_temperature",
    "Global radiation (ten minutes mean)":                   "hist_radiation",
    "Sunshine duration (ten minutes total)":                 "hist_sunshine",
    "Relative air humidity 2m above ground (current value)": "hist_humidity",
}

# Variables de prevision a acquerir
PRED_VARS_GOLDEN = {
    "PRED_T_2M":      "pred_temperature",
    "PRED_GLOB":      "pred_radiation",
    "PRED_DURSUN":    "pred_sunshine",
    "PRED_RELHUM_2M": "pred_humidity",
}

# Sous-types uniquement ctrl et stde
SUBTYPES_GOLDEN = ["ctrl", "stde"]


# ── Helpers InfluxDB ──────────────────────────────────────────────────────
def make_client() -> InfluxDBClient:
    return InfluxDBClient(
        url=URL, token=TOKEN, org=ORG,
        ssl_ca_cert=certifi.where(),
        timeout=1_000_000,
    )


def build_query(measurement: str, site: str,
                run_filter: str = "") -> str:
    q = (
        f'from(bucket: "{BUCKET}")'
        f' |> range(start: {GOLDEN_START}, stop: {GOLDEN_STOP})'
        f' |> filter(fn: (r) => r["_measurement"] == "{measurement}")'
        f' |> filter(fn: (r) => r["Site"] == "{site}")'
    )
    if run_filter:
        q += f' |> filter(fn: (r) => {run_filter})'
    q += ' |> keep(columns: ["_time", "_value", "Prediction"])'
    return q


def query_to_df(api, measurement: str, site: str,
                col_name: str,
                run_filter: str = "") -> pl.DataFrame:
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


# ── Acquisition Oiken ─────────────────────────────────────────────────────
def acquire_oiken_golden() -> None:
    print("\n" + "="*65)
    print("  OIKEN GOLDEN")
    print("="*65)

    df = pl.read_csv(
        str(GOLDEN_CSV),
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

    # Filtrer uniquement les nouvelles donnees
    golden_start = pl.datetime(2025, 9, 30, 0, 0, 0,
                                time_unit="us", time_zone="UTC")
    df = df.filter(pl.col("timestamp") >= golden_start)

    df.write_parquet(FILE_OIKEN_GOLDEN)
    size_mb = FILE_OIKEN_GOLDEN.stat().st_size / 1024 / 1024
    print(f"  {FILE_OIKEN_GOLDEN.name}  "
          f"({df.shape[0]:,} lignes x {df.shape[1]} col, {size_mb:.1f} MB)")
    print(f"  Periode : {df['timestamp'].min()} -> {df['timestamp'].max()}")


# ── Acquisition historique Sion ───────────────────────────────────────────
def acquire_historical_golden(api) -> None:
    print(f"\n{'='*65}")
    print(f"  [HIST] {SITE} — golden")
    print(f"{'='*65}")

    dfs = []
    for measurement, col_name in HISTORICAL_GOLDEN.items():
        df = query_to_df(api, measurement, SITE, col_name)
        n  = df.shape[0]
        print(f"    {col_name:<28} {n:>9,} points")

        if n == 0:
            continue

        dfs.append(df.drop("run_id"))

    if not dfs:
        print(f"    Aucune donnee pour {SITE} — fichier non cree")
        return

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.join(df, on="timestamp", how="full", coalesce=True)
    merged = merged.sort("timestamp")

    merged.write_parquet(FILE_METEO_HIST_GOLDEN)
    size_mb = FILE_METEO_HIST_GOLDEN.stat().st_size / 1024 / 1024
    print(f"\n  -> {FILE_METEO_HIST_GOLDEN.name}  "
          f"({merged.shape[0]:,} lignes x {merged.shape[1]} col, "
          f"{size_mb:.1f} MB)")
    print(f"  Periode : {merged['timestamp'].min()} -> "
          f"{merged['timestamp'].max()}")


# ── Acquisition predictions Sion ──────────────────────────────────────────
def acquire_predictions_golden(api) -> None:
    print(f"\n{'='*65}")
    print(f"  [PRED] {SITE} — golden")
    print(f"{'='*65}")

    runs_filter = " or ".join([f'r["Prediction"] == "{r}"' for r in RUNS])

    long_data: dict[tuple, dict[str, float]] = {}

    for var_key, col_base in PRED_VARS_GOLDEN.items():
        for subtype in SUBTYPES_GOLDEN:
            measurement = f"{var_key}_{subtype}"
            col         = f"{col_base}_{subtype}"
            print(f"    {col:<35} ...", end=" ", flush=True)

            df = query_to_df(api, measurement, SITE,
                             col_name=col,
                             run_filter=runs_filter)
            n = df.shape[0]
            print(f"{n:>9,} points")

            if n == 0:
                continue

            for row in df.iter_rows(named=True):
                ts_target = row["timestamp"]
                run_id    = row["run_id"]
                run_num   = int(run_id)
                ts_emit   = ts_target - timedelta(hours=run_num)

                key = (ts_emit, ts_target, run_id)
                if key not in long_data:
                    long_data[key] = {}
                long_data[key][col] = row[col]

    if not long_data:
        print(f"    Aucune donnee pour {SITE} — fichier non cree")
        return

    # Colonnes wide : uniquement ctrl et stde
    all_wide_cols = [
        f"{col_base}_{subtype}_run{run}"
        for col_base in PRED_VARS_GOLDEN.values()
        for subtype   in SUBTYPES_GOLDEN
        for run       in RUNS
    ]

    wide_data: dict[tuple, dict[str, float]] = {}
    for (ts_emit, ts_target, run_id), values in long_data.items():
        key = (ts_emit, ts_target)
        if key not in wide_data:
            wide_data[key] = {}
        for col, val in values.items():
            wide_col = f"{col}_run{run_id}"
            wide_data[key][wide_col] = val

    keys    = sorted(wide_data.keys())
    records = []
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

    df_wide.write_parquet(FILE_METEO_PRED_GOLDEN)
    size_mb = FILE_METEO_PRED_GOLDEN.stat().st_size / 1024 / 1024
    print(f"\n  -> {FILE_METEO_PRED_GOLDEN.name}  "
          f"({df_wide.shape[0]:,} lignes x {df_wide.shape[1]} col, "
          f"{size_mb:.1f} MB)")
    print(f"  Periode : {df_wide['timestamp_target'].min()} -> "
          f"{df_wide['timestamp_target'].max()}")


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    GOLDEN_DIR.mkdir(exist_ok=True)

    # Oiken (depuis CSV)
    acquire_oiken_golden()

    # MeteoSuisse (depuis InfluxDB)
    print("\nConnexion a InfluxDB...")
    client = make_client()
    api    = client.query_api()

    acquire_historical_golden(api)
    acquire_predictions_golden(api)

    client.close()

    # Resume
    print(f"\n{'='*65}")
    print("FICHIERS GENERES")
    print(f"{'='*65}")
    for p in sorted(GOLDEN_DIR.glob("*_raw.parquet")):
        df      = pl.read_parquet(p)
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  {p.name:<50} {df.shape[0]:>9,} lignes  "
              f"{df.shape[1]:>5} col  {size_mb:>6.1f} MB")