import certifi
import polars as pl
from pathlib import Path
from influxdb_client import InfluxDBClient

#Configuration globale pour la recherche de donnée sur l'API

URL    = "https://timeseries.hevs.ch"
ORG    = "HESSOVS"
BUCKET = "MeteoSuisse"
TOKEN  = "ixOI8jiwG1nn6a2MaE1pGa8XCiIJ2rqEX6ZCnluhwAyeZcrT6FHoDgnQhNy5k0YmVrk7hZGPpvb_5aaA-ZxhIw=="

START = "2022-10-01T00:00:00Z"
STOP  = "2025-10-01T00:00:00Z"

OUTPUT_DIR = Path("data")

RUNS = [f"{i:02d}" for i in range(1, 34)]

STATIONS = {
    "Sion":               "sion",
    "Visp":               "visp",
    "Montagnier, Bagnes": "montagnier_bagnes",
    "Montana":            "crans_montana",
    "Simplon-Dorf":       "simplon_dorf",
    "Les Marécottes":     "les_marecottes",
    "Evionnaz":           "evionnaz",
}

HISTORICAL = {
    "Air temperature 2m above ground (current value)":       "hist_temperature",
    "Global radiation (ten minutes mean)":                   "hist_radiation",
    "Sunshine duration (ten minutes total)":                 "hist_sunshine",
    "Precipitation (ten minutes total)":                     "hist_precipitation",
    "Relative air humidity 2m above ground (current value)": "hist_humidity",
    "Atmospheric pressure at barometric altitude":           "hist_pressure",
    "Wind speed scalar (ten minutes mean)":                  "hist_wind_speed",
    "Wind Direction (ten minutes mean)":                     "hist_wind_dir",
    "Gust peak (one second) (maximum)":                      "hist_gust",
}

PRED_VARS = {
    "PRED_GLOB":      "pred_radiation",
    "PRED_DURSUN":    "pred_sunshine",
    "PRED_T_2M":      "pred_temperature",
    "PRED_TOT_PREC":  "pred_precipitation",
    "PRED_RELHUM_2M": "pred_humidity",
    "PRED_PS":        "pred_pressure",
    "PRED_FF_10M":    "pred_wind_speed",
    "PRED_DD_10M":    "pred_wind_dir",
}
SUBTYPES = ["ctrl", "q10", "q90", "stde"]

#Connexion à InfluxDB

def make_client() -> InfluxDBClient:
    return InfluxDBClient(
        url=URL, token=TOKEN, org=ORG,
        ssl_ca_cert=certifi.where(),
        timeout=1_000_000
    )

#Construction de la requête en language Flux

def build_query(measurement: str, site: str, run_filter: str = "") -> str:
    q = (
        'from(bucket: "' + BUCKET + '")'
        + ' |> range(start: ' + START + ', stop: ' + STOP + ')'
        + ' |> filter(fn: (r) => r["_measurement"] == "' + measurement + '")'
        + ' |> filter(fn: (r) => r["Site"] == "' + site + '")'
    )
    if run_filter:
        q += ' |> filter(fn: (r) => ' + run_filter + ')'
    q += ' |> keep(columns: ["_time", "_value", "Prediction"])'
    return q

#Execution de la requête en language Flux

def query_to_df(api, measurement: str, site: str,
                col_name: str, run_filter: str = "") -> pl.DataFrame:
    q = build_query(measurement, site, run_filter)
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
        "run_id":    pl.Series(preds, dtype=pl.Utf8),
    })

#Acquisition des données historique

def acquire_historical(api, site: str) -> pl.DataFrame:
    dfs = []
    for measurement, col_name in HISTORICAL.items():
        df = query_to_df(api, measurement, site, col_name)
        if df.shape[0] == 0:
            continue
        dfs.append(df.drop("run_id"))

    if not dfs:
        return pl.DataFrame()

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.join(df, on="timestamp", how="full", coalesce=True)
    return merged.sort("timestamp")

#Acquisition des données de prédiction

def acquire_predictions_wide(api, site: str) -> pl.DataFrame:
    runs_filter = " or ".join([f'r["Prediction"] == "{r}"' for r in RUNS])

    long_data: dict[tuple, dict[str, float]] = {}

#Collecte les données en format long
    for var_key, col_base in PRED_VARS.items():
        for subtype in SUBTYPES:
            measurement = f"{var_key}_{subtype}"
            df = query_to_df(api, measurement, site,
                             col_name=f"{col_base}_{subtype}",
                             run_filter=runs_filter)
            if df.shape[0] == 0:
                continue

            col = f"{col_base}_{subtype}"
            for row in df.iter_rows(named=True):
                key = (row["timestamp"], row["run_id"])
                if key not in long_data:
                    long_data[key] = {}
                long_data[key][col] = row[col]

    if not long_data:
        return pl.DataFrame()
    
    #Construction du DataFrame
    all_wide_cols = [
        f"{col_base}_{subtype}_run{run}"
        for col_base in PRED_VARS.values()
        for subtype in SUBTYPES
        for run in RUNS
    ]
    #Pivote au format wide
    wide_data: dict = {}
    for (ts, run_id), values in long_data.items():
        if ts not in wide_data:
            wide_data[ts] = {}
        for col, val in values.items():
            wide_col = f"{col}_run{run_id}"
            wide_data[ts][wide_col] = val

    timestamps = sorted(wide_data.keys())
    records = []
    for ts in timestamps:
        row = {"timestamp_target": ts}
        for col in all_wide_cols:
            row[col] = wide_data[ts].get(col, None)
        records.append(row)

    schema = {"timestamp_target": pl.Datetime("us", "UTC")}
    for col in all_wide_cols:
        schema[col] = pl.Float64

    return pl.DataFrame(records, schema=schema).sort("timestamp_target")

#Acquisition générale météo par site
def process_site(api, site: str, file_name: str):
    df_hist = acquire_historical(api, site)
    df_pred = acquire_predictions_wide(api, site)

    path = OUTPUT_DIR / f"meteo_{file_name}.parquet"

    if df_hist.is_empty() and df_pred.is_empty():
        return

    if not df_hist.is_empty() and not df_pred.is_empty():
        df_pred_renamed = df_pred.rename({"timestamp_target": "timestamp"})
        merged = df_hist.join(df_pred_renamed, on="timestamp", how="full", coalesce=True)
        merged = merged.sort("timestamp")
    elif not df_hist.is_empty():
        merged = df_hist
    else:
        merged = df_pred.rename({"timestamp_target": "timestamp"})

    merged.write_parquet(path)

#Acquisition des données de Oiken
def process_oiken(csv_path: str = "Data_Oiken.csv"):
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
    #Passage au format UTC
    df = df.with_columns(
        pl.col("timestamp")
          .str.strptime(pl.Datetime("us"), "%d/%m/%Y %H:%M")
          .dt.replace_time_zone("UTC")
          .alias("timestamp")
    )
    #Renomination des colonnes
    df = df.rename({
        "standardised load [-]":                 "load",
        "standardised forecast load [-]":        "forecast_load",
        "central valais solar production [kWh]": "pv_central_valais",
        "sion area solar production [kWh]":      "pv_sion",
        "sierre area production [kWh]":          "pv_sierre",
        "remote solar production [kWh]":         "pv_remote_raw",
    })

    #Supression de l'offset min du PV remote
    night_min = (df
        .filter(df["timestamp"].dt.hour() < 4)["pv_remote_raw"]
        .min())

    df = df.with_columns(
        (pl.col("pv_remote_raw") - night_min)
          .clip(lower_bound=0)
          .alias("pv_remote")
    ).drop("pv_remote_raw")

    df = df.with_columns(
        (pl.col("pv_central_valais") + pl.col("pv_sion") +
         pl.col("pv_sierre")         + pl.col("pv_remote"))
          .alias("pv_total")
    )

    df = df.sort("timestamp")
    path = OUTPUT_DIR / "oiken.parquet"
    df.write_parquet(path)

#Bouclage des données sur toutes les stations

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    process_oiken("Data_Oiken.csv")

    client = make_client()
    api    = client.query_api()

    available = [r["_value"]
                 for t in client.query_api().query(
                     org=ORG,
                     query='import "influxdata/influxdb/schema"\nschema.tagValues(bucket: "' + BUCKET + '", tag: "Site")'
                 )
                 for r in t.records]

    for site, file_name in STATIONS.items():
        if site not in available:
            matches = [s for s in available if site.split(",")[0].strip().lower() in s.lower()]
            if matches:
                site = matches[0]
            else:
                continue
        process_site(api, site, file_name)

    client.close()

if __name__ == "__main__":
    main()