"""
Configuration centrale du projet
=================================
Tous les parametres du projet sont definis ici.
Les autres scripts importent depuis ce fichier.
"""

from pathlib import Path

# ════════════════════════════════════════════════════════════════════════════
# CHEMINS
# ════════════════════════════════════════════════════════════════════════════
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"

# ════════════════════════════════════════════════════════════════════════════
# INFLUXDB
# ════════════════════════════════════════════════════════════════════════════
URL    = "https://timeseries.hevs.ch"
ORG    = "HESSOVS"
BUCKET = "MeteoSuisse"
TOKEN  = "ixOI8jiwG1nn6a2MaE1pGa8XCiIJ2rqEX6ZCnluhwAyeZcrT6FHoDgnQhNy5k0YmVrk7hZGPpvb_5aaA-ZxhIw=="

# ════════════════════════════════════════════════════════════════════════════
# PERIODE
# ════════════════════════════════════════════════════════════════════════════
START = "2022-10-01T00:00:00Z"
STOP  = "2025-10-01T00:00:00Z"

# ════════════════════════════════════════════════════════════════════════════
# STATIONS
# (nom exact InfluxDB -> nom fichier)
# ════════════════════════════════════════════════════════════════════════════
STATIONS = {
    "Sion":               "sion",
    "Visp":               "visp",
    "Montagnier, Bagnes": "montagnier_bagnes",
    "Montana":            "crans_montana",
    "Simplon-Dorf":       "simplon_dorf",
    "Les Marécottes":     "les_marecottes",
    "Evionnaz":           "evionnaz",
}

# ════════════════════════════════════════════════════════════════════════════
# RUNS ENSEMBLISTES
# MeteoSuisse fournit 33 runs toutes les 3 heures.
# Chaque run couvre les 33 prochaines heures au pas de 1h.
# ════════════════════════════════════════════════════════════════════════════
RUNS = [f"{i:02d}" for i in range(1, 34)]

# Frequence d'emission des predictions (toutes les 3h)
PRED_EMISSION_FREQ_H = 3

# Horizon de prediction par run (33h)
PRED_HORIZON_H = 33

# ════════════════════════════════════════════════════════════════════════════
# VARIABLES HISTORIQUES
# (nom exact mesure InfluxDB -> nom colonne)
# Pas nominal : 10 minutes
# ════════════════════════════════════════════════════════════════════════════
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

# ════════════════════════════════════════════════════════════════════════════
# VARIABLES DE PREDICTION
# (nom base mesure InfluxDB -> nom colonne)
# Le nom complet dans InfluxDB est {PRED_VARS_KEY}_{subtype}
# ex : PRED_GLOB_ctrl, PRED_GLOB_q10, ...
# Pas nominal : 1 heure
# ════════════════════════════════════════════════════════════════════════════
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

# Sous-types de predictions
# ctrl : valeur centrale du modele
# q10  : quantile 10% (borne basse de l'intervalle de confiance)
# q90  : quantile 90% (borne haute de l'intervalle de confiance)
# stde : ecart-type entre les runs (mesure de l'incertitude)
SUBTYPES = ["ctrl", "q10", "q90", "stde"]

# ════════════════════════════════════════════════════════════════════════════
# PAS DE TEMPS
# ════════════════════════════════════════════════════════════════════════════
OIKEN_STEP_MIN = 15   # pas Oiken en minutes
METEO_HIST_STEP_MIN  = 10   # pas historique MeteoSuisse en minutes
METEO_PRED_STEP_H    = 1    # pas predictions MeteoSuisse en heures
TARGET_STEP_MIN      = 15   # pas cible apres normalisation (toutes sources)

# ════════════════════════════════════════════════════════════════════════════
# NORMALISATION
# ════════════════════════════════════════════════════════════════════════════
# Seuil d'interpolation : trous <= 2h sont interpoled lineairement
# Au-dela, les valeurs restent NaN et sont flaggees {col}_missing = 1
OIKEN_INTERP_MAX_PTS = 8    # 2h / 15min = 8 points
METEO_INTERP_MAX_PTS = 12   # 2h / 10min = 12 points (avant resampling)

# ════════════════════════════════════════════════════════════════════════════
# MODELE — CONTRAINTE TEMPORELLE
# ════════════════════════════════════════════════════════════════════════════
# Le modele predit a 11h heure locale pour le lendemain (0h -> 24h, pas 15min)
# Heure locale Valais : UTC+1 en hiver, UTC+2 en ete
# -> Prediction emise entre 9h et 10h UTC selon la saison
# -> La derniere emission de predictions disponible est celle de 9h UTC
#    (emission toutes les 3h : 0h, 3h, 6h, 9h, 12h, ...)
# -> Le run emis a 9h UTC couvre 9h UTC + 1h -> 9h UTC + 33h
#    soit jusqu'a 18h UTC le lendemain, ce qui couvre bien les 96 pas cibles

PREDICTION_CUTOFF_UTC_H = 9   # derniere emission utilisable (heure UTC)
FORECAST_HORIZON_H      = 33  # horizon couvert par chaque run

# ════════════════════════════════════════════════════════════════════════════
# FICHIERS DE SORTIE
# ════════════════════════════════════════════════════════════════════════════
# Exploration
FILE_EXPLORATION_REPORT = DATA_DIR / "html/exploration_report.html"

# Acquisition (donnees brutes)
FILE_OIKEN_RAW          = DATA_DIR / "oiken_raw.parquet"
FILE_METEO_HIST_RAW     = lambda station: DATA_DIR / f"meteo_{station}_hist_raw.parquet"
FILE_METEO_PRED_RAW     = lambda station: DATA_DIR / f"meteo_{station}_pred_raw.parquet"

# Normalisation (donnees nettoyees et alignees sur 15min UTC)
FILE_OIKEN_CLEAN        = DATA_DIR / "oiken_clean.parquet"
FILE_METEO_HIST_CLEAN   = lambda station: DATA_DIR / f"meteo_{station}_hist_clean.parquet"
FILE_METEO_PRED_CLEAN   = lambda station: DATA_DIR / f"meteo_{station}_pred_clean.parquet"
FILE_CALENDAR           = DATA_DIR / "calendar.parquet"

# Features
FILE_FEATURES           = DATA_DIR / "features.parquet"
FILE_FEATURES_SCHEMA    = DATA_DIR / "features_schema.json"

# Rapports
FILE_NORMALISATION_REPORT = DATA_DIR / "normalisation_report.html"
FILE_FEATURES_REPORT      = DATA_DIR / "features_report.html"