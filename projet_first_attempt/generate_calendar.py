import math
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
import polars as pl

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Recherche de Pâques avec l'algorithme de Meeus

def easter(year):
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day   = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)

#Recherche des jour fériés en Valais à partir de Pâques

def get_holidays_valais(year):
    e = easter(year)
    return {
        date(year,  1,  1),
        date(year,  3, 19),
        date(year,  5,  1),
        date(year,  8,  1),
        date(year,  8, 15),
        date(year, 11,  1),
        date(year, 12,  8),
        date(year, 12, 25),
        date(year, 12, 26),
        e - timedelta(days=2),
        e + timedelta(days=1),
        e + timedelta(days=39),
        e + timedelta(days=50),
        e + timedelta(days=60),
    }

#Vacances scolaires de la période observées

SCHOOL_HOLIDAYS = [
    (date(2022,  4,  9), date(2022,  4, 24)),
    (date(2022,  6, 30), date(2022,  8, 14)),
    (date(2022, 10,  1), date(2022, 10, 16)),
    (date(2022, 12, 24), date(2023,  1,  8)),
    (date(2023,  2, 11), date(2023,  2, 19)),
    (date(2023,  4,  1), date(2023,  4, 16)),
    (date(2023,  6, 29), date(2023,  8, 13)),
    (date(2023,  9, 30), date(2023, 10, 15)),
    (date(2023, 12, 23), date(2024,  1,  7)),
    (date(2024,  2, 10), date(2024,  2, 18)),
    (date(2024,  3, 28), date(2024,  4, 14)),
    (date(2024,  6, 29), date(2024,  8, 11)),
    (date(2024, 10,  5), date(2024, 10, 20)),
    (date(2024, 12, 21), date(2025,  1,  5)),
    (date(2025,  2, 15), date(2025,  2, 23)),
    (date(2025,  4,  5), date(2025,  4, 20)),
    (date(2025,  6, 28), date(2025,  8, 17)),
]

#Création d'un set contenant tous les jours de vacances scolaires à paritr de la liste SCHOOL_HOLIDAYS

def build_school_set(periods):
    days = set()
    for start, end in periods:
        d = start
        while d <= end:
            days.add(d)
            d += timedelta(days=1)
    return days

#Génération des timestamps

def generate_calendar(
    start = datetime(2022, 10,  1,  0,  0, 0, tzinfo=timezone.utc),
    end   = datetime(2025,  9, 30, 23, 45, 0, tzinfo=timezone.utc),
):
    df = pl.DataFrame({
        "timestamp": pl.datetime_range(
            start, end, interval="15m",
            time_unit="us", time_zone="UTC", eager=True
        )
    })

#Ajout des colonnes
    df = df.with_columns([
        pl.col("timestamp").dt.hour().alias("hour"),
        pl.col("timestamp").dt.weekday().alias("day_of_week"),
        pl.col("timestamp").dt.month().alias("month"),
        pl.col("timestamp").dt.ordinal_day().alias("_doy"),
        pl.col("timestamp").dt.date().alias("_date"),
    ])

#Génération des set de jours spéciaux
    all_holidays = set()
    for y in range(start.year, end.year + 1):
        all_holidays |= get_holidays_valais(y)
    school_set = build_school_set(SCHOOL_HOLIDAYS)

    holiday_list = [1 if d in all_holidays else 0 for d in df["_date"].to_list()]
    school_list  = [1 if d in school_set  else 0 for d in df["_date"].to_list()]

#Ajout de colonnes de jours spéciaux
    df = df.with_columns([
        pl.Series("is_weekend",        [1 if dow >= 5 else 0 for dow in df["day_of_week"].to_list()], dtype=pl.Int8),
        pl.Series("is_holiday",        holiday_list, dtype=pl.Int8),
        pl.Series("is_school_holiday", school_list,  dtype=pl.Int8),
    ])

#Ajout de colonnes à signaux sin et cos (heure, mois et semaine)
    tau = 2 * math.pi
    df = df.with_columns([
        (pl.col("hour").cast(pl.Float64) * (tau / 24)).sin().alias("hour_sin"),
        (pl.col("hour").cast(pl.Float64) * (tau / 24)).cos().alias("hour_cos"),
        (pl.col("month").cast(pl.Float64) * (tau / 12)).sin().alias("month_sin"),
        (pl.col("month").cast(pl.Float64) * (tau / 12)).cos().alias("month_cos"),
        (pl.col("day_of_week").cast(pl.Float64) * (tau / 7)).sin().alias("day_of_week_sin"),
        (pl.col("day_of_week").cast(pl.Float64) * (tau / 7)).cos().alias("day_of_week_cos"),
    ])
#Calcul des jours dans l'années 
    doy_vals  = df["_doy"].to_list()
    year_vals = df["timestamp"].dt.year().to_list()
    doy_sin, doy_cos = [], []
    for doy, yr in zip(doy_vals, year_vals):
        diy = 366 if (yr % 4 == 0 and (yr % 100 != 0 or yr % 400 == 0)) else 365
        doy_sin.append(math.sin(tau * doy / diy))
        doy_cos.append(math.cos(tau * doy / diy))

#Ajout de colonnes à signaux sin et cos (jour dans l'année)
    df = df.with_columns([
        pl.Series("day_of_year_sin", doy_sin, dtype=pl.Float64),
        pl.Series("day_of_year_cos", doy_cos, dtype=pl.Float64),
    ])

#Suppression des colonnes de construction
    df = df.drop(["_doy", "_date"])

    return df

#Lancement du code
if __name__ == "__main__":
    df = generate_calendar()
    path = OUTPUT_DIR / "calendar.parquet"
    df.write_parquet(path)
    print(f"Exporte -> {path}  ({df.shape[0]:,} lignes x {df.shape[1]} col)")