"""
Generation du calendrier
==========================
Ecrit: data/calendar.parquet

Colonnes :
  timestamp           datetime UTC 15min
  hour                0-23 (heure locale Europe/Zurich)
  day_of_week         0=lundi 6=dimanche (heure locale)
  month               1-12 (heure locale)
  is_weekend          0/1
  is_holiday          0/1  jours feries Valais
  is_school_holiday   0/1  vacances scolaires Sion
  hour_sin/cos        encodage cyclique heure
  day_of_year_sin/cos encodage cyclique jour de l'annee
  month_sin/cos       encodage cyclique mois
  day_of_week_sin/cos encodage cyclique jour semaine
"""

import math
from datetime import date, timedelta
from zoneinfo import ZoneInfo

import polars as pl

from config import DATA_DIR, FILE_CALENDAR, TARGET_STEP_MIN

PERIOD_START = pl.datetime(2022, 10,  1,  0, 15, 0, time_unit="us", time_zone="UTC")
PERIOD_END   = pl.datetime(2025,  9, 30,  0,  0, 0, time_unit="us", time_zone="UTC")

TZ_LOCAL = ZoneInfo("Europe/Zurich")


# ── Jours feries valaisans ────────────────────────────────────────────────────
def easter(year: int) -> date:
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


def get_holidays_valais(year: int) -> set:
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


# ── Vacances scolaires Sion ───────────────────────────────────────────────────
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


def build_school_set(periods: list) -> set:
    days = set()
    for start, end in periods:
        d = start
        while d <= end:
            days.add(d)
            d += timedelta(days=1)
    return days


# ── Generation ────────────────────────────────────────────────────────────────
def generate_calendar() -> pl.DataFrame:
    print("Generation de la grille 15min UTC...")

    df = pl.DataFrame({
        "timestamp": pl.datetime_range(
            PERIOD_START, PERIOD_END,
            interval=f"{TARGET_STEP_MIN}m",
            time_unit="us", time_zone="UTC", eager=True,
        )
    })
    print(f"  {df.shape[0]:,} lignes")

    # ── Heure locale (Europe/Zurich) ──────────────────────────────────────
    df = df.with_columns(
        pl.col("timestamp")
          .dt.convert_time_zone("Europe/Zurich")
          .alias("_ts_local")
    )

    # ── Composantes temporelles basees sur l'heure locale ─────────────────
    df = df.with_columns([
        pl.col("_ts_local").dt.hour().alias("hour"),
        pl.col("_ts_local").dt.weekday().alias("day_of_week"),
        pl.col("_ts_local").dt.month().alias("month"),
        pl.col("_ts_local").dt.ordinal_day().alias("_doy"),
        pl.col("_ts_local").dt.date().alias("_date"),
    ])

    # ── Jours feries et vacances scolaires ────────────────────────────────
    all_holidays = set()
    for y in range(2022, 2026):
        all_holidays |= get_holidays_valais(y)
    school_set = build_school_set(SCHOOL_HOLIDAYS)

    print(f"  Jours feries : {len(all_holidays)}")
    print(f"  Jours vacances : {len(school_set)}")

    date_list     = df["_date"].to_list()
    holiday_list  = [1 if d in all_holidays else 0 for d in date_list]
    school_list   = [1 if d in school_set   else 0 for d in date_list]
    weekend_list = [1 if dow >= 6 else 0 for dow in df["day_of_week"].to_list()]

    df = df.with_columns([
        pl.Series("is_weekend",       weekend_list, dtype=pl.Int8),
        pl.Series("is_holiday",       holiday_list, dtype=pl.Int8),
        pl.Series("is_school_holiday",school_list,  dtype=pl.Int8),
    ])

    # ── Encodages cycliques ───────────────────────────────────────────────
    tau = 2 * math.pi
    df = df.with_columns([
        (pl.col("hour").cast(pl.Float64) * (tau / 24)).sin().alias("hour_sin"),
        (pl.col("hour").cast(pl.Float64) * (tau / 24)).cos().alias("hour_cos"),
        (pl.col("month").cast(pl.Float64) * (tau / 12)).sin().alias("month_sin"),
        (pl.col("month").cast(pl.Float64) * (tau / 12)).cos().alias("month_cos"),
        (pl.col("day_of_week").cast(pl.Float64) * (tau / 7)).sin().alias("day_of_week_sin"),
        (pl.col("day_of_week").cast(pl.Float64) * (tau / 7)).cos().alias("day_of_week_cos"),
    ])

    # Encodage cyclique jour de l'annee (avec gestion annees bissextiles)
    doy_vals  = df["_doy"].to_list()
    year_vals = df["_ts_local"].dt.year().to_list()
    doy_sin, doy_cos = [], []
    for doy, yr in zip(doy_vals, year_vals):
        diy = 366 if (yr % 4 == 0 and (yr % 100 != 0 or yr % 400 == 0)) else 365
        doy_sin.append(math.sin(tau * doy / diy))
        doy_cos.append(math.cos(tau * doy / diy))

    df = df.with_columns([
        pl.Series("day_of_year_sin", doy_sin, dtype=pl.Float64),
        pl.Series("day_of_year_cos", doy_cos, dtype=pl.Float64),
    ])

    # ── Nettoyage colonnes intermediaires ─────────────────────────────────
    df = df.drop(["_ts_local", "_doy", "_date"])

    return df


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)

    df = generate_calendar()

    print(f"\nApercu :")
    print(df.head(8))

    print(f"\nVerification jours feries detectes :")
    hol = (df.filter(pl.col("is_holiday") == 1)
             .with_columns(pl.col("timestamp").dt.convert_time_zone("Europe/Zurich")
                             .dt.date().alias("date"))
             ["date"].unique().sort().to_list())
    for d in hol:
        print(f"  {d}")

    print(f"\nVerification vacances scolaires : "
          f"{df.filter(pl.col('is_school_holiday') == 1).with_columns(pl.col('timestamp').dt.convert_time_zone('Europe/Zurich').dt.date().alias('date'))['date'].n_unique()} jours distincts")

    print(f"\nRepartition is_weekend :")
    print(df.group_by("is_weekend").agg(pl.len().alias("count")).sort("is_weekend"))

    FILE_CALENDAR.parent.mkdir(exist_ok=True)
    df.write_parquet(FILE_CALENDAR)
    size_mb = FILE_CALENDAR.stat().st_size / 1024 / 1024
    print(f"\n-> {FILE_CALENDAR.name}  "
          f"({df.shape[0]:,} lignes x {df.shape[1]} col, {size_mb:.1f} MB)")