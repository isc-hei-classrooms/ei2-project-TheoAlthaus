"""
Feature Engineering — Day-Ahead Model (prediction a 11h pour J+1)
==================================================================
Construit la matrice d'entrainement :
  - Une ligne par (jour_emission, quart_heure_cible)
  - Emission fixee a 11h chaque jour
  - Cible : les 96 quarts d'heure de J+1

Blocs de features :
  1. Calendaire de l'heure cible (hour_sin/cos, dow, month, holidays...)
  2. Meteo Sion — run ctrl le plus recent disponible a 11h (tag minimal)
  3. Lags du load (J-1 meme heure, J-7 meme heure, moyennes journalieres)
  4. Lags PV (J-1 meme heure, J-7 meme heure)

Lit  : data/oiken_clean.parquet (ou oiken.parquet)
       data/meteo_sion_clean.parquet (ou meteo_sion.parquet)
       data/calendar.parquet
Ecrit: data/features_day_ahead.parquet

Dependances : pip install polars pyarrow
Execution   : python feature_engineering.py
"""

from pathlib import Path
import polars as pl

DATA_DIR = Path("data")

# ── Constantes ────────────────────────────────────────────────────────────────
EMISSION_HOUR  = 11          # heure fixe de prediction (11h UTC)
STEP_MIN       = 15          # pas de temps Oiken (minutes)
STEPS_PER_DAY  = 96          # 24h / 15min
STEPS_PER_HOUR = 4           # 1h / 15min

# Tags disponibles dans MeteoSuisse (runs emis toutes les 3h)
# Tag = horizon en heures entre emission du run et heure cible
# Les runs vont de '01' a '33'
AVAILABLE_TAGS = list(range(1, 34))   # 1, 2, ..., 33

# Variables meteo a prendre (ctrl uniquement pour l'instant)
METEO_VARS = [
    "pred_radiation_ctrl",
    "pred_sunshine_ctrl",
    "pred_temperature_ctrl",
    "pred_humidity_ctrl",
]


# ════════════════════════════════════════════════════════════════════════════
# CHARGEMENT
# ════════════════════════════════════════════════════════════════════════════
def load_oiken() -> pl.DataFrame:
    for name in ["oiken_clean.parquet", "oiken.parquet"]:
        p = DATA_DIR / name
        if p.exists():
            df = pl.read_parquet(p).sort("timestamp")
            print(f"  Oiken  : {p.name}  ({df.shape[0]:,} lignes)")
            return df
    raise FileNotFoundError("Aucun fichier oiken*.parquet dans data/")


def load_calendar() -> pl.DataFrame:
    p = DATA_DIR / "calendar.parquet"
    if not p.exists():
        raise FileNotFoundError("calendar.parquet introuvable dans data/")
    df = pl.read_parquet(p).sort("timestamp")
    print(f"  Calendrier : {p.name}  ({df.shape[0]:,} lignes)")
    return df


def load_meteo_predictions() -> pl.DataFrame:
    """
    Charge le fichier meteo Sion et extrait les colonnes de prediction.
    Format attendu : colonnes pred_{var}_ctrl_run{XX}
    ex: pred_radiation_ctrl_run01, pred_radiation_ctrl_run13, ...
    """
    for name in ["meteo_sion_clean.parquet",
                 "meteo_sion_historical.parquet",
                 "meteo_sion.parquet"]:
        p = DATA_DIR / name
        if p.exists():
            df = pl.read_parquet(p).sort("timestamp")
            # Garder timestamp + colonnes de prediction _ctrl_runXX
            pred_cols = [c for c in df.columns
                         if any(v.replace("_ctrl","") in c for v in METEO_VARS)
                         and "_run" in c
                         and "_q10" not in c
                         and "_q90" not in c
                         and "_stde" not in c]
            if not pred_cols:
                print(f"  ⚠ Aucune colonne pred_*_ctrl_runXX dans {name}")
                continue
            print(f"  Meteo  : {p.name}  "
                  f"({df.shape[0]:,} lignes, {len(pred_cols)} cols pred)")
            return df.select(["timestamp"] + pred_cols)
    raise FileNotFoundError("Aucun fichier meteo_sion*.parquet dans data/")


# ════════════════════════════════════════════════════════════════════════════
# SELECTION DU RUN METEO LE PLUS RECENT DISPONIBLE
# ════════════════════════════════════════════════════════════════════════════
def get_best_run(target_hour: int) -> int | None:
    """
    Retourne le run_id optimal pour predire l'heure cible H de J+1 a 11h J.

    Structure du fichier MeteoSuisse :
      - run_id = tag = horizon en heures de la prevision
      - Chaque run couvre uniquement les heures H ou H % 3 == run_id % 3
        (run01,04,07... → H%3==1 | run02,05,08... → H%3==2 | run03,06,09... → H%3==0)
      - Emission du run = heure_cible - run_id
      - A 11h J, le run est disponible si emission <= 11h J
        => (24 + H) - run_id <= 11  =>  run_id >= H + 13

    On prend le plus petit run_id valide :
      - run_id >= H + 13   (emis avant 11h J)
      - run_id % 3 == H % 3  (couvre cette heure)
      - run_id <= 33       (limite du fichier)
    """
    min_run_id = target_hour + (24 - EMISSION_HOUR)   # = H + 13
    for r in range(min_run_id, 34):
        if r % 3 == target_hour % 3:
            return r
    return None


def select_meteo_for_target(df_meteo: pl.DataFrame,
                             target_date: pl.Date,
                             target_hour: int,
                             var: str) -> float | None:
    """
    Pour une heure cible (date + heure) et une variable,
    selectionne la valeur du run le plus recent disponible a 11h.

    Le timestamp dans df_meteo correspond a l'heure CIBLE de la prevision.
    La colonne est : {var}_run{tag:02d}
    """
    horizon_h = (24 - EMISSION_HOUR) + target_hour   # heures depuis 11h J
    best_tag  = get_best_run_tag(horizon_h)
    if best_tag is None:
        return None

    col_name = f"{var}_run{best_tag:02d}"
    if col_name not in df_meteo.columns:
        # Chercher le prochain tag disponible
        min_tag = best_tag
        for tag in AVAILABLE_TAGS:
            if tag >= min_tag:
                col_try = f"{var}_run{tag:02d}"
                if col_try in df_meteo.columns:
                    col_name = col_try
                    break
        else:
            return None

    # Filtrer sur l'heure cible (timestamp = date J+1 a target_hour:00)
    row = df_meteo.filter(
        (pl.col("timestamp").dt.date() == target_date) &
        (pl.col("timestamp").dt.hour() == target_hour)
    )
    if row.is_empty() or col_name not in row.columns:
        return None
    val = row[col_name][0]
    return float(val) if val is not None else None


# ════════════════════════════════════════════════════════════════════════════
# CONSTRUCTION DE LA MATRICE
# ════════════════════════════════════════════════════════════════════════════
def build_features(df_oiken: pl.DataFrame,
                   df_cal:   pl.DataFrame,
                   df_meteo: pl.DataFrame) -> pl.DataFrame:
    """
    Construit la matrice d'entrainement complete.

    Pour chaque jour d'emission (J) :
      Pour chaque quart d'heure cible de J+1 (96 lignes) :
        → Assembler toutes les features
    """
    from datetime import date, timedelta, datetime, timezone

    # Index rapide : timestamp -> load et PV
    # Convertir en dict pour acces O(1)
    print("  Construction des index rapides...")

    oiken_dict = {
        row["timestamp"]: row
        for row in df_oiken.to_dicts()
    }
    cal_dict = {
        row["timestamp"]: row
        for row in df_cal.to_dicts()
    }

    # Pre-calculer un index meteo par (date, heure) pour chaque var/run
    # IMPORTANT : filtrer sur minute==0 car les pred sont horaires
    # mais le fichier meteo a un pas de 10min — les autres lignes ont NaN
    print("  Construction de l'index meteo...")
    df_meteo_hourly = df_meteo.filter(
        pl.col("timestamp").dt.minute() == 0
    )
    print(f"  Lignes meteo horaires : {df_meteo_hourly.shape[0]:,}")
    meteo_dict = {}
    for row in df_meteo_hourly.to_dicts():
        ts  = row["timestamp"]
        key = (ts.date() if hasattr(ts, "date") else ts, ts.hour)
        meteo_dict[key] = row

    # Plage de dates d'emission
    ts_min = df_oiken["timestamp"].min()
    ts_max = df_oiken["timestamp"].max()

    # On commence a J+1 minimum (besoin de J-7 pour les lags)
    start_date = (ts_min.date() if hasattr(ts_min, "date")
                  else ts_min) + timedelta(days=8)
    end_date   = (ts_max.date() if hasattr(ts_max, "date")
                  else ts_max) - timedelta(days=1)

    print(f"  Periode d'emission : {start_date} -> {end_date}")

    rows = []
    n_days = (end_date - start_date).days + 1
    n_missing_load = 0
    n_missing_meteo = 0

    for day_offset in range(n_days):
        emission_date = start_date + timedelta(days=day_offset)
        target_date   = emission_date + timedelta(days=1)

        if day_offset % 100 == 0:
            print(f"  Jour {day_offset}/{n_days} — {emission_date}")
        
        # Debug pour emission_date=2023-06-14 (target=2023-06-15 H=8)
        if str(emission_date) == "2023-06-14":
            meteo_key_test = (target_date, 8)
            row_test = meteo_dict.get(meteo_key_test, {})
            H = 8
            min_run = H + 13  # 21
            best = None
            for r in range(min_run, 34):
                if r % 3 == H % 3:
                    best = r
                    break
            col_test = f"pred_radiation_ctrl_run{best:02d}" if best else "N/A"
            val_test = row_test.get(col_test) if row_test else "ROW_EMPTY"
            # Also check if col exists in row
            col_exists = col_test in row_test if row_test else False
            print(f"  [DEBUG 2023-06-14]")
            print(f"    target_date={target_date}, type={type(target_date)}")
            print(f"    meteo_key={meteo_key_test}")
            print(f"    row found: {bool(row_test)}")
            print(f"    best_run={best}, col={col_test}")
            print(f"    col_exists={col_exists}")
            print(f"    val={val_test}")
            if row_test:
                all_keys = [k for k in row_test.keys() if "pred_radiation_ctrl_run" in k]
                print(f"    all pred_radiation_ctrl_run cols in row: {all_keys[:5]}")





        # Heure d'emission = 11h UTC
        emission_ts = datetime(
            emission_date.year, emission_date.month, emission_date.day,
            EMISSION_HOUR, 0, 0, tzinfo=timezone.utc
        )

        # ── Lags journaliers (calcules une fois par jour) ─────────────────
        # Moyenne load sur les 24h completes avant 11h (J entier avant emission)
        load_mean_J1 = _day_mean_load(oiken_dict, emission_date, 0)
        load_mean_J7 = _day_mean_load(oiken_dict, emission_date, 6)

        # ── 96 quarts d'heure de J+1 ─────────────────────────────────────
        for qh in range(STEPS_PER_DAY):
            target_hour    = qh // STEPS_PER_HOUR
            target_quarter = qh %  STEPS_PER_HOUR
            target_minute  = target_quarter * STEP_MIN

            target_ts = datetime(
                target_date.year, target_date.month, target_date.day,
                target_hour, target_minute, 0, tzinfo=timezone.utc
            )

            # ── Valeur cible (load a predire) ─────────────────────────────
            load_target = oiken_dict.get(target_ts, {}).get("load")

            # ── Bloc 1 : features calendaires de l'heure cible ───────────
            cal = cal_dict.get(target_ts, {})

            # ── Bloc 2 : meteo (run le plus recent disponible a 11h) ──────
            # Tag = horizon de prevision (heures a l'avance)
            # Chercher le plus petit tag >= horizon_min avec des donnees
            best_run = get_best_run(target_hour)
            meteo_key = (target_date, target_hour)
            meteo_row = meteo_dict.get(meteo_key, {})

            # DEBUG : verifier la cle exacte la premiere fois
            if day_offset == 0 and qh == 32:  # 08h00
                print(f"  [DEBUG] target_date={target_date} type={type(target_date)}")
                print(f"  [DEBUG] target_hour={target_hour}")
                print(f"  [DEBUG] meteo_key={meteo_key} type={type(meteo_key[0])}")
                print(f"  [DEBUG] best_run={best_run}")
                print(f"  [DEBUG] key in dict: {meteo_key in meteo_dict}")
                if meteo_key in meteo_dict:
                    row_test = meteo_dict[meteo_key]
                    col_test = f"pred_radiation_ctrl_run{best_run:02d}"
                    print(f"  [DEBUG] {col_test} = {row_test.get(col_test)}")
                else:
                    sample = list(meteo_dict.keys())[:3]
                    print(f"  [DEBUG] Sample dict keys: {sample}")
                    print(f"  [DEBUG] Sample key types: {[(type(k[0]), type(k[1])) for k in sample]}")

            meteo_vals = {}
            actual_tag_used = None
            for var in METEO_VARS:
                val = None
                if best_run is not None:
                    col = f"{var}_run{best_run:02d}"
                    v   = meteo_row.get(col)
                    if v is not None:
                        val = float(v)
                        actual_tag_used = best_run
                    else:
                        # Fallback : essayer les runs suivants du meme groupe (H%3)
                        for r in range(best_run + 3, 34, 3):
                            col2 = f"{var}_run{r:02d}"
                            v2 = meteo_row.get(col2)
                            if v2 is not None:
                                val = float(v2)
                                actual_tag_used = r
                                break
                meteo_vals[var] = val
                if val is None:
                    n_missing_meteo += 1

            # ── Bloc 3 : lags du load ─────────────────────────────────────
            # J-1 meme heure que la cible
            ts_J1 = datetime(
                emission_date.year, emission_date.month, emission_date.day,
                target_hour, target_minute, 0, tzinfo=timezone.utc
            )
            # J-7 meme heure que la cible
            date_J7 = target_date - timedelta(days=7)
            ts_J7 = datetime(
                date_J7.year, date_J7.month, date_J7.day,
                target_hour, target_minute, 0, tzinfo=timezone.utc
            )

            load_J1 = oiken_dict.get(ts_J1, {}).get("load")
            load_J7 = oiken_dict.get(ts_J7, {}).get("load")
            if load_J1 is None or load_J7 is None:
                n_missing_load += 1

            # ── Bloc 4 : lags PV ─────────────────────────────────────────
            pv_J1 = oiken_dict.get(ts_J1, {}).get("pv_total")
            pv_J7 = oiken_dict.get(ts_J7, {}).get("pv_total")

            # ── Assembler la ligne ────────────────────────────────────────
            row = {
                # Identifiants
                "emission_date":   emission_date.isoformat(),
                "target_timestamp": target_ts.isoformat(),

                # Bloc 1 — Calendaire heure cible
                "hour_sin":          cal.get("hour_sin"),
                "hour_cos":          cal.get("hour_cos"),
                "quarter":           target_quarter,
                "dow_sin":           cal.get("day_of_week_sin"),
                "dow_cos":           cal.get("day_of_week_cos"),
                "is_weekend":        cal.get("is_weekend"),
                "is_holiday":        cal.get("is_holiday"),
                "is_school_holiday": cal.get("is_school_holiday"),
                "month_sin":         cal.get("month_sin"),
                "month_cos":         cal.get("month_cos"),
                "doy_sin":           cal.get("day_of_year_sin"),
                "doy_cos":           cal.get("day_of_year_cos"),

                # Bloc 2 — Meteo Sion (run ctrl le plus recent)
                "pred_radiation":    meteo_vals.get("pred_radiation_ctrl"),
                "pred_sunshine":     meteo_vals.get("pred_sunshine_ctrl"),
                "pred_temperature":  meteo_vals.get("pred_temperature_ctrl"),
                "pred_humidity":     meteo_vals.get("pred_humidity_ctrl"),
                "meteo_tag_used":    actual_tag_used,  # tag qui a fourni la valeur

                # Bloc 3 — Lags load
                "load_lag_J1":       load_J1,
                "load_lag_J7":       load_J7,
                "load_mean_J1":      load_mean_J1,
                "load_mean_J7":      load_mean_J7,

                # Bloc 4 — Lags PV
                "pv_lag_J1":         pv_J1,
                "pv_lag_J7":         pv_J7,

                # Cible
                "load":              load_target,
            }
            rows.append(row)

    print(f"\n  Lignes generees      : {len(rows):,}")
    print(f"  Load manquants       : {n_missing_load:,}")
    print(f"  Meteo manquante      : {n_missing_meteo:,}")

    df = pl.DataFrame(rows)
    return df


def _day_mean_load(oiken_dict: dict,
                   ref_date,
                   days_back: int) -> float | None:
    """
    Calcule la moyenne du load sur les 24h completes du jour
    ref_date - days_back.
    """
    from datetime import date, datetime, timezone, timedelta
    target_day = ref_date - timedelta(days=days_back)
    vals = []
    for h in range(24):
        for m in [0, 15, 30, 45]:
            ts = datetime(target_day.year, target_day.month, target_day.day,
                          h, m, 0, tzinfo=timezone.utc)
            v = oiken_dict.get(ts, {}).get("load")
            if v is not None:
                vals.append(v)
    return sum(vals) / len(vals) if vals else None


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*65)
    print("Feature Engineering — Day-Ahead")
    print("="*65)

    print("\nChargement des donnees...")
    df_oiken = load_oiken()
    df_cal   = load_calendar()
    df_meteo = load_meteo_predictions()

    print("\nConstruction de la matrice de features...")
    df_feat = build_features(df_oiken, df_cal, df_meteo)

    # ── Statistiques de la matrice ────────────────────────────────────────
    print("\n" + "="*65)
    print("STATISTIQUES DE LA MATRICE")
    print("="*65)
    print(f"  Shape  : {df_feat.shape[0]:,} lignes x {df_feat.shape[1]} colonnes")
    print(f"  Colonnes : {df_feat.columns}")

    print("\n  Valeurs nulles par colonne :")
    for col in df_feat.columns:
        n = df_feat[col].is_null().sum()
        if n > 0:
            pct = n / df_feat.shape[0] * 100
            print(f"    {col:<30} {n:>8,} ({pct:.1f}%)")

    print("\n  Apercu des 3 premieres lignes :")
    print(df_feat.head(3))

    # ── Export ────────────────────────────────────────────────────────────
    out = DATA_DIR / "features_day_ahead.parquet"
    df_feat.write_parquet(out)
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"\n✅ Exporte -> {out}")
    print(f"   {df_feat.shape[0]:,} lignes x {df_feat.shape[1]} colonnes  ({size_mb:.1f} MB)")

    # ── Verification des tags utilises ────────────────────────────────────
    print("\n  Distribution des tags meteo utilises :")
    tag_dist = (df_feat
                .group_by("meteo_tag_used")
                .agg(pl.len().alias("count"))
                .sort("meteo_tag_used"))
    print(tag_dist)