"""
Sliding window walk-forward avec Optuna — LightGBM — golden dataset
=====================================================================
Lit  : data/features_v6.parquet
       golden_data/features_v6_golden.parquet
       data/oiken_clean.parquet
       golden_data/oiken_golden_clean.parquet
Ecrit: golden_data/sw_results_lgb_golden.parquet
       golden_data/sw_predictions_lgb_golden.parquet
       golden_data/sw_hyperparams_lgb_golden.parquet

Structure :
  - Fenetre train : 22 mois glissants
  - Fenetre val   : 4 mois (optimisation Optuna, 30 essais)
  - Fenetre test  : 1 mois
  - Pas           : 1 mois
  - Dataset combine : features_v6 + features_v6_golden
  - Apres optuna  : reentraine sur train+val, evalue sur test
"""

import datetime
import warnings
import polars as pl
import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

from config import DATA_DIR, FILE_OIKEN_CLEAN

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

GOLDEN_DIR = DATA_DIR.parent / "golden_data"
MODELS_DIR = DATA_DIR / "models"

FILE_FEAT_TRAIN  = DATA_DIR   / "features_v6.parquet"
FILE_FEAT_GOLDEN = GOLDEN_DIR / "features_v6_golden.parquet"

FILE_RES   = GOLDEN_DIR / "sw_results_lgb_golden.parquet"
FILE_PREDS = GOLDEN_DIR / "sw_predictions_lgb_golden.parquet"
FILE_PARAMS= GOLDEN_DIR / "sw_hyperparams_lgb_golden.parquet"

TRAIN_MONTHS = 22
VAL_MONTHS   = 4
TEST_MONTHS  = 1
N_TRIALS     = 30

BUG_START = datetime.datetime(2025, 9, 13,  2, 15, 0,
                               tzinfo=datetime.timezone.utc)
BUG_END   = datetime.datetime(2025, 9, 17,  2,  0, 0,
                               tzinfo=datetime.timezone.utc)


def add_months(d: datetime.date, n: int) -> datetime.date:
    month = d.month + n
    year  = d.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    return datetime.date(year, month, 1)


def build_oiken_mask(timestamps: list) -> np.ndarray:
    mask = []
    for ts in timestamps:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=datetime.timezone.utc)
        mask.append(not (BUG_START <= ts <= BUG_END))
    return np.array(mask)


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray,
                mask: np.ndarray = None) -> float:
    m = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask is not None:
        m = m & mask
    if m.sum() == 0:
        return np.nan
    return mean_absolute_error(y_true[m], y_pred[m])


def get_window(df: pl.DataFrame,
               start: datetime.date,
               end: datetime.date) -> pl.DataFrame:
    return df.filter(
        (pl.col("date_locale") >= start) &
        (pl.col("date_locale") < end)
    )


def optuna_objective(trial, X_train, y_train,
                     X_val, y_val) -> float:
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth":         trial.suggest_int("max_depth", 4, 10),
        "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "n_jobs":            -1,
        "random_state":      42,
        "verbose":           -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return compute_mae(y_val, model.predict(X_val))


if __name__ == "__main__":
    GOLDEN_DIR.mkdir(exist_ok=True)

    # ── Chargement et combinaison des features ────────────────────────────
    print("Chargement des features...")
    df_train  = pl.read_parquet(FILE_FEAT_TRAIN)
    df_golden = pl.read_parquet(FILE_FEAT_GOLDEN)

    # Aligner les colonnes — garder uniquement les colonnes communes
    common_cols = [c for c in df_train.columns if c in df_golden.columns]
    print(f"  Features train  : {df_train.shape[1]} col")
    print(f"  Features golden : {df_golden.shape[1]} col")
    print(f"  Colonnes communes : {len(common_cols)}")

    # Combiner
    df_full = pl.concat([
        df_train.select(common_cols),
        df_golden.select(common_cols),
    ]).sort("date_locale").unique(subset=["timestamp"], keep="first")

    feat_cols = [c for c in common_cols
                 if c not in ("timestamp", "date_locale", "target", "split")]

    print(f"  Dataset complet : {df_full.shape[0]:,} lignes")
    print(f"  Periode : {df_full['date_locale'].min()} -> "
          f"{df_full['date_locale'].max()}")
    print(f"  Features : {len(feat_cols)}")

    # ── Chargement forecast Oiken ─────────────────────────────────────────
    print("\nChargement forecast Oiken...")
    oiken_train  = pl.read_parquet(FILE_OIKEN_CLEAN)
    oiken_golden = pl.read_parquet(GOLDEN_DIR / "oiken_golden_clean.parquet")

    oiken_full = pl.concat([
        oiken_train.select(["timestamp", "forecast_load"]),
        oiken_golden.select(["timestamp", "forecast_load"]),
    ]).unique(subset=["timestamp"], keep="first")

    fc_map = {ts: fc for ts, fc in zip(
        oiken_full["timestamp"].to_list(),
        oiken_full["forecast_load"].to_list()
    )}

    # ── Definition des fenetres ───────────────────────────────────────────
    data_start = df_full["date_locale"].min()
    data_end   = df_full["date_locale"].max()

    golden_start = datetime.date(2025, 10, 1)
    first_test   = golden_start
    last_test    = data_end

    test_months = []
    t = first_test
    while t <= last_test:
        test_months.append(t)
        t = add_months(t, TEST_MONTHS)

    print(f"\n  Iterations prevues : {len(test_months)}")
    print(f"  Premier test : {test_months[0]}")
    print(f"  Dernier test : {test_months[-1]}")

    # Identifier les mois golden
    golden_start = datetime.date(2025, 10, 1)
    golden_months = [t for t in test_months if t >= golden_start]
    print(f"  Mois test golden : {len(golden_months)} "
          f"({golden_months[0] if golden_months else 'aucun'} → "
          f"{golden_months[-1] if golden_months else 'aucun'})")

    # ── Boucle principale ─────────────────────────────────────────────────
    all_results = []
    all_preds   = []
    all_params  = []

    for iter_num, test_start in enumerate(test_months, 1):

        test_end    = add_months(test_start,  TEST_MONTHS)
        val_start   = add_months(test_start, -VAL_MONTHS)
        val_end     = test_start
        train_end   = val_start
        train_start = add_months(train_end, -TRAIN_MONTHS)

        is_golden = test_start >= golden_start

        print(f"\n{'='*65}")
        print(f"  Iter {iter_num:>2}/{len(test_months)} — "
              f"Test : {test_start} -> {test_end}"
              f"{'  [GOLDEN]' if is_golden else ''}")
        print(f"  Train : {train_start} -> {train_end}  "
              f"Val : {val_start} -> {val_end}")
        print(f"{'='*65}")

        df_train_w = get_window(df_full, train_start, train_end)
        df_val_w   = get_window(df_full, val_start,   val_end)
        df_test_w  = get_window(df_full, test_start,  test_end)

        if df_train_w.shape[0] == 0 or df_val_w.shape[0] == 0 or \
           df_test_w.shape[0] == 0:
            print("  Fenetre vide, iteration ignoree")
            continue

        X_train = df_train_w.select(feat_cols).to_numpy()
        y_train = df_train_w["target"].to_numpy()
        X_val   = df_val_w.select(feat_cols).to_numpy()
        y_val   = df_val_w["target"].to_numpy()
        X_test  = df_test_w.select(feat_cols).to_numpy()
        y_test  = df_test_w["target"].to_numpy()
        ts_test = df_test_w["timestamp"].to_list()

        print(f"  Train : {X_train.shape[0]:,} pts "
              f"({X_train.shape[0]//96} jours)")
        print(f"  Val   : {X_val.shape[0]:,} pts "
              f"({X_val.shape[0]//96} jours)")
        print(f"  Test  : {X_test.shape[0]:,} pts "
              f"({X_test.shape[0]//96} jours)")

        mask_test = build_oiken_mask(ts_test)
        n_excl    = (~mask_test).sum()
        if n_excl > 0:
            print(f"  Points Oiken exclus : {n_excl:,}")

        # ── Optuna ────────────────────────────────────────────────────────
        print(f"\n  Optimisation Optuna ({N_TRIALS} essais)...")
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(
            lambda trial: optuna_objective(
                trial, X_train, y_train, X_val, y_val
            ),
            n_trials=N_TRIALS,
            show_progress_bar=False,
        )

        best_params  = study.best_params
        best_mae_val = study.best_value
        print(f"  MAE val (best) = {best_mae_val:.5f}")
        print(f"  n_estimators   = {best_params['n_estimators']}")
        print(f"  learning_rate  = {best_params['learning_rate']:.4f}")
        print(f"  max_depth      = {best_params['max_depth']}")
        print(f"  num_leaves     = {best_params['num_leaves']}")

        all_params.append({
            "iter":       iter_num,
            "test_month": str(test_start),
            "is_golden":  is_golden,
            "mae_val":    best_mae_val,
            **{k: float(v) if isinstance(v, float) else int(v)
               for k, v in best_params.items()},
        })

        # ── Reentraine sur train+val ───────────────────────────────────────
        print(f"\n  Reentraine sur train+val...")
        X_trainval = np.vstack([X_train, X_val])
        y_trainval = np.concatenate([y_train, y_val])

        final_model = lgb.LGBMRegressor(
            **best_params, n_jobs=-1, random_state=42, verbose=-1
        )
        final_model.fit(X_trainval, y_trainval)

        # ── Evaluation sur test ───────────────────────────────────────────
        y_pred = final_model.predict(X_test)
        y_fc   = np.array([fc_map.get(ts, np.nan) for ts in ts_test])

        mae_model = compute_mae(y_test, y_pred, mask_test)
        mae_oiken = compute_mae(y_test, y_fc,   mask_test)

        print(f"\n  Evaluation test ({test_start}) :")
        print(f"  LightGBM       MAE={mae_model:.5f}")
        print(f"  Oiken forecast MAE={mae_oiken:.5f}")
        diff = (mae_model - mae_oiken) / mae_oiken * 100
        sign = "▼" if diff < 0 else "▲"
        print(f"  vs Oiken       {sign} {abs(diff):.1f}%")

        all_results.append({
            "iter":        iter_num,
            "test_month":  str(test_start),
            "is_golden":   is_golden,
            "mae_model":   mae_model,
            "mae_oiken":   mae_oiken,
            "diff_pct":    diff,
            "n_train":     X_train.shape[0],
            "n_val":       X_val.shape[0],
            "n_test":      X_test.shape[0],
            "n_excl":      int(n_excl),
        })

        for j, ts in enumerate(ts_test):
            all_preds.append({
                "timestamp":   ts,
                "test_month":  str(test_start),
                "iter":        iter_num,
                "is_golden":   is_golden,
                "true":        float(y_test[j]),
                "pred":        float(y_pred[j]),
                "forecast":    float(y_fc[j]) if not np.isnan(y_fc[j]) else None,
                "oiken_valid": bool(mask_test[j]),
            })

    # ── Sauvegarde ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("RESUME FINAL")
    print(f"{'='*65}")

    df_results = pl.DataFrame(all_results)
    df_preds   = pl.DataFrame(all_preds).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
    )
    df_params  = pl.DataFrame(all_params)

    df_results.write_parquet(FILE_RES)
    df_preds.write_parquet(FILE_PREDS)
    df_params.write_parquet(FILE_PARAMS)

    # Resume par mois
    print(f"\n  {'Mois test':<12} {'MAE model':>10} {'MAE Oiken':>10} "
          f"{'Diff':>8} {'Golden':>8}")
    print(f"  {'-'*52}")
    for row in df_results.iter_rows(named=True):
        sign   = "▼" if row["diff_pct"] < 0 else "▲"
        golden = "★" if row["is_golden"] else ""
        print(f"  {row['test_month']:<12} "
              f"{row['mae_model']:>10.5f} "
              f"{row['mae_oiken']:>10.5f} "
              f"{sign}{abs(row['diff_pct']):>6.1f}% "
              f"{golden:>8}")

    # Stats globales
    n_total  = df_results.shape[0]
    n_golden = df_results.filter(pl.col("is_golden")).shape[0]
    n_better = (df_results["diff_pct"] < 0).sum()
    n_better_golden = (
        df_results.filter(pl.col("is_golden"))["diff_pct"] < 0
    ).sum()

    print(f"\n  Total iterations        : {n_total}")
    print(f"  Dont golden             : {n_golden}")
    print(f"  Mois meilleurs qu'Oiken : {n_better}/{n_total}")
    print(f"  Golden meilleurs Oiken  : {n_better_golden}/{n_golden}")

    mae_mean_all    = df_results["mae_model"].mean()
    mae_mean_golden = df_results.filter(
        pl.col("is_golden"))["mae_model"].mean()
    oik_mean_all    = df_results["mae_oiken"].mean()
    oik_mean_golden = df_results.filter(
        pl.col("is_golden"))["mae_oiken"].mean()

    print(f"\n  MAE moyenne (tous)      : "
          f"model={mae_mean_all:.5f}  oiken={oik_mean_all:.5f}")
    if n_golden > 0:
        print(f"  MAE moyenne (golden)    : "
              f"model={mae_mean_golden:.5f}  oiken={oik_mean_golden:.5f}")

    print(f"\nFichiers sauvegardes :")
    for p in [FILE_RES, FILE_PREDS, FILE_PARAMS]:
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  {p.name:<45} {size_mb:.1f} MB")