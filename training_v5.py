"""
Entrainement modeles — version 5
==================================
Lit  : data/features_v5.parquet
       data/oiken_clean.parquet
Ecrit: data/models/results_v5.parquet
       data/models/predictions_v5.parquet
       data/models/wf_results_v5.parquet
       data/models/wf_predictions_v5.parquet

Modeles : LightGBM, XGBoost (single-output)
Validation : simple + walk-forward (pas=30j)
Comparaison avec Oiken forecast
Correction : exclusion periode Oiken bugguee
             2025-09-13 02:15 UTC -> 2025-09-17 02:00 UTC
"""

import pickle
import datetime
import polars as pl
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import xgboost as xgb

from config import DATA_DIR, FILE_OIKEN_CLEAN

MODELS_DIR = DATA_DIR / "models"
FILE_FEAT  = DATA_DIR / "features_v5.parquet"

WF_STEP = 30 * 96  # 30 jours en quarts d'heure

LGB_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=63,
    min_child_samples=10,
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)
XGB_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    verbosity=0,
)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    label: str,
                    ts_mask: np.ndarray = None) -> dict:
    """
    ts_mask : masque boolen optionnel pour exclure certains timestamps
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if ts_mask is not None:
        mask = mask & ts_mask

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"  {label:<30} MAE={mae:.5f}  RMSE={rmse:.5f}"
          f"  (n={len(y_true):,})")
    return {"model": label, "mae": mae, "rmse": rmse}


def build_oiken_mask(timestamps: list) -> np.ndarray:
    """Masque True = garder, False = exclure (periode buguee)."""
    bug_start = datetime.datetime(2025, 9, 13,  2, 15, 0,
                                  tzinfo=datetime.timezone.utc)
    bug_end   = datetime.datetime(2025, 9, 17,  2,  0, 0,
                                  tzinfo=datetime.timezone.utc)
    mask = []
    for ts in timestamps:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=datetime.timezone.utc)
        mask.append(not (bug_start <= ts <= bug_end))
    return np.array(mask)


def load_data() -> tuple:
    df = pl.read_parquet(FILE_FEAT).sort("timestamp")

    feat_cols = [c for c in df.columns
                 if c not in ("timestamp", "date_locale",
                               "target", "split")]

    train = df.filter(pl.col("split") == "train")
    val   = df.filter(pl.col("split") == "val")
    test  = df.filter(pl.col("split") == "test")

    X_train = train.select(feat_cols).to_numpy()
    y_train = train["target"].to_numpy()
    X_val   = val.select(feat_cols).to_numpy()
    y_val   = val["target"].to_numpy()
    X_test  = test.select(feat_cols).to_numpy()
    y_test  = test["target"].to_numpy()

    ts_val  = val["timestamp"].to_list()
    ts_test = test["timestamp"].to_list()

    print(f"Train : {X_train.shape[0]:,} pts ({X_train.shape[0]//96} jours)")
    print(f"Val   : {X_val.shape[0]:,} pts ({X_val.shape[0]//96} jours)")
    print(f"Test  : {X_test.shape[0]:,} pts ({X_test.shape[0]//96} jours)")
    print(f"Features : {X_train.shape[1]}")

    # Oiken forecast
    oiken  = pl.read_parquet(FILE_OIKEN_CLEAN).sort("timestamp")
    fc_map = {ts: fc for ts, fc in zip(
        oiken["timestamp"].to_list(),
        oiken["forecast_load"].to_list()
    )}

    y_fc_val  = np.array([fc_map.get(ts, np.nan) for ts in ts_val])
    y_fc_test = np.array([fc_map.get(ts, np.nan) for ts in ts_test])

    # Masques exclusion periode buguee
    mask_val  = build_oiken_mask(ts_val)
    mask_test = build_oiken_mask(ts_test)

    n_excl = (~mask_test).sum()
    print(f"\nBaseline Oiken forecast :")
    print(f"  Points exclus (periode buguee) : {n_excl:,}")

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            ts_val, ts_test, feat_cols,
            y_fc_val, y_fc_test,
            mask_val, mask_test, df)


def train_simple(name: str, model,
                 X_train, y_train,
                 X_val, y_val, y_fc_val, mask_val,
                 X_test, y_test, y_fc_test, mask_test) -> tuple:
    print(f"\n{'='*60}")
    print(f"  SIMPLE — {name.upper()}")
    print(f"{'='*60}")
    print("  Entrainement...")
    model.fit(X_train, y_train)

    print("  Evaluation :")
    mv = compute_metrics(y_val,  model.predict(X_val),  "val")
    mt = compute_metrics(y_test, model.predict(X_test), "test")
    mv["split"] = "val"
    mt["split"] = "test"

    return model, mv, mt


def run_walkforward(name: str, model_class, model_params: dict,
                    X_train_init, y_train_init,
                    X_wf_all, y_wf_all,
                    ts_wf, splits_wf,
                    fc_map: dict,
                    mask_val: np.ndarray,
                    mask_test: np.ndarray) -> tuple:

    print(f"\n{'='*60}")
    print(f"  WALK-FORWARD {name.upper()} (pas={WF_STEP//96}j)")
    print(f"{'='*60}")

    n_wf    = X_wf_all.shape[0]
    n_iters = (n_wf + WF_STEP - 1) // WF_STEP

    X_train = X_train_init.copy()
    y_train = y_train_init.copy()

    ts_all   = []
    sp_all   = []
    true_all = []
    pred_all = []
    fc_all   = []

    for i in range(0, n_wf, WF_STEP):
        iter_num = i // WF_STEP + 1
        i_end    = min(i + WF_STEP, n_wf)

        model = model_class(**model_params)
        model.fit(X_train, y_train)

        X_pred = X_wf_all[i:i_end]
        y_true = y_wf_all[i:i_end]
        y_pred = model.predict(X_pred)
        ts_seg = ts_wf[i:i_end]
        sp_seg = splits_wf[i:i_end]
        fc_seg = np.array([fc_map.get(ts, np.nan) for ts in ts_seg])

        ts_all.extend(ts_seg)
        sp_all.extend(sp_seg)
        true_all.extend(y_true.tolist())
        pred_all.extend(y_pred.tolist())
        fc_all.extend(fc_seg.tolist())

        X_train = np.vstack([X_train, X_pred])
        y_train = np.concatenate([y_train, y_true])

        if iter_num % 5 == 0 or iter_num == n_iters:
            print(f"  Iter {iter_num:>3}/{n_iters}  "
                  f"(train={X_train.shape[0]:,} pts)")

    df_pred = pl.DataFrame({
        "timestamp": ts_all,
        "split":     sp_all,
        "model":     [name] * len(ts_all),
        "true":      true_all,
        "pred":      pred_all,
        "forecast":  fc_all,
    }).with_columns([
        pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
        pl.col("forecast").cast(pl.Float64),
    ])

    # Metriques par split avec exclusion periode buguee
    metrics = []
    ts_array = np.array(ts_all)
    for split, mask in [("val", mask_val), ("test", mask_test)]:
        idx    = [i for i, s in enumerate(sp_all) if s == split]
        y_true = np.array(true_all)[idx]
        y_pred = np.array(pred_all)[idx]
        y_fc   = np.array(fc_all)[idx]
        m_mask = mask if split == "test" else np.ones(len(y_true), dtype=bool)

        print(f"\n  {split} :")
        m  = compute_metrics(y_true, y_pred, name,  m_mask)
        mf = compute_metrics(y_true, y_fc,   "oiken_forecast", m_mask)
        m["split"]  = split
        mf["split"] = split
        metrics.append(m)
        if split == "val":
            metrics.append(mf)

    return df_pred, metrics


if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Chargement features v5...")
    (X_train, y_train, X_val, y_val, X_test, y_test,
     ts_val, ts_test, feat_cols,
     y_fc_val, y_fc_test,
     mask_val, mask_test, df_full) = load_data()

    oiken  = pl.read_parquet(FILE_OIKEN_CLEAN).sort("timestamp")
    fc_map = {ts: fc for ts, fc in zip(
        oiken["timestamp"].to_list(),
        oiken["forecast_load"].to_list()
    )}

    # ── Baseline Oiken ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  BASELINE OIKEN FORECAST")
    print(f"{'='*60}")
    print("  Evaluation (periode buguee exclue pour test) :")
    fc_val_m  = compute_metrics(y_val,  y_fc_val,  "val")
    fc_test_m = compute_metrics(y_test, y_fc_test, "test", mask_test)
    fc_val_m["split"]  = "val"
    fc_test_m["split"] = "test"

    simple_metrics = [
        {**fc_val_m,  "model": "oiken_forecast"},
        {**fc_test_m, "model": "oiken_forecast"},
    ]
    simple_preds = []

    # ── Validation simple ─────────────────────────────────────────────────
    for name, model in [
        ("lightgbm", lgb.LGBMRegressor(**LGB_PARAMS)),
        ("xgboost",  xgb.XGBRegressor(**XGB_PARAMS)),
    ]:
        model, mv, mt = train_simple(
            name, model,
            X_train, y_train,
            X_val, y_val, y_fc_val, mask_val,
            X_test, y_test, y_fc_test, mask_test,
        )
        simple_metrics.extend([
            {**mv, "model": name},
            {**mt, "model": name},
        ])
        with open(MODELS_DIR / f"{name}_v5.pkl", "wb") as f:
            pickle.dump(model, f)

        for split, X, y_true, ts, y_fc in [
            ("val",  X_val,  y_val,  ts_val,  y_fc_val),
            ("test", X_test, y_test, ts_test, y_fc_test),
        ]:
            y_pred = model.predict(X)
            simple_preds.append(pl.DataFrame({
                "timestamp": ts,
                "split":     [split] * len(ts),
                "model":     [name]  * len(ts),
                "true":      y_true.tolist(),
                "pred":      y_pred.tolist(),
                "forecast":  y_fc.tolist(),
            }).with_columns(
                pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
            ))

    pl.concat(simple_preds).write_parquet(MODELS_DIR / "predictions_v5.parquet")
    pl.DataFrame(simple_metrics).write_parquet(MODELS_DIR / "results_v5.parquet")

    # ── Walk-forward ──────────────────────────────────────────────────────
    df_wf     = df_full.filter(pl.col("split").is_in(["val", "test"]))
    X_wf_all  = df_wf.select(feat_cols).to_numpy()
    y_wf_all  = df_wf["target"].to_numpy()
    ts_wf     = df_wf["timestamp"].to_list()
    splits_wf = df_wf["split"].to_list()

    wf_all_preds   = []
    wf_all_metrics = []

    for name, model_class, params in [
        ("lightgbm", lgb.LGBMRegressor, LGB_PARAMS),
        ("xgboost",  xgb.XGBRegressor,  XGB_PARAMS),
    ]:
        df_wf_pred, metrics = run_walkforward(
            name, model_class, params,
            X_train, y_train,
            X_wf_all, y_wf_all,
            ts_wf, splits_wf, fc_map,
            mask_val, mask_test,
        )
        wf_all_preds.append(df_wf_pred)
        wf_all_metrics.extend(metrics)

    pl.concat(wf_all_preds).write_parquet(
        MODELS_DIR / "wf_predictions_v5.parquet"
    )
    pl.DataFrame(wf_all_metrics).write_parquet(
        MODELS_DIR / "wf_results_v5.parquet"
    )

    # ── Resume final ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESUME FINAL — v5")
    print(f"{'='*60}")
    print("  (periode Oiken buguee exclue du calcul test)")

    print("\n  Simple :")
    print(pl.DataFrame(simple_metrics)
            .select(["model", "split", "mae", "rmse"])
            .sort(["split", "mae"]))

    print("\n  Walk-forward :")
    print(pl.DataFrame(wf_all_metrics)
            .select(["model", "split", "mae", "rmse"])
            .sort(["split", "mae"]))

    # ── Comparaison v4 vs v5 ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("COMPARAISON v4 vs v5 (walk-forward)")
    print(f"{'='*60}")

    for vname, vpath in [
        ("v4", MODELS_DIR / "wf_results_v4.parquet"),
    ]:
        if vpath.exists():
            res = pl.read_parquet(vpath)
            for model_name in ["lightgbm", "xgboost"]:
                for split in ["val", "test"]:
                    row = res.filter(
                        (pl.col("model") == model_name) &
                        (pl.col("split") == split)
                    )
                    if row.shape[0] > 0:
                        print(f"  {vname} {model_name:<15} {split} "
                              f"MAE={row['mae'][0]:.5f}")

    print()
    for m in wf_all_metrics:
        print(f"  v5 {m['model']:<15} {m['split']} "
              f"MAE={m['mae']:.5f}")

    print(f"\nFichiers sauvegardes dans {MODELS_DIR} :")
    for p in sorted(MODELS_DIR.glob("*v5*")):
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  {p.name:<45} {size_mb:.1f} MB")