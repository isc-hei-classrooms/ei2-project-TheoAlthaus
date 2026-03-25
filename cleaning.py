"""
Pipeline de preprocessing des donnees
======================================
Lit  : data/oiken.parquet + data/meteo_*.parquet
Ecrit: data/oiken_clean.parquet + data/meteo_*_clean.parquet

Etapes :
  1. Oiken  — suppression doublons, interpolation NaN courts, flag grands trous
  2. Meteo  — interpolation NaN courts, flag grands trous
  3. Rapport HTML : data/preprocessing_report.html

Dependances : pip install polars plotly pyarrow
Execution   : python preprocessing.py
"""

from pathlib import Path
from datetime import timedelta
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

DATA_DIR = Path("data")

# ── Seuils ────────────────────────────────────────────────────────────────────
# Oiken : pas de 15min → 1h = 4 points, 2h = 8 points
OIKEN_STEP_MIN      = 15
OIKEN_INTERP_MAX    = 8    # interpoler si trou <= 2h (8 quarts d'heure)

# Meteo : pas de 10min → 1h = 6 points, 2h = 12 points
METEO_STEP_MIN      = 10
METEO_INTERP_MAX    = 12   # interpoler si trou <= 2h (12 dizaines de minutes)

# ── Palette rapport ───────────────────────────────────────────────────────────
BG       = "#0F1117"
BG_PAPER = "#161B22"
TEXT     = "#E6EDF3"
SUBTEXT  = "#8B949E"
C_GRID   = "rgba(200,200,200,0.25)"
C_OK     = "#27AE60"
C_WARN   = "#F39C12"
C_ERR    = "#E74C3C"

LAYOUT_BASE = dict(
    paper_bgcolor=BG_PAPER, plot_bgcolor=BG,
    font=dict(color=TEXT, family="Courier New, monospace", size=12),
    margin=dict(l=60, r=30, t=55, b=50),
)

def apply_base(fig, title, height=380):
    fig.update_layout(**LAYOUT_BASE, height=height,
        title=dict(text=title, font=dict(size=13, color=TEXT), x=0.01))
    fig.update_xaxes(gridcolor=C_GRID, showline=True, linecolor="#30363D")
    fig.update_yaxes(gridcolor=C_GRID, showline=True, linecolor="#30363D")
    return fig


# ════════════════════════════════════════════════════════════════════════════
# ETAPE 1 — SUPPRESSION DES DOUBLONS
# ════════════════════════════════════════════════════════════════════════════
def remove_duplicates(df: pl.DataFrame, label: str) -> tuple[pl.DataFrame, dict]:
    """Garde la premiere occurrence pour chaque timestamp duplique."""
    n_before = df.shape[0]
    n_dups   = df.shape[0] - df["timestamp"].n_unique()

    if n_dups > 0:
        # Identifier les timestamps en double
        dup_ts = (df.filter(pl.col("timestamp").is_duplicated())
                    ["timestamp"].unique().sort())
        print(f"  [{label}] {n_dups} doublons sur {dup_ts.shape[0]} timestamps → suppression")
        for ts in dup_ts.to_list()[:5]:
            print(f"    {ts}")
        if dup_ts.shape[0] > 5:
            print(f"    ... et {dup_ts.shape[0]-5} autres")

        df = df.unique(subset=["timestamp"], keep="first").sort("timestamp")

    stats = {
        "n_before":   n_before,
        "n_removed":  n_dups,
        "n_after":    df.shape[0],
        "dup_timestamps": n_dups,
    }
    return df, stats


# ════════════════════════════════════════════════════════════════════════════
# ETAPE 2 — REGULARISATION DU PAS DE TEMPS
# ════════════════════════════════════════════════════════════════════════════
def regularize_timestamps(df: pl.DataFrame, step_min: int,
                           label: str) -> tuple[pl.DataFrame, dict]:
    """
    Cree une grille temporelle reguliere et aligne les donnees dessus.
    Les timestamps manquants deviennent des lignes avec NaN.
    """
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()

    # Grille reguliere
    grid = pl.DataFrame({
        "timestamp": pl.datetime_range(
            ts_min, ts_max,
            interval=f"{step_min}m",
            time_unit="us", time_zone="UTC", eager=True
        )
    })

    n_expected = grid.shape[0]
    n_actual   = df.shape[0]
    n_gaps     = n_expected - n_actual

    print(f"  [{label}] Grille reguliere : {n_expected:,} points attendus, "
          f"{n_actual:,} presents → {n_gaps:,} manquants")

    # Jointure pour aligner sur la grille
    df_aligned = grid.join(df, on="timestamp", how="left")

    stats = {
        "n_expected": n_expected,
        "n_actual":   n_actual,
        "n_gaps":     n_gaps,
    }
    return df_aligned, stats


# ════════════════════════════════════════════════════════════════════════════
# ETAPE 3 — INTERPOLATION DES NaN COURTS + FLAG DES GRANDS TROUS
# ════════════════════════════════════════════════════════════════════════════
def interpolate_and_flag(df: pl.DataFrame, numeric_cols: list[str],
                         max_interp_pts: int,
                         label: str) -> tuple[pl.DataFrame, dict]:
    """
    Pour chaque colonne numerique :
      - Identifie les sequences de NaN
      - Interpole lineairement si longueur <= max_interp_pts
      - Laisse NaN si longueur > max_interp_pts
    Ajoute une colonne binaire `{col}_missing` (1 = grand trou non interpole).
    """
    stats = {}

    for col in numeric_cols:
        if col not in df.columns:
            continue

        n_nan_before = df[col].is_null().sum()
        if n_nan_before == 0:
            stats[col] = {"n_nan": 0, "n_interpolated": 0, "n_flagged": 0}
            continue

        # ── Identifier les runs de NaN ────────────────────────────────────
        is_null = df[col].is_null().cast(pl.Int8)
        # Run-length encoding : groupes consecutifs
        runs = (
            df.with_columns(is_null.alias("_is_null"))
              .with_columns(
                  (pl.col("_is_null") != pl.col("_is_null").shift(1))
                    .fill_null(True).cum_sum().alias("_run_id")
              )
              .group_by("_run_id", maintain_order=True)
              .agg([
                  pl.col("_is_null").first().alias("is_null"),
                  pl.len().alias("run_len"),
                  pl.first("timestamp").alias("run_start"),
              ])
              .filter(pl.col("is_null") == 1)
        )

        n_short = runs.filter(pl.col("run_len") <= max_interp_pts).shape[0]
        n_long  = runs.filter(pl.col("run_len") >  max_interp_pts).shape[0]

        # ── Interpolation lineaire globale ────────────────────────────────
        # Strategie : interpoler tout, puis re-mettre NaN pour les grands trous
        df = df.with_columns(
            pl.col(col).interpolate().alias(f"_{col}_interp")
        )

        # ── Re-appliquer NaN pour les grands trous ────────────────────────
        # Construire un masque : True = position dans un grand trou
        long_runs = runs.filter(pl.col("run_len") > max_interp_pts)
        large_gap_mask = pl.Series("mask", [False] * df.shape[0])

        if long_runs.shape[0] > 0:
            for row in long_runs.iter_rows(named=True):
                t_start = row["run_start"]
                # Trouver les indices correspondants
                idx_mask = (df["timestamp"] >= t_start) & \
                           (df[col].is_null())  # positions nulles dans ce run
                # On utilise le masque original
                large_gap_mask = large_gap_mask | idx_mask

            # Re-mettre NaN aux grands trous
            df = df.with_columns(
                pl.when(large_gap_mask)
                  .then(None)
                  .otherwise(pl.col(f"_{col}_interp"))
                  .alias(f"_{col}_interp")
            )

        # Remplacer la colonne originale
        df = df.with_columns(
            pl.col(f"_{col}_interp").alias(col)
        ).drop(f"_{col}_interp")

        # ── Flag des valeurs encore nulles (grands trous) ─────────────────
        df = df.with_columns(
            pl.col(col).is_null().cast(pl.Int8).alias(f"{col}_missing")
        )

        n_nan_after = df[col].is_null().sum()
        n_interpolated = n_nan_before - n_nan_after

        stats[col] = {
            "n_nan":          n_nan_before,
            "n_interpolated": n_interpolated,
            "n_flagged":      n_nan_after,
            "n_short_runs":   n_short,
            "n_long_runs":    n_long,
        }

        flag = "✓" if n_nan_after == 0 else ("~" if n_nan_after < 100 else "⚠")
        print(f"  [{label}] {col:<30} "
              f"NaN avant={n_nan_before:>6,}  "
              f"interpoles={n_interpolated:>6,}  "
              f"restants={n_nan_after:>6,}  {flag}")

    return df, stats


# ════════════════════════════════════════════════════════════════════════════
# PIPELINE OIKEN
# ════════════════════════════════════════════════════════════════════════════
def process_oiken() -> dict:
    path_in  = DATA_DIR / "oiken.parquet"
    path_out = DATA_DIR / "oiken_clean.parquet"

    print("\n" + "="*65)
    print("  OIKEN")
    print("="*65)

    df = pl.read_parquet(path_in)
    print(f"  Charge : {df.shape[0]:,} lignes | cols : {df.columns}")

    report = {"label": "Oiken", "steps": {}}

    # 1. Doublons
    df, s = remove_duplicates(df, "Oiken")
    report["steps"]["duplicates"] = s

    # 2. Regularisation
    df, s = regularize_timestamps(df, OIKEN_STEP_MIN, "Oiken")
    report["steps"]["regularize"] = s

    # 3. Interpolation
    num_cols = ["load", "forecast_load",
                "pv_central_valais", "pv_sion", "pv_sierre", "pv_remote", "pv_total"]
    num_cols = [c for c in num_cols if c in df.columns]
    df, s = interpolate_and_flag(df, num_cols, OIKEN_INTERP_MAX, "Oiken")
    report["steps"]["interpolation"] = s

    df.write_parquet(path_out)
    size_mb = path_out.stat().st_size / 1024 / 1024
    print(f"\n  ✅ {path_out}  ({df.shape[0]:,} lignes × {df.shape[1]} col, {size_mb:.1f} MB)")
    report["shape_out"] = df.shape
    return report


# ════════════════════════════════════════════════════════════════════════════
# PIPELINE METEO
# ════════════════════════════════════════════════════════════════════════════
def process_meteo() -> list[dict]:
    reports = []
    hist_files = [p for p in sorted(DATA_DIR.glob("meteo_*.parquet"))
                  if "_predictions" not in p.name
                  and "_clean" not in p.name]

    for path_in in hist_files:
        name = (path_in.stem
                .replace("meteo_", "")
                .replace("_historical", "")
                .replace("_", " ").title())

        print(f"\n{'='*65}")
        print(f"  METEO — {name}")
        print(f"{'='*65}")

        df = pl.read_parquet(path_in)
        hist_cols = [c for c in df.columns if c.startswith("hist_")]

        if not hist_cols:
            print(f"  Aucune colonne hist_* — ignore")
            continue

        print(f"  Charge : {df.shape[0]:,} lignes | {len(hist_cols)} variables historiques")

        report = {"label": name, "steps": {}}

        # 1. Doublons
        df, s = remove_duplicates(df, name)
        report["steps"]["duplicates"] = s

        # 2. Regularisation
        df, s = regularize_timestamps(df, METEO_STEP_MIN, name)
        report["steps"]["regularize"] = s

        # 3. Interpolation sur les colonnes hist_ uniquement
        df, s = interpolate_and_flag(df, hist_cols, METEO_INTERP_MAX, name)
        report["steps"]["interpolation"] = s

        # Nom du fichier de sortie
        stem_out = path_in.stem.replace("_historical", "") + "_clean"
        path_out = DATA_DIR / f"{stem_out}.parquet"
        df.write_parquet(path_out)
        size_mb = path_out.stat().st_size / 1024 / 1024
        print(f"\n  ✅ {path_out}  ({df.shape[0]:,} lignes × {df.shape[1]} col, {size_mb:.1f} MB)")
        report["shape_out"] = df.shape
        reports.append(report)

    return reports


# ════════════════════════════════════════════════════════════════════════════
# RAPPORT HTML
# ════════════════════════════════════════════════════════════════════════════
def fig_nan_before_after(reports: list[dict]) -> go.Figure:
    """Barres NaN avant/apres par dataset et par colonne."""
    datasets, cols_list, before_vals, after_vals = [], [], [], []

    for r in reports:
        interp = r["steps"].get("interpolation", {})
        for col, s in interp.items():
            if s["n_nan"] == 0:
                continue
            datasets.append(r["label"])
            cols_list.append(col.replace("hist_", "").replace("_", " "))
            before_vals.append(s["n_nan"])
            after_vals.append(s["n_flagged"])

    if not datasets:
        fig = go.Figure()
        fig.update_layout(**LAYOUT_BASE, height=300,
            title=dict(text="Aucun NaN detecte — donnees propres", x=0.01,
                       font=dict(size=13, color=C_OK)))
        return fig

    labels = [f"{d} / {c}" for d, c in zip(datasets, cols_list)]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=before_vals,
        name="NaN avant", marker_color=C_ERR, opacity=0.8,
    ))
    fig.add_trace(go.Bar(
        x=labels, y=after_vals,
        name="NaN apres (grands trous)", marker_color=C_WARN, opacity=0.9,
    ))
    apply_base(fig, "NaN avant et apres preprocessing", height=400)
    fig.update_layout(barmode="group", xaxis_tickangle=-40,
                      yaxis_title="Nb de points")
    return fig


def build_html_report(oiken_report: dict, meteo_reports: list[dict]) -> str:
    all_reports = [oiken_report] + meteo_reports

    # ── Section texte par dataset ─────────────────────────────────────────
    sections_html = []
    for r in all_reports:
        label = r["label"]
        steps = r["steps"]
        dup   = steps.get("duplicates", {})
        reg   = steps.get("regularize", {})
        interp= steps.get("interpolation", {})

        lines = [
            f"<details open><summary style='cursor:pointer;color:#FFC000;"
            f"font-family:Courier New;font-size:14px;padding:6px 0'>"
            f"▶ {label}</summary>",
            "<pre style='background:#0F1117;color:#E6EDF3;padding:14px;"
            "border-radius:6px;font-size:12px;border:1px solid #30363D;"
            "margin:8px 0 16px 0'>",
        ]

        # Doublons
        n_dup = dup.get("n_removed", 0)
        lines.append(f"  Doublons supprimes      : {n_dup:>6,}  "
                     f"{'✓' if n_dup == 0 else '⚠ corriges'}")

        # Regularisation
        n_gaps = reg.get("n_gaps", 0)
        lines.append(f"  Gaps temporels detectes : {n_gaps:>6,}  "
                     f"{'✓' if n_gaps == 0 else '~ traites par interpolation'}")

        # Interpolation
        if interp:
            lines.append(f"\n  {'Variable':<32} {'NaN avant':>10} "
                         f"{'Interpoles':>12} {'Restants':>10}  Verdict")
            lines.append("  " + "─"*70)
            for col, s in interp.items():
                if s["n_nan"] == 0:
                    verdict = "✓ propre"
                    vc = ""
                elif s["n_flagged"] == 0:
                    verdict = "✓ tout interpole"
                    vc = ""
                elif s["n_flagged"] < 50:
                    verdict = "~ grands trous flagges"
                    vc = ""
                else:
                    verdict = "⚠ verifier"
                    vc = ""
                col_short = col.replace("hist_", "").replace("_", " ")
                lines.append(
                    f"  {col_short:<32} {s['n_nan']:>10,} "
                    f"{s['n_interpolated']:>12,} {s['n_flagged']:>10,}  {verdict}"
                )

        shape = r.get("shape_out", ("?", "?"))
        lines.append(f"\n  Shape finale : {shape[0]:,} lignes x {shape[1]} colonnes")
        lines.append("</pre></details>")
        sections_html.append("\n".join(lines))

    # ── Graphique NaN avant/apres ─────────────────────────────────────────
    fig = fig_nan_before_after(all_reports)
    fig_html = pio.to_html(fig, include_plotlyjs=True,
                           full_html=False, config={"responsive": True})

    return f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="UTF-8">
<title>Preprocessing Report</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:{BG}; color:{TEXT}; font-family:'Courier New',monospace; padding:24px; }}
  h1 {{ font-size:20px; letter-spacing:2px; margin-bottom:4px; }}
  .subtitle {{ color:{SUBTEXT}; font-size:13px; margin-bottom:28px; }}
  .section {{ margin-bottom:24px; border:1px solid #21262D;
              border-radius:8px; overflow:hidden; }}
  .sec-title {{ background:#161B22; padding:10px 16px; font-size:13px;
                color:{SUBTEXT}; letter-spacing:1px;
                border-bottom:1px solid #21262D; }}
  .content {{ padding:16px; }}
</style></head><body>
<h1>Rapport de Preprocessing</h1>
<div class="subtitle">
  Doublons, regularisation temporelle, interpolation NaN courts, flag grands trous
</div>

<div class="section">
  <div class="sec-title">Visualisation NaN avant / apres</div>
  {fig_html}
</div>

<div class="section">
  <div class="sec-title">Detail par dataset</div>
  <div class="content">
    {''.join(sections_html)}
  </div>
</div>

<div class="section">
  <div class="sec-title">Fichiers generes</div>
  <div class="content">
    <pre style='background:#0F1117;color:#E6EDF3;padding:14px;
    border-radius:6px;font-size:12px;border:1px solid #30363D'>
  data/oiken_clean.parquet         ← Oiken nettoye
  data/meteo_*_clean.parquet       ← Meteo nettoyes (un par station)

  Colonnes ajoutees : {{col}}_missing = 1 si grand trou non interpole
  Colonnes PV       : offset pv_remote soustrait, pv_total recalcule
    </pre>
  </div>
</div>
</body></html>"""


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Pipeline de preprocessing")
    print("="*65)

    oiken_report  = process_oiken()
    meteo_reports = process_meteo()

    print("\nGeneration du rapport HTML...")
    html = build_html_report(oiken_report, meteo_reports)
    out  = DATA_DIR / "preprocessing_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"✅ Rapport -> {out}  ({out.stat().st_size/1024:.0f} KB)")

    # Resume final
    print("\n" + "="*65)
    print("FICHIERS GENERES")
    print("="*65)
    for p in sorted(DATA_DIR.glob("*_clean.parquet")):
        df = pl.read_parquet(p)
        size_mb = p.stat().st_size / 1024 / 1024
        missing_cols = [c for c in df.columns if c.endswith("_missing")]
        total_flags  = sum(df[c].sum() for c in missing_cols)
        print(f"  {p.name:<45} {df.shape[0]:>9,} lignes  "
              f"{df.shape[1]:>4} col  {size_mb:>5.1f} MB  "
              f"flags={total_flags:,}")
