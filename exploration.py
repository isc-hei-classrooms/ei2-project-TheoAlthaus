"""
Exploration des donnees brutes
==============================
Lit  : Data_Oiken.csv  +  InfluxDB MeteoSuisse
Ecrit: data/exploration_report.html

Aucune modification des donnees — lecture seule.
"""

from pathlib import Path
from datetime import timezone
import math
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import certifi
from influxdb_client import InfluxDBClient

from config import (
    URL, TOKEN, ORG, BUCKET,
    START, STOP,
    STATIONS, HISTORICAL, PRED_VARS, SUBTYPES, RUNS,
    DATA_DIR,
)

# ── Palette ───────────────────────────────────────────────────────────────────
BG       = "#0F1117"
BG_PAPER = "#161B22"
TEXT     = "#E6EDF3"
SUBTEXT  = "#8B949E"
C_GRID   = "rgba(200,200,200,0.25)"
COLORS   = ["#2E75B6","#ED7D31","#FFC000","#27AE60","#E74C3C","#9B59B6","#1ABC9C"]

LAYOUT_BASE = dict(
    paper_bgcolor=BG_PAPER, plot_bgcolor=BG,
    font=dict(color=TEXT, family="Courier New, monospace", size=12),
    legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="#30363D", borderwidth=1),
    margin=dict(l=60, r=30, t=55, b=50),
)

def apply_base(fig, title, height=400):
    fig.update_layout(**LAYOUT_BASE, height=height,
        title=dict(text=title, font=dict(size=14, color=TEXT), x=0.01))
    fig.update_xaxes(gridcolor=C_GRID, showline=True, linecolor="#30363D")
    fig.update_yaxes(gridcolor=C_GRID, showline=True, linecolor="#30363D")
    return fig


# ════════════════════════════════════════════════════════════════════════════
# EXPLORATION OIKEN
# ════════════════════════════════════════════════════════════════════════════
def explore_oiken(csv_path: str = "Data_Oiken.csv") -> tuple[pl.DataFrame, dict]:
    """
    Charge le CSV Oiken sans aucune modification.
    Retourne le DataFrame brut et un dictionnaire de statistiques.
    """
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

    # Parser le timestamp sans modifier les donnees
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

    # ── Statistiques descriptives ─────────────────────────────────────────
    numeric_cols = ["load", "forecast_load",
                    "pv_central_valais", "pv_sion", "pv_sierre", "pv_remote"]

    stats = {
        "n_rows":      df.shape[0],
        "ts_min":      df["timestamp"].min(),
        "ts_max":      df["timestamp"].max(),
        "step_min":    None,
        "n_duplicates": df.shape[0] - df["timestamp"].n_unique(),
        "columns":     {},
        "gaps":        [],
        "pv_structure": {},
    }

    # Pas de temps dominant
    diffs = df["timestamp"].diff().drop_nulls().dt.total_seconds()
    stats["step_min"] = diffs.mode().to_list()[0] / 60

    # Stats par colonne
    for col in numeric_cols:
        s = df[col]
        n_null = s.is_null().sum()
        vals   = s.drop_nulls()
        mu     = vals.mean()
        sigma  = vals.std()
        stats["columns"][col] = {
            "n_null":     n_null,
            "pct_null":   n_null / df.shape[0] * 100,
            "min":        vals.min(),
            "max":        vals.max(),
            "mean":       mu,
            "std":        sigma,
            "n_negative": (vals < 0).sum(),
            "n_outliers": int(((vals - mu).abs() > 4 * sigma).sum()),
        }

    # Trous temporels (gaps > pas nominal)
    step_s  = stats["step_min"] * 60
    ts_list = df["timestamp"].to_list()
    diffs_s = df["timestamp"].diff().drop_nulls().dt.total_seconds().to_list()
    for i, d in enumerate(diffs_s):
        if d > step_s:
            stats["gaps"].append({
                "t_start": ts_list[i],
                "t_end":   ts_list[i + 1],
                "duration_min": d / 60,
            })

    # Structure PV
    pv_cols = ["pv_central_valais", "pv_sion", "pv_sierre"]
    df_day  = df.filter(
        (pl.col("timestamp").dt.hour() >= 10) &
        (pl.col("timestamp").dt.hour() <= 14)
    )
    centrales_sum = sum(
        df_day[c].drop_nulls().sum() for c in pv_cols
    )
    remote_sum = df_day["pv_remote"].drop_nulls().sum()
    particuliers = remote_sum - centrales_sum

    stats["pv_structure"] = {
        "centrales_total_kwh":   centrales_sum,
        "remote_total_kwh":      remote_sum,
        "particuliers_kwh":      particuliers,
        "pct_particuliers":      particuliers / remote_sum * 100 if remote_sum > 0 else 0,
    }

    # Offset nocturne pv_remote (sans corriger)
    night = df.filter(
        (pl.col("timestamp").dt.hour() >= 22) |
        (pl.col("timestamp").dt.hour() <= 3)
    )
    stats["pv_remote_night"] = {
        "min":    night["pv_remote"].drop_nulls().min(),
        "max":    night["pv_remote"].drop_nulls().max(),
        "median": night["pv_remote"].drop_nulls().median(),
        "n_nonzero": int((night["pv_remote"].drop_nulls() != 0).sum()),
    }

    return df, stats


# ── Graphiques Oiken ──────────────────────────────────────────────────────────
def fig_oiken_load(df: pl.DataFrame) -> go.Figure:
    daily = df.group_by_dynamic("timestamp", every="1d").agg([
        pl.col("load").mean().alias("mean"),
        pl.col("load").min().alias("min"),
        pl.col("load").max().alias("max"),
        pl.col("forecast_load").mean().alias("forecast_mean"),
    ]).sort("timestamp").to_pandas()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["timestamp"], y=daily["max"],
        fill=None, mode="lines", line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=daily["timestamp"], y=daily["min"],
        fill="tonexty", mode="lines", line=dict(width=0),
        fillcolor="rgba(46,117,182,0.12)", name="plage min-max",
    ))
    fig.add_trace(go.Scatter(
        x=daily["timestamp"], y=daily["mean"],
        mode="lines", line=dict(color=COLORS[0], width=1.5),
        name="load moyen journalier",
    ))
    fig.add_trace(go.Scatter(
        x=daily["timestamp"], y=daily["forecast_mean"],
        mode="lines", line=dict(color=COLORS[1], width=1, dash="dot"),
        name="forecast Oiken",
    ))
    apply_base(fig, "Oiken — charge normalisee (vue 3 ans, moyenne journaliere)", height=380)
    fig.update_layout(xaxis_title="Date", yaxis_title="Load normalise [-]")
    return fig


def fig_oiken_pv(df: pl.DataFrame) -> go.Figure:
    daily = df.group_by_dynamic("timestamp", every="1d").agg([
        pl.col("pv_central_valais").sum(),
        pl.col("pv_sion").sum(),
        pl.col("pv_sierre").sum(),
        pl.col("pv_remote").sum(),
    ]).sort("timestamp").to_pandas()

    fig = go.Figure()
    for col, name, color in [
        ("pv_remote",        "Remote (total)",      COLORS[2]),
        ("pv_central_valais","Centrale Valais",      COLORS[0]),
        ("pv_sion",          "Centrale Sion",        COLORS[1]),
        ("pv_sierre",        "Centrale Sierre",      COLORS[4]),
    ]:
        fig.add_trace(go.Scatter(
            x=daily["timestamp"], y=daily[col],
            mode="lines", line=dict(color=color, width=1.2),
            name=name,
        ))
    apply_base(fig, "Oiken — production PV journaliere [kWh] (donnees brutes)", height=380)
    fig.update_layout(xaxis_title="Date", yaxis_title="kWh/jour")
    return fig


def fig_oiken_pv_night(df: pl.DataFrame) -> go.Figure:
    night = df.filter(
        (pl.col("timestamp").dt.hour() >= 22) |
        (pl.col("timestamp").dt.hour() <= 3)
    ).to_pandas()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=night["timestamp"], y=night["pv_remote"],
        mode="lines", line=dict(color=COLORS[4], width=0.5),
        name="pv_remote nuit",
    ))
    apply_base(fig, "Oiken — pv_remote la nuit (22h-3h) : offset a documenter", height=360)
    fig.update_layout(yaxis_title="pv_remote [kWh]")
    return fig


def fig_oiken_nan(df: pl.DataFrame) -> go.Figure:
    cols = ["load", "forecast_load",
            "pv_central_valais", "pv_sion", "pv_sierre", "pv_remote"]
    monthly = df.group_by_dynamic("timestamp", every="1mo").agg([
        pl.col(c).is_null().sum().alias(f"nan_{c}") for c in cols
    ]).sort("timestamp").to_pandas()

    fig = go.Figure()
    for col, color in zip(cols, COLORS):
        total = df[col].is_null().sum()
        fig.add_trace(go.Bar(
            x=monthly["timestamp"], y=monthly[f"nan_{col}"],
            name=f"{col} (total={total:,})",
            marker_color=color,
        ))
    apply_base(fig, "Oiken — valeurs manquantes par mois", height=360)
    fig.update_layout(barmode="group", yaxis_title="Nb NaN")
    return fig


# ════════════════════════════════════════════════════════════════════════════
# EXPLORATION INFLUXDB
# ════════════════════════════════════════════════════════════════════════════
def explore_influx(client: InfluxDBClient) -> dict:
    """
    Interroge InfluxDB pour documenter ce qui est disponible.
    Aucune modification — lecture seule.
    """
    api   = client.query_api()
    stats = {"stations": {}}

    # ── Sites disponibles ─────────────────────────────────────────────────
    q_sites = (
        'import "influxdata/influxdb/schema"\n'
        f'schema.tagValues(bucket: "{BUCKET}", tag: "Site")'
    )
    available_sites = sorted([
        r["_value"]
        for t in api.query(org=ORG, query=q_sites)
        for r in t.records
    ])
    stats["available_sites"] = available_sites
    print(f"  Sites disponibles dans InfluxDB : {available_sites}")

    # ── Pour chaque station configuree ────────────────────────────────────
    for site_name, file_name in STATIONS.items():
        print(f"\n  Station : {site_name}")
        site_stats = {
            "historical": {},
            "predictions": {},
        }

        # ── Historique : verifier la disponibilite et la plage ────────────
        for measurement, col_name in HISTORICAL.items():
            q = (
                f'from(bucket: "{BUCKET}")'
                f' |> range(start: {START}, stop: {STOP})'
                f' |> filter(fn: (r) => r["_measurement"] == "{measurement}")'
                f' |> filter(fn: (r) => r["Site"] == "{site_name}")'
                f' |> keep(columns: ["_time", "_value"])'
                f' |> count()'
            )
            try:
                tables = api.query(org=ORG, query=q)
                n = sum(r["_value"] for t in tables for r in t.records)
            except Exception:
                n = 0

            site_stats["historical"][col_name] = {"n_points": n}
            print(f"    [HIST] {col_name:<28} {n:>9,} points")

        # ── Predictions : verifier runs disponibles et plage ──────────────
        # On sonde uniquement le run 01 de chaque variable pour ne pas
        # surcharger InfluxDB a l'exploration
        for var_key, col_base in PRED_VARS.items():
            for subtype in SUBTYPES:
                measurement = f"{var_key}_{subtype}"
                q = (
                    f'from(bucket: "{BUCKET}")'
                    f' |> range(start: {START}, stop: {STOP})'
                    f' |> filter(fn: (r) => r["_measurement"] == "{measurement}")'
                    f' |> filter(fn: (r) => r["Site"] == "{site_name}")'
                    f' |> filter(fn: (r) => r["Prediction"] == "01")'
                    f' |> keep(columns: ["_time", "_value"])'
                    f' |> count()'
                )
                try:
                    tables = api.query(org=ORG, query=q)
                    n = sum(r["_value"] for t in tables for r in t.records)
                except Exception:
                    n = 0

                site_stats["predictions"][f"{col_base}_{subtype}"] = {
                    "n_points_run01": n
                }
                print(f"    [PRED] {col_base}_{subtype:<22} run01 : {n:>6,} points")

        # ── Verifier les runs disponibles (sur hist_temperature uniquement) ─
        q_runs = (
            'import "influxdata/influxdb/schema"\n'
            f'schema.tagValues(bucket: "{BUCKET}", tag: "Prediction",'
            f' predicate: (r) => r["_measurement"] == "PRED_T_2M_ctrl"'
            f' and r["Site"] == "{site_name}")'
        )
        try:
            tables    = api.query(org=ORG, query=q_runs)
            runs_avail = sorted([r["_value"] for t in tables for r in t.records])
        except Exception:
            runs_avail = []
        site_stats["runs_available"] = runs_avail
        print(f"    Runs disponibles : {runs_avail[:5]}{'...' if len(runs_avail)>5 else ''}"
              f" ({len(runs_avail)} total)")

        # ── Structure temporelle des predictions (run 01, temperature) ────
        # Verifier le pas entre les timestamps target
        q_sample = (
            f'from(bucket: "{BUCKET}")'
            f' |> range(start: "2024-01-01T00:00:00Z", stop: "2024-01-03T00:00:00Z")'
            f' |> filter(fn: (r) => r["_measurement"] == "PRED_T_2M_ctrl")'
            f' |> filter(fn: (r) => r["Site"] == "{site_name}")'
            f' |> filter(fn: (r) => r["Prediction"] == "01")'
            f' |> keep(columns: ["_time", "_value"])'
        )
        try:
            tables  = api.query(org=ORG, query=q_sample)
            times   = sorted([r["_time"] for t in tables for r in t.records])
            if len(times) >= 2:
                diffs_h = [(times[i+1] - times[i]).total_seconds() / 3600
                           for i in range(min(5, len(times)-1))]
                site_stats["pred_step_h"] = diffs_h[0]
                print(f"    Pas temporel predictions : {diffs_h[0]}h")
                print(f"    Exemple timestamps : {times[:3]}")
            else:
                site_stats["pred_step_h"] = None
        except Exception:
            site_stats["pred_step_h"] = None

        stats["stations"][site_name] = site_stats

    return stats


# ── Graphiques InfluxDB ───────────────────────────────────────────────────────
def fig_influx_coverage(influx_stats: dict) -> go.Figure:
    """Heatmap : nb de points historiques par station x variable."""
    stations  = list(influx_stats["stations"].keys())
    variables = list(HISTORICAL.values())

    z = []
    for site in stations:
        row = []
        for col in variables:
            n = influx_stats["stations"][site]["historical"].get(col, {}).get("n_points", 0)
            row.append(n)
        z.append(row)

    # Normaliser par le max pour la couleur
    max_n = max(max(r) for r in z) if z else 1
    z_pct = [[v / max_n * 100 for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z_pct,
        x=[c.replace("hist_", "").replace("_", " ") for c in variables],
        y=stations,
        colorscale=[[0,"#E74C3C"],[0.01,"#F39C12"],[1,"#27AE60"]],
        zmin=0, zmax=100,
        text=[[f"{v:,.0f}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=10, color="white"),
        colorbar=dict(title="% du max", thickness=14),
        hoverongaps=False,
    ))
    apply_base(fig, "InfluxDB — nb de points historiques par station x variable", height=350)
    fig.update_xaxes(tickangle=-30)
    return fig


def fig_influx_pred_coverage(influx_stats: dict) -> go.Figure:
    """Heatmap : nb de points predictions (run01) par station x variable."""
    stations  = list(influx_stats["stations"].keys())
    pred_keys = [f"{col}_{st}"
                 for col in PRED_VARS.values()
                 for st in SUBTYPES]

    z = []
    for site in stations:
        row = []
        for key in pred_keys:
            n = influx_stats["stations"][site]["predictions"].get(key, {}).get("n_points_run01", 0)
            row.append(n)
        z.append(row)

    max_n = max(max(r) for r in z) if z else 1
    z_pct = [[v / max_n * 100 for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z_pct,
        x=[k.replace("pred_", "").replace("_", " ") for k in pred_keys],
        y=stations,
        colorscale=[[0,"#E74C3C"],[0.01,"#F39C12"],[1,"#27AE60"]],
        zmin=0, zmax=100,
        text=[[f"{v:,.0f}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=9, color="white"),
        colorbar=dict(title="% du max", thickness=14),
        hoverongaps=False,
    ))
    apply_base(fig,
        "InfluxDB — nb de points predictions run01 par station x variable x sous-type",
        height=350)
    fig.update_xaxes(tickangle=-40, tickfont=dict(size=9))
    return fig


# ════════════════════════════════════════════════════════════════════════════
# RAPPORT TEXTE
# ════════════════════════════════════════════════════════════════════════════
def build_text_report(oiken_stats: dict, influx_stats: dict) -> str:
    lines = []

    # ── Oiken ─────────────────────────────────────────────────────────────
    lines.append("<h2 style='color:#FFC000;font-family:Courier New;"
                 "margin-bottom:10px'>Rapport Oiken</h2>")
    lines.append("<pre style='background:#0F1117;color:#E6EDF3;padding:14px;"
                 "border-radius:6px;font-size:12px;border:1px solid #30363D'>")

    lines.append(f"  Periode       : {oiken_stats['ts_min']} -> {oiken_stats['ts_max']}")
    lines.append(f"  Lignes totales: {oiken_stats['n_rows']:,}")
    lines.append(f"  Pas dominant  : {oiken_stats['step_min']:.0f} min")
    lines.append(f"  Doublons      : {oiken_stats['n_duplicates']:,}")

    lines.append("\n  Statistiques par colonne")
    lines.append(f"  {'Colonne':<25} {'NaN':>8} {'%NaN':>7} {'Min':>10} "
                 f"{'Max':>10} {'Moy':>10} {'Neg':>8} {'Out(4s)':>10}")
    lines.append("  " + "-"*95)
    for col, s in oiken_stats["columns"].items():
        lines.append(
            f"  {col:<25} {s['n_null']:>8,} {s['pct_null']:>6.2f}% "
            f"{s['min']:>10.4f} {s['max']:>10.4f} {s['mean']:>10.4f} "
            f"{s['n_negative']:>8,} {s['n_outliers']:>10,}"
        )

    lines.append(f"\n  Trous temporels (gaps > {oiken_stats['step_min']:.0f} min) : "
                 f"{len(oiken_stats['gaps'])}")
    for g in oiken_stats["gaps"][:10]:
        lines.append(f"    {g['t_start']}  ->  {g['t_end']}"
                     f"  ({g['duration_min']:.0f} min)")
    if len(oiken_stats["gaps"]) > 10:
        lines.append(f"    ... et {len(oiken_stats['gaps'])-10} autres")

    pv = oiken_stats["pv_structure"]
    lines.append(f"\n  Structure PV (heures 10h-14h)")
    lines.append(f"    Centrales (central+sion+sierre) : {pv['centrales_total_kwh']:>12,.0f} kWh")
    lines.append(f"    pv_remote (total)               : {pv['remote_total_kwh']:>12,.0f} kWh")
    lines.append(f"    dont particuliers estimes       : {pv['particuliers_kwh']:>12,.0f} kWh"
                 f"  ({pv['pct_particuliers']:.1f}% du remote)")

    n = oiken_stats["pv_remote_night"]
    lines.append(f"\n  Offset nocturne pv_remote (22h-3h) — a corriger en normalisation")
    lines.append(f"    Min    : {n['min']:.4f} kWh")
    lines.append(f"    Max    : {n['max']:.4f} kWh")
    lines.append(f"    Mediane: {n['median']:.4f} kWh")
    lines.append(f"    Valeurs != 0 : {n['n_nonzero']:,}")
    lines.append("</pre>")

    # ── InfluxDB ──────────────────────────────────────────────────────────
    lines.append("<h2 style='color:#FFC000;font-family:Courier New;"
                 "margin:18px 0 10px'>Rapport InfluxDB MeteoSuisse</h2>")
    lines.append("<pre style='background:#0F1117;color:#E6EDF3;padding:14px;"
                 "border-radius:6px;font-size:12px;border:1px solid #30363D'>")

    lines.append(f"  Sites disponibles : {influx_stats['available_sites']}")

    for site, s in influx_stats["stations"].items():
        lines.append(f"\n  Station : {site}")
        lines.append(f"    Runs disponibles : {len(s.get('runs_available', []))} runs")
        lines.append(f"    Pas predictions  : {s.get('pred_step_h', '?')}h")

        lines.append("    Historique :")
        for col, cs in s["historical"].items():
            flag = "" if cs["n_points"] > 0 else "  ABSENT"
            lines.append(f"      {col:<28} {cs['n_points']:>9,} points{flag}")

        lines.append("    Predictions (run01) :")
        for key, cs in s["predictions"].items():
            flag = "" if cs["n_points_run01"] > 0 else "  ABSENT"
            lines.append(f"      {key:<35} {cs['n_points_run01']:>6,} points{flag}")

    lines.append("</pre>")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# ASSEMBLAGE HTML
# ════════════════════════════════════════════════════════════════════════════
def build_html(oiken_figs, influx_figs, text_report) -> str:
    all_figs = oiken_figs + influx_figs
    sections = [
        "1. Oiken — charge normalisee (3 ans)",
        "2. Oiken — production PV journaliere",
        "3. Oiken — offset nocturne pv_remote",
        "4. Oiken — valeurs manquantes par mois",
        "5. InfluxDB — couverture historique",
        "6. InfluxDB — couverture predictions (run01)",
    ]

    parts = [f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="UTF-8">
<title>Exploration — EI2</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:{BG}; color:{TEXT}; font-family:'Courier New',monospace; padding:24px; }}
  h1 {{ font-size:20px; letter-spacing:2px; margin-bottom:4px; }}
  .subtitle {{ color:{SUBTEXT}; font-size:13px; margin-bottom:28px; }}
  .section {{ margin-bottom:28px; border:1px solid #21262D;
              border-radius:8px; overflow:hidden; }}
  .sec-title {{ background:#161B22; padding:10px 16px; font-size:13px;
                color:{SUBTEXT}; letter-spacing:1px;
                border-bottom:1px solid #21262D; }}
  .content {{ padding:16px; }}
</style></head><body>
<h1>Exploration des donnees brutes</h1>
<div class="subtitle">Lecture seule — aucune modification des donnees</div>
"""]

    for i, (title, fig) in enumerate(zip(sections, all_figs)):
        fig_html = pio.to_html(fig, include_plotlyjs=(i == 0),
                               full_html=False, config={"responsive": True})
        parts.append(f"""
<div class="section">
  <div class="sec-title">{title}</div>
  {fig_html}
</div>""")

    parts.append(f"""
<div class="section">
  <div class="sec-title">Rapport textuel</div>
  <div class="content">{text_report}</div>
</div>
</body></html>""")

    return "".join(parts)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)

    # ── Oiken ─────────────────────────────────────────────────────────────
    print("Exploration Oiken...")
    df_oiken, oiken_stats = explore_oiken("Data_Oiken.csv")

    oiken_figs = [
        fig_oiken_load(df_oiken),
        fig_oiken_pv(df_oiken),
        fig_oiken_pv_night(df_oiken),
        fig_oiken_nan(df_oiken),
    ]

    # ── InfluxDB ──────────────────────────────────────────────────────────
    print("\nExploration InfluxDB...")
    client = InfluxDBClient(
        url=URL, token=TOKEN, org=ORG,
        ssl_ca_cert=certifi.where(),
        timeout=1_000_000,
    )
    influx_stats = explore_influx(client)
    client.close()

    influx_figs = [
        fig_influx_coverage(influx_stats),
        fig_influx_pred_coverage(influx_stats),
    ]

    # ── Rapport ───────────────────────────────────────────────────────────
    print("\nAssemblage du rapport...")
    text_report = build_text_report(oiken_stats, influx_stats)
    html        = build_html(oiken_figs, influx_figs, text_report)

    out = DATA_DIR / "html/exploration_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"Rapport -> {out}  ({out.stat().st_size/1024:.0f} KB)")