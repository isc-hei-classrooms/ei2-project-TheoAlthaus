import math
from pathlib import Path
from datetime import datetime, timezone

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

C_LOAD      = "#2E75B6"
C_FORECAST  = "#ED7D31"
C_PV_TOTAL  = "#FFC000"
C_PV_REMOTE = "#FF6B6B"
C_ANOM      = "#E74C3C"
C_OK        = "#27AE60"
C_GRID      = "rgba(200,200,200,0.3)"
BG          = "#0F1117"
BG_PAPER    = "#161B22"
TEXT        = "#E6EDF3"
SUBTEXT     = "#8B949E"

LAYOUT_BASE = dict(
    paper_bgcolor=BG_PAPER,
    plot_bgcolor=BG,
    font=dict(color=TEXT, family="Courier New, monospace", size=12),
    legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="#30363D", borderwidth=1),
    xaxis=dict(gridcolor=C_GRID, showline=True, linecolor="#30363D"),
    yaxis=dict(gridcolor=C_GRID, showline=True, linecolor="#30363D"),
    margin=dict(l=60, r=30, t=60, b=50),
)


def load_oiken(path="Data_Oiken.csv") -> pl.DataFrame:
    df = pl.read_csv(
        path, separator=",", try_parse_dates=False,
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
        "central valais solar production [kWh]": "pv_central",
        "sion area solar production [kWh]":      "pv_sion",
        "sierre area production [kWh]":          "pv_sierre",
        "remote solar production [kWh]":         "pv_remote",
    })
    df = df.with_columns(
        (pl.col("pv_central") + pl.col("pv_sion") +
         pl.col("pv_sierre") + pl.col("pv_remote")).alias("pv_total")
    )
    return df.sort("timestamp")


def apply_base(fig, title, height=420):
    fig.update_layout(**LAYOUT_BASE, title=dict(
        text=title, font=dict(size=15, color=TEXT), x=0.01
    ), height=height)
    return fig


def fig_load_overview(df: pl.DataFrame):
    daily = df.group_by_dynamic("timestamp", every="1d").agg([
        pl.col("load").mean().alias("load_mean"),
        pl.col("load").min().alias("load_min"),
        pl.col("load").max().alias("load_max"),
        pl.col("forecast_load").mean().alias("forecast_mean"),
    ]).sort("timestamp")
    pd_daily = daily.to_pandas()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd_daily["timestamp"], y=pd_daily["load_max"],
        fill=None, mode="lines", line=dict(width=0),
        showlegend=False, name="max"
    ))
    fig.add_trace(go.Scatter(
        x=pd_daily["timestamp"], y=pd_daily["load_min"],
        fill="tonexty", mode="lines", line=dict(width=0),
        fillcolor="rgba(46,117,182,0.15)", name="plage min-max"
    ))
    fig.add_trace(go.Scatter(
        x=pd_daily["timestamp"], y=pd_daily["load_mean"],
        mode="lines", line=dict(color=C_LOAD, width=1.5),
        name="load (moy. jour.)"
    ))
    fig.add_trace(go.Scatter(
        x=pd_daily["timestamp"], y=pd_daily["forecast_mean"],
        mode="lines", line=dict(color=C_FORECAST, width=1, dash="dot"),
        name="forecast Oiken (moy.)"
    ))
    apply_base(fig, "Vue globale — charge normalisee (moyenne journaliere)", height=380)
    fig.update_layout(xaxis_title="Date", yaxis_title="Load normalise [-]")
    return fig


def fig_load_zoom(df: pl.DataFrame, n_days=14):
    start = datetime(2023, 10, 2, tzinfo=timezone.utc)
    end   = datetime(2023, 10, 2 + n_days, tzinfo=timezone.utc)
    sub = df.filter(
        (pl.col("timestamp") >= start) & (pl.col("timestamp") < end)
    ).to_pandas()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub["timestamp"], y=sub["load"],
        mode="lines", line=dict(color=C_LOAD, width=1.5),
        name="load reel"
    ))
    fig.add_trace(go.Scatter(
        x=sub["timestamp"], y=sub["forecast_load"],
        mode="lines", line=dict(color=C_FORECAST, width=1.2, dash="dash"),
        name="forecast Oiken"
    ))
    apply_base(fig, f"Zoom 15min — {n_days} jours (oct. 2023)", height=380)
    fig.update_layout(
        xaxis=dict(
            gridcolor=C_GRID, showline=True, linecolor="#30363D",
            rangeselector=dict(
                bgcolor="#21262D", activecolor="#2E75B6",
                buttons=[
                    dict(count=1, label="1j",   step="day", stepmode="backward"),
                    dict(count=3, label="3j",   step="day", stepmode="backward"),
                    dict(count=7, label="1sem", step="day", stepmode="backward"),
                    dict(step="all", label="tout"),
                ]
            ),
            rangeslider=dict(visible=True, bgcolor="#161B22", thickness=0.07),
            type="date"
        ),
        yaxis_title="Load normalise [-]"
    )
    return fig


def fig_pv_overview(df: pl.DataFrame):
    daily = df.group_by_dynamic("timestamp", every="1d").agg([
        pl.col("pv_total").sum().alias("pv_total_daily"),
        pl.col("pv_central").sum(),
        pl.col("pv_sion").sum(),
        pl.col("pv_sierre").sum(),
        pl.col("pv_remote").sum(),
    ]).sort("timestamp")
    pd_d = daily.to_pandas()

    fig = go.Figure()
    for col, name, color in [
        ("pv_central", "Central Valais", "#FFC000"),
        ("pv_sion",    "Sion",           "#FF8C00"),
        ("pv_sierre",  "Sierre",         "#FF6B35"),
        ("pv_remote",  "Remote (brut)",  C_PV_REMOTE),
    ]:
        fig.add_trace(go.Bar(
            x=pd_d["timestamp"], y=pd_d[col],
            name=name, marker_color=color
        ))
    apply_base(fig, "Production PV journaliere par zone [kWh]", height=380)
    fig.update_layout(barmode="stack", xaxis_title="Date", yaxis_title="kWh/jour")
    return fig


def fig_pv_zoom(df: pl.DataFrame):
    start = datetime(2023, 7, 3, tzinfo=timezone.utc)
    end   = datetime(2023, 7, 10, tzinfo=timezone.utc)
    sub = df.filter(
        (pl.col("timestamp") >= start) & (pl.col("timestamp") < end)
    ).to_pandas()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub["timestamp"], y=sub["pv_total"],
        mode="lines", line=dict(color=C_PV_TOTAL, width=1.5),
        fill="tozeroy", fillcolor="rgba(255,192,0,0.1)",
        name="PV total"
    ))
    fig.add_trace(go.Scatter(
        x=sub["timestamp"], y=sub["pv_remote"],
        mode="lines", line=dict(color=C_PV_REMOTE, width=1, dash="dot"),
        name="Remote (brut)"
    ))
    apply_base(fig, "Profil PV 15min — semaine ete (juillet 2023)", height=360)
    fig.update_layout(xaxis_title="Date", yaxis_title="kWh")
    return fig


def fig_remote_pv_night(df: pl.DataFrame):
    night = df.filter(
        (pl.col("timestamp").dt.hour() >= 22) |
        (pl.col("timestamp").dt.hour() <= 3)
    ).to_pandas()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=night["timestamp"], y=night["pv_remote"],
        mode="lines", line=dict(color=C_PV_REMOTE, width=0.5),
        name="pv_remote (nuit)"
    ))
    night_pl = df.filter(
        (pl.col("timestamp").dt.hour() >= 22) |
        (pl.col("timestamp").dt.hour() <= 3)
    )
    offset = night_pl["pv_remote"].median()
    fig.add_hline(y=offset, line_dash="dash", line_color="#FFC000",
                  annotation_text=f"Mediane nuit = {offset:.3f} kWh",
                  annotation_font_color="#FFC000")
    fig.add_hline(y=0, line_dash="dot", line_color=C_OK,
                  annotation_text="Zero attendu",
                  annotation_font_color=C_OK)
    apply_base(fig, "Anomalie : production PV remote la nuit (offset constant attendu)", height=360)
    fig.update_layout(xaxis_title="Date", yaxis_title="pv_remote [kWh]")
    return fig


def fig_nan_timeline(df: pl.DataFrame):
    cols_check = ["load", "forecast_load", "pv_central", "pv_sion", "pv_sierre", "pv_remote"]
    monthly = df.group_by_dynamic("timestamp", every="1mo").agg([
        pl.col(c).is_null().sum().alias(f"nan_{c}") for c in cols_check
    ]).sort("timestamp")
    pd_m = monthly.to_pandas()

    fig = go.Figure()
    colors = [C_LOAD, C_FORECAST, "#FFC000", "#FF8C00", "#FF6B35", C_PV_REMOTE]
    for col, color in zip(cols_check, colors):
        total_nan = df[col].is_null().sum()
        fig.add_trace(go.Bar(
            x=pd_m["timestamp"], y=pd_m[f"nan_{col}"],
            name=f"{col} (total={total_nan:,})",
            marker_color=color
        ))
    apply_base(fig, "Valeurs NaN par mois et par colonne", height=380)
    fig.update_layout(barmode="group", xaxis_title="Mois", yaxis_title="Nombre de NaN")
    return fig


def fig_load_hourly_profile(df: pl.DataFrame):
    profile = df.with_columns([
        pl.col("timestamp").dt.hour().alias("hour"),
        (pl.col("timestamp").dt.weekday() >= 5).cast(pl.Int8).alias("is_weekend")
    ]).group_by(["hour", "is_weekend"]).agg([
        pl.col("load").mean().alias("load_mean"),
        pl.col("load").std().alias("load_std"),
    ]).sort(["is_weekend", "hour"])

    fig = go.Figure()
    for is_we, label, color in [(0, "Semaine", C_LOAD), (1, "Weekend", C_FORECAST)]:
        sub = profile.filter(pl.col("is_weekend") == is_we).to_pandas()
        fig.add_trace(go.Scatter(
            x=sub["hour"], y=sub["load_mean"] + sub["load_std"],
            mode="lines", line=dict(width=0), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=sub["hour"], y=sub["load_mean"] - sub["load_std"],
            fill="tonexty", mode="lines", line=dict(width=0),
            fillcolor=f"rgba({'46,117,182' if is_we == 0 else '237,125,49'},0.15)",
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=sub["hour"], y=sub["load_mean"],
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=5),
            name=label
        ))
    apply_base(fig, "Profil horaire moyen — Semaine vs Weekend", height=380)
    fig.update_layout(
        xaxis=dict(gridcolor=C_GRID, tickmode="linear", dtick=2,
                   title="Heure de la journee", showline=True, linecolor="#30363D"),
        yaxis_title="Load normalise moyen [-]"
    )
    return fig


def fig_pv_vs_load(df: pl.DataFrame):
    sample = df.filter(pl.col("pv_total") > 0).sample(n=min(5000, df.shape[0]), seed=42)
    pd_s = sample.to_pandas()

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=pd_s["pv_total"], y=pd_s["load"],
        mode="markers",
        marker=dict(
            color=pd_s["pv_total"],
            colorscale="Plasma",
            size=3, opacity=0.5,
            colorbar=dict(title="PV total [kWh]", thickness=12)
        ),
        name="load vs PV"
    ))
    apply_base(fig, "Correlation PV total vs Load (heures ensoleillees)", height=400)
    fig.update_layout(xaxis_title="PV total [kWh]", yaxis_title="Load normalise [-]")
    return fig


def fig_monthly_stats(df: pl.DataFrame):
    monthly = df.group_by_dynamic("timestamp", every="1mo").agg([
        pl.col("load").mean().alias("load_mean"),
        pl.col("pv_total").sum().alias("pv_sum"),
        pl.col("forecast_load").mean().alias("forecast_mean"),
    ]).sort("timestamp").to_pandas()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=monthly["timestamp"], y=monthly["pv_sum"],
        name="PV total mensuel [kWh]",
        marker_color="rgba(255,192,0,0.6)",
    ), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=monthly["timestamp"], y=monthly["load_mean"],
        mode="lines+markers", line=dict(color=C_LOAD, width=2),
        marker=dict(size=6), name="Load moyen mensuel"
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=monthly["timestamp"], y=monthly["forecast_mean"],
        mode="lines+markers",
        line=dict(color=C_FORECAST, width=1.5, dash="dash"),
        marker=dict(size=4), name="Forecast moyen mensuel"
    ), secondary_y=False)
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Statistiques mensuelles — Load & PV", x=0.01,
                   font=dict(size=15, color=TEXT)),
        height=400
    )
    fig.update_yaxes(title_text="Load normalise [-]",     secondary_y=False, gridcolor=C_GRID)
    fig.update_yaxes(title_text="PV total mensuel [kWh]", secondary_y=True,  gridcolor="rgba(0,0,0,0)")
    return fig


def build_anomaly_report(df: pl.DataFrame) -> str:
    lines = []
    lines.append("<h2 style='color:#E6EDF3;font-family:Courier New'>Rapport d'anomalies</h2>")
    lines.append("<pre style='background:#161B22;color:#E6EDF3;padding:16px;"
                 "border-radius:6px;font-size:13px;border:1px solid #30363D'>")

    lines.append("Valeurs manquantes (NaN)")
    cols = ["load", "forecast_load", "pv_central", "pv_sion", "pv_sierre", "pv_remote"]
    for c in cols:
        n   = df[c].is_null().sum()
        pct = n / df.shape[0] * 100
        lines.append(f"  {c:<35} {n:>6,} NaN  ({pct:.2f}%)")

    lines.append("\nTrous temporels (gaps > 15min)")
    ts        = df["timestamp"].sort()
    diffs_sec = ts.diff().drop_nulls().dt.total_seconds()
    gap_idx   = [i for i, s in enumerate(diffs_sec.to_list()) if s > 900]
    if len(gap_idx) == 0:
        lines.append("  Aucun trou detecte")
    else:
        for idx in gap_idx[:10]:
            t1  = ts[idx]
            t2  = ts[idx + 1]
            dur = diffs_sec[idx] / 60
            lines.append(f"  Gap de {dur:.0f} min entre {t1} et {t2}")
        if len(gap_idx) > 10:
            lines.append(f"  ... et {len(gap_idx) - 10} autres gaps")

    lines.append("\nAnomalie pv_remote (production nocturne)")
    night = df.filter(
        (pl.col("timestamp").dt.hour() >= 22) |
        (pl.col("timestamp").dt.hour() <= 3)
    )
    remote_night = night["pv_remote"].drop_nulls()
    lines.append(f"  Min nuit    : {remote_night.min():.4f} kWh")
    lines.append(f"  Max nuit    : {remote_night.max():.4f} kWh")
    lines.append(f"  Mediane nuit: {remote_night.median():.4f} kWh")
    lines.append(f"  Valeurs > 0 : {(remote_night > 0).sum():,} sur {len(remote_night):,}")

    lines.append("\nValeurs aberrantes load (|z-score| > 4)")
    load       = df["load"].drop_nulls()
    mu, sigma  = load.mean(), load.std()
    n_outliers = ((load - mu).abs() > 4 * sigma).sum()
    lines.append(f"  Mean  : {mu:.4f}  |  Std : {sigma:.4f}")
    lines.append(f"  Min   : {load.min():.4f}  |  Max : {load.max():.4f}")
    lines.append(f"  Outliers (|z|>4) : {n_outliers:,}")

    pv_neg = (df["pv_total"] < 0).sum()
    lines.append(f"\nPV total negatif : {pv_neg} cas")

    n_dup = df.shape[0] - df["timestamp"].n_unique()
    lines.append(f"\nDoublons de timestamps : {n_dup}")

    lines.append("</pre>")
    return "\n".join(lines)


def build_html(figures: list, anomaly_report: str) -> str:
    titles = [
        "1. Vue globale — charge 3 ans",
        "2. Zoom interactif 15min",
        "3. Production PV journaliere",
        "4. Profil PV (semaine ete)",
        "5. Anomalie pv_remote (nuit)",
        "6. Valeurs NaN par mois",
        "7. Profil horaire moyen",
        "8. Correlation PV vs Load",
        "9. Stats mensuelles",
    ]
    html_parts = [f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>EDA — Projet Energy Informatics 2</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: {BG}; color: {TEXT}; font-family: 'Courier New', monospace; padding: 24px; }}
  h1 {{ font-size: 22px; color: {TEXT}; margin-bottom: 4px; letter-spacing: 2px; }}
  .subtitle {{ color: {SUBTEXT}; font-size: 13px; margin-bottom: 28px; }}
  .section {{ margin-bottom: 32px; border: 1px solid #21262D;
              border-radius: 8px; overflow: hidden; }}
  .sec-title {{ background: #161B22; padding: 10px 16px;
                font-size: 13px; color: {SUBTEXT}; letter-spacing: 1px;
                border-bottom: 1px solid #21262D; }}
  .chart-wrap {{ padding: 0; }}
</style>
</head>
<body>
<h1>EDA — Projet Energy Informatics 2</h1>
<div class="subtitle">Oiken load curve | Valais | Oct 2022 — Sep 2025 | pas 15min</div>
"""]

    for i, (fig, title) in enumerate(zip(figures, titles)):
        fig_html = pio.to_html(fig, include_plotlyjs=(i == 0),
                               full_html=False, config={"responsive": True})
        html_parts.append(f"""
<div class="section">
  <div class="sec-title">{title}</div>
  <div class="chart-wrap">{fig_html}</div>
</div>""")

    html_parts.append(f"""
<div class="section">
  <div class="sec-title">Rapport d'anomalies</div>
  <div style="padding:16px">{anomaly_report}</div>
</div>
</body></html>""")

    return "".join(html_parts)


if __name__ == "__main__":
    df = load_oiken("Data_Oiken.csv")

    figures = [
        fig_load_overview(df),
        fig_load_zoom(df),
        fig_pv_overview(df),
        fig_pv_zoom(df),
        fig_remote_pv_night(df),
        fig_nan_timeline(df),
        fig_load_hourly_profile(df),
        fig_pv_vs_load(df),
        fig_monthly_stats(df),
    ]

    anomaly_report = build_anomaly_report(df)
    html = build_html(figures, anomaly_report)

    out = OUTPUT_DIR / "eda_report.html"
    out.write_text(html, encoding="utf-8")