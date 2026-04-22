from pathlib import Path
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

DATA_DIR   = Path("data")
OUTPUT_DIR = Path("data")

HIST_COLS = {
    "hist_temperature":   ("Temperature 2m",      "°C",      "#E74C3C"),
    "hist_radiation":     ("Rayonnement global",   "W/m²",    "#FFC000"),
    "hist_sunshine":      ("Ensoleillement",       "s/10min", "#F39C12"),
    "hist_precipitation": ("Precipitations",       "mm",      "#3498DB"),
    "hist_humidity":      ("Humidite relative",    "%",       "#1ABC9C"),
    "hist_pressure":      ("Pression atmos.",      "hPa",     "#9B59B6"),
    "hist_wind_speed":    ("Vitesse vent",          "m/s",    "#2ECC71"),
    "hist_wind_dir":      ("Direction vent",        "°",      "#95A5A6"),
    "hist_gust":          ("Rafale max",            "m/s",    "#E67E22"),
}

BG       = "#0F1117"
BG_PAPER = "#161B22"
TEXT     = "#E6EDF3"
SUBTEXT  = "#8B949E"
C_GRID   = "rgba(200,200,200,0.25)"

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


def discover_stations() -> dict[str, pl.DataFrame]:
    stations = {}
    for p in sorted(DATA_DIR.glob("meteo_*.parquet")):
        if "_predictions" in p.name:
            continue
        df = pl.read_parquet(p)
        hist_cols = [c for c in df.columns if c.startswith("hist_")]
        if not hist_cols:
            continue
        name = (p.stem
                .replace("meteo_", "")
                .replace("_historical", "")
                .replace("_", " ")
                .title())
        stations[name] = df.select(["timestamp"] + hist_cols).sort("timestamp")
    return stations


def fig_coverage(stations: dict) -> go.Figure:
    fig = go.Figure()
    colors = ["#2E75B6","#ED7D31","#FFC000","#27AE60","#E74C3C","#9B59B6","#1ABC9C"]

    for i, (station, df) in enumerate(stations.items()):
        color = colors[i % len(colors)]
        monthly = df.group_by_dynamic("timestamp", every="1mo").agg([
            (pl.col("hist_temperature").is_not_null().sum() /
             pl.col("hist_temperature").len() * 100).alias("coverage")
        ]).sort("timestamp")
        pd_m = monthly.to_pandas()
        fig.add_trace(go.Bar(
            x=pd_m["timestamp"],
            y=[i + 1] * len(pd_m),
            base=0,
            marker=dict(
                color=[f"rgba({int(color[1:3],16)},"
                       f"{int(color[3:5],16)},"
                       f"{int(color[5:7],16)},{v/100:.2f})"
                       for v in pd_m["coverage"]],
                line=dict(width=0)
            ),
            name=station,
            hovertemplate="%{x|%b %Y}<br>Couverture: %{customdata:.0f}%",
            customdata=pd_m["coverage"],
            showlegend=True,
        ))

    apply_base(fig, "Couverture des donnees par station (opacite = % non-null)", height=350)
    fig.update_layout(barmode="overlay", bargap=0.1)
    fig.update_yaxes(
        tickvals=list(range(1, len(stations)+1)),
        ticktext=list(stations.keys()),
        title=""
    )
    fig.update_xaxes(title="Mois")
    return fig


def fig_nan_heatmap(stations: dict) -> go.Figure:
    station_names = list(stations.keys())
    var_names     = [HIST_COLS[c][0] for c in HIST_COLS if c in
                     list(stations.values())[0].columns]
    var_cols      = [c for c in HIST_COLS if c in
                     list(stations.values())[0].columns]

    z = []
    for station, df in stations.items():
        row = []
        for col in var_cols:
            pct = df[col].is_null().sum() / df.shape[0] * 100 if col in df.columns else 100.0
            row.append(pct)
        z.append(row)

    fig = go.Figure(go.Heatmap(
        z=z, x=var_names, y=station_names,
        colorscale=[[0,"#27AE60"],[0.01,"#F39C12"],[1,"#E74C3C"]],
        zmin=0, zmax=100,
        text=[[f"{v:.1f}%" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=11, color="white"),
        colorbar=dict(title="% NaN", thickness=14, len=0.8),
        hoverongaps=False,
    ))
    apply_base(fig, "Heatmap des valeurs manquantes — stations x variables (%)", height=350)
    fig.update_xaxes(tickangle=-30)
    return fig


def fig_variable_overview(stations: dict, col: str) -> go.Figure:
    label, unit, _ = HIST_COLS.get(col, (col, "", "#888"))
    colors = ["#2E75B6","#ED7D31","#FFC000","#27AE60","#E74C3C","#9B59B6","#1ABC9C"]

    fig = go.Figure()
    for i, (station, df) in enumerate(stations.items()):
        if col not in df.columns:
            continue
        daily = df.group_by_dynamic("timestamp", every="1d").agg(
            pl.col(col).mean().alias("mean")
        ).sort("timestamp").to_pandas()

        fig.add_trace(go.Scatter(
            x=daily["timestamp"], y=daily["mean"],
            mode="lines", line=dict(color=colors[i % len(colors)], width=1.2),
            name=station, opacity=0.85,
        ))

    apply_base(fig, f"{label} — moyenne journaliere par station [{unit}]", height=380)
    fig.update_layout(
        xaxis=dict(
            gridcolor=C_GRID, showline=True, linecolor="#30363D",
            rangeselector=dict(
                bgcolor="#21262D", activecolor="#2E75B6",
                buttons=[
                    dict(count=1, label="1m",  step="month", stepmode="backward"),
                    dict(count=3, label="3m",  step="month", stepmode="backward"),
                    dict(count=6, label="6m",  step="month", stepmode="backward"),
                    dict(count=1, label="1an", step="year",  stepmode="backward"),
                    dict(step="all", label="tout"),
                ]
            ),
            rangeslider=dict(visible=True, bgcolor="#161B22", thickness=0.06),
            type="date",
        ),
        yaxis_title=f"{label} [{unit}]"
    )
    return fig


def fig_seasonal_profiles(stations: dict) -> go.Figure:
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Temperature moyenne par mois [°C]",
                                        "Radiation moyenne par mois [W/m²]"])
    colors = ["#2E75B6","#ED7D31","#FFC000","#27AE60","#E74C3C","#9B59B6","#1ABC9C"]
    month_labels = ["Jan","Fev","Mar","Avr","Mai","Jun",
                    "Jul","Aou","Sep","Oct","Nov","Dec"]

    for i, (station, df) in enumerate(stations.items()):
        c = colors[i % len(colors)]
        agg_exprs = []
        if "hist_temperature" in df.columns:
            agg_exprs.append(pl.col("hist_temperature").mean().alias("temp"))
        if "hist_radiation" in df.columns:
            agg_exprs.append(pl.col("hist_radiation").mean().alias("rad"))
        if not agg_exprs:
            continue
        prof = df.with_columns(
            pl.col("timestamp").dt.month().alias("month")
        ).group_by("month").agg(agg_exprs).sort("month").to_pandas()

        if "temp" in prof.columns:
            fig.add_trace(go.Scatter(
                x=month_labels, y=prof["temp"], mode="lines+markers",
                line=dict(color=c, width=1.5), marker=dict(size=5),
                name=station, legendgroup=station, showlegend=True,
            ), row=1, col=1)
        if "rad" in prof.columns:
            fig.add_trace(go.Scatter(
                x=month_labels, y=prof["rad"], mode="lines+markers",
                line=dict(color=c, width=1.5), marker=dict(size=5),
                name=station, legendgroup=station, showlegend=False,
            ), row=1, col=2)

    fig.update_layout(**LAYOUT_BASE, height=400,
        title=dict(text="Profils saisonniers par station", x=0.01,
                   font=dict(size=14, color=TEXT)))
    fig.update_yaxes(gridcolor=C_GRID)
    fig.update_xaxes(gridcolor=C_GRID)
    return fig


def fig_daily_profile(stations: dict) -> go.Figure:
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Radiation moyenne par heure [W/m²]",
                                        "Temperature moyenne par heure [°C]"])
    colors = ["#2E75B6","#ED7D31","#FFC000","#27AE60","#E74C3C","#9B59B6","#1ABC9C"]

    for i, (station, df) in enumerate(stations.items()):
        c = colors[i % len(colors)]
        agg_exprs = []
        if "hist_radiation" in df.columns:
            agg_exprs.append(pl.col("hist_radiation").mean().alias("rad"))
        if "hist_temperature" in df.columns:
            agg_exprs.append(pl.col("hist_temperature").mean().alias("temp"))
        if not agg_exprs:
            continue
        hourly = df.with_columns(
            pl.col("timestamp").dt.hour().alias("hour")
        ).group_by("hour").agg(agg_exprs).sort("hour").to_pandas()

        if "rad" in hourly.columns:
            fig.add_trace(go.Scatter(
                x=hourly["hour"], y=hourly["rad"], mode="lines+markers",
                line=dict(color=c, width=1.5), marker=dict(size=4),
                name=station, legendgroup=station, showlegend=True,
                fill="tozeroy" if i == 0 else None,
                fillcolor="rgba(255,192,0,0.05)" if i == 0 else None,
            ), row=1, col=1)
        if "temp" in hourly.columns:
            fig.add_trace(go.Scatter(
                x=hourly["hour"], y=hourly["temp"], mode="lines+markers",
                line=dict(color=c, width=1.5), marker=dict(size=4),
                name=station, legendgroup=station, showlegend=False,
            ), row=1, col=2)

    fig.update_layout(**LAYOUT_BASE, height=400,
        title=dict(text="Profils horaires moyens par station", x=0.01,
                   font=dict(size=14, color=TEXT)))
    fig.update_xaxes(gridcolor=C_GRID, tickmode="linear", dtick=3, title="Heure")
    fig.update_yaxes(gridcolor=C_GRID)
    return fig


def fig_correlation_matrix(df: pl.DataFrame, station_name: str) -> go.Figure:
    hist_cols = [c for c in df.columns if c.startswith("hist_")]
    labels    = [HIST_COLS.get(c, (c, "", ""))[0] for c in hist_cols]

    corr_matrix = []
    for c1 in hist_cols:
        row = []
        for c2 in hist_cols:
            try:
                r = df.select(pl.corr(c1, c2)).item()
                row.append(round(r, 3) if r is not None else 0)
            except Exception:
                row.append(0)
        corr_matrix.append(row)

    fig = go.Figure(go.Heatmap(
        z=corr_matrix, x=labels, y=labels,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr_matrix],
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorbar=dict(title="r", thickness=14, len=0.8),
        hoverongaps=False,
    ))
    apply_base(fig, f"Matrice de correlation entre variables — {station_name}", height=420)
    fig.update_xaxes(tickangle=-35)
    return fig


def build_anomaly_report(stations: dict) -> str:
    lines = []
    lines.append("<h2 style='color:#E6EDF3;font-family:Courier New;margin-bottom:12px'>"
                 "Rapport d'anomalies meteorologiques</h2>")

    for station, df in stations.items():
        lines.append(f"<details open><summary style='cursor:pointer;color:#FFC000;"
                     f"font-family:Courier New;font-size:14px;padding:6px 0'>"
                     f"▶ {station}</summary>")
        lines.append("<pre style='background:#0F1117;color:#E6EDF3;padding:14px;"
                     "border-radius:6px;font-size:12px;border:1px solid #30363D;"
                     "margin:8px 0 16px 0'>")

        lines.append(f"  Periode : {df['timestamp'].min()} -> {df['timestamp'].max()}")
        lines.append(f"  Lignes  : {df.shape[0]:,}")

        lines.append("\n  Valeurs manquantes")
        hist_cols = [c for c in df.columns if c.startswith("hist_")]
        for c in hist_cols:
            n   = df[c].is_null().sum()
            pct = n / df.shape[0] * 100
            lbl = HIST_COLS.get(c, (c,))[0]
            lines.append(f"  {lbl:<28} {n:>8,} NaN  ({pct:5.2f}%)")

        lines.append("\n  Trous temporels (gaps > 20min)")
        ts    = df["timestamp"].sort()
        diffs = ts.diff().drop_nulls().dt.total_seconds()
        gaps  = [(i, s) for i, s in enumerate(diffs.to_list()) if s > 1200]
        if not gaps:
            lines.append("  Aucun trou detecte")
        else:
            for idx, sec in gaps[:8]:
                t1 = ts[idx]
                t2 = ts[idx + 1]
                lines.append(f"  Gap {sec/60:.0f} min  :  {t1}  ->  {t2}")
            if len(gaps) > 8:
                lines.append(f"  ... et {len(gaps)-8} autres gaps")

        lines.append("\n  Anomalies physiques")
        if "hist_radiation" in df.columns:
            night   = df.filter(
                (pl.col("timestamp").dt.hour() >= 22) |
                (pl.col("timestamp").dt.hour() <= 3)
            )["hist_radiation"].drop_nulls()
            n_night = (night > 0).sum()
            lines.append(f"  Radiation nocturne > 0    : {n_night:>8,} pts")

        if "hist_radiation" in df.columns:
            rad   = df["hist_radiation"].drop_nulls()
            n_neg = (rad < 0).sum()
            lines.append(f"  Radiation negative         : {n_neg:>8,} pts")

        if "hist_humidity" in df.columns:
            hum   = df["hist_humidity"].drop_nulls()
            n_bad = ((hum < 0) | (hum > 100)).sum()
            lines.append(f"  Humidite hors [0,100]%    : {n_bad:>8,} pts")

        if "hist_wind_dir" in df.columns:
            wd    = df["hist_wind_dir"].drop_nulls()
            n_bad = ((wd < 0) | (wd > 360)).sum()
            lines.append(f"  Direction vent hors [0,360]: {n_bad:>8,} pts")

        if "hist_temperature" in df.columns:
            t          = df["hist_temperature"].drop_nulls()
            mu, std    = t.mean(), t.std()
            n_out      = ((t - mu).abs() > 4 * std).sum()
            lines.append(f"  Temperature outliers |z|>4 : {n_out:>8,} pts"
                         f"  [min={t.min():.1f}°C  max={t.max():.1f}°C]")

        n_dup = df.shape[0] - df["timestamp"].n_unique()
        lines.append(f"\n  Doublons de timestamps    : {n_dup:>8,}")

        lines.append("</pre></details>")

    return "\n".join(lines)


def build_html(sections: list, anomaly_report: str) -> str:
    html_parts = [f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>EDA Meteo — Projet Energy Informatics 2</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: {BG}; color: {TEXT}; font-family: 'Courier New', monospace; padding: 24px; }}
  h1 {{ font-size: 22px; color: {TEXT}; margin-bottom: 4px; letter-spacing: 2px; }}
  .subtitle {{ color: {SUBTEXT}; font-size: 13px; margin-bottom: 28px; }}
  .section {{ margin-bottom: 28px; border: 1px solid #21262D;
              border-radius: 8px; overflow: hidden; }}
  .sec-title {{ background: #161B22; padding: 10px 16px;
                font-size: 13px; color: {SUBTEXT}; letter-spacing: 1px;
                border-bottom: 1px solid #21262D; }}
  details summary::-webkit-details-marker {{ color: #FFC000; }}
</style>
</head>
<body>
<h1>EDA Meteorologique — MeteoSuisse</h1>
<div class="subtitle">Donnees historiques | Valais | pas 10min | oct 2022 — sept 2025</div>
"""]

    for i, (title, fig) in enumerate(sections):
        fig_html = pio.to_html(fig, include_plotlyjs=(i == 0),
                               full_html=False, config={"responsive": True})
        html_parts.append(f"""
<div class="section">
  <div class="sec-title">{title}</div>
  {fig_html}
</div>""")

    html_parts.append(f"""
<div class="section">
  <div class="sec-title">Rapport d'anomalies</div>
  <div style="padding:16px">{anomaly_report}</div>
</div>
</body></html>""")
    return "".join(html_parts)


if __name__ == "__main__":
    stations = discover_stations()

    main_station_name = next(iter(stations))
    main_station_df   = stations[main_station_name]

    sections = [
        ("1. Couverture temporelle par station",
            fig_coverage(stations)),
        ("2. Heatmap NaN — stations x variables",
            fig_nan_heatmap(stations)),
        ("3. Temperature — vue globale toutes stations",
            fig_variable_overview(stations, "hist_temperature")),
        ("4. Rayonnement global — vue globale",
            fig_variable_overview(stations, "hist_radiation")),
        (f"5. Matrice de correlation — {main_station_name}",
            fig_correlation_matrix(main_station_df, main_station_name)),
        ("6. Profils saisonniers (temp. & radiation)",
            fig_seasonal_profiles(stations)),
        ("7. Profils horaires moyens",
            fig_daily_profile(stations)),
    ]

    anomaly_report = build_anomaly_report(stations)
    html = build_html(sections, anomaly_report)

    out = OUTPUT_DIR / "eda_meteo.html"
    out.write_text(html, encoding="utf-8")