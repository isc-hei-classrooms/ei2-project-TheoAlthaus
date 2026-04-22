from pathlib import Path
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import math

DATA_DIR = Path("data")

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


def load_stations() -> dict:
    stations = {}
    for p in sorted(DATA_DIR.glob("meteo_*.parquet")):
        if "_predictions" in p.name:
            continue
        df = pl.read_parquet(p)
        if "hist_radiation" not in df.columns:
            continue
        name = (p.stem
                .replace("meteo_", "")
                .replace("_historical", "")
                .replace("_", " ").title())
        stations[name] = df.select(["timestamp", "hist_radiation"]).sort("timestamp")
    return stations


def fig_night_histogram(stations: dict) -> go.Figure:
    n       = len(stations)
    n_cols  = min(4, n)
    n_rows  = math.ceil(n / n_cols)

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=list(stations.keys()),
        vertical_spacing=0.18,
        horizontal_spacing=0.08,
    )
    for i, (name, df) in enumerate(stations.items()):
        row = i // n_cols + 1
        col = i %  n_cols + 1
        night = df.filter(
            (pl.col("timestamp").dt.hour() >= 22) |
            (pl.col("timestamp").dt.hour() <= 3)
        )["hist_radiation"].drop_nulls()

        fig.add_trace(go.Histogram(
            x=night.to_list(),
            nbinsx=60,
            marker_color=COLORS[i % len(COLORS)],
            opacity=0.85,
            name=name,
            showlegend=False,
        ), row=row, col=col)

        fig.add_vline(x=0, line_color="#E74C3C", line_dash="dash",
                      line_width=1.5, row=row, col=col)

        p95 = night.quantile(0.95)
        p99 = night.quantile(0.99)
        mx  = night.max()
        fig.add_annotation(
            text=f"p95={p95:.1f}<br>p99={p99:.1f}<br>max={mx:.1f}",
            xref=f"x{'' if i == 0 else i+1}",
            yref=f"y{'' if i == 0 else i+1}",
            x=0.98, y=0.98,
            xanchor="right", yanchor="top",
            showarrow=False,
            font=dict(size=10, color="#FFC000"),
            bgcolor="rgba(0,0,0,0.5)",
            row=row, col=col,
        )

    apply_base(fig, "Distribution du rayonnement nocturne (22h-4h) par station [W/m²]",
               height=300 * n_rows)
    fig.update_xaxes(title_text="W/m²")
    fig.update_yaxes(title_text="Nb points")
    return fig


def fig_night_vs_day(stations: dict) -> go.Figure:
    fig = go.Figure()

    for i, (name, df) in enumerate(stations.items()):
        color = COLORS[i % len(COLORS)]

        night = df.filter(
            (pl.col("timestamp").dt.hour() >= 22) |
            (pl.col("timestamp").dt.hour() <= 3)
        )["hist_radiation"].drop_nulls()

        day = df.filter(
            (pl.col("timestamp").dt.hour() >= 10) &
            (pl.col("timestamp").dt.hour() <= 14)
        )["hist_radiation"].drop_nulls()

        if len(night) > 30000:
            night = night.sample(30000, seed=42)
        if len(day) > 30000:
            day = day.sample(30000, seed=42)

        fig.add_trace(go.Box(
            y=night.to_list(),
            name=f"{name} — nuit",
            marker_color=color,
            line_color=color,
            boxmean=True,
            legendgroup=name,
            offsetgroup=f"{i}_nuit",
        ))
        fig.add_trace(go.Box(
            y=day.to_list(),
            name=f"{name} — jour",
            marker_color=color,
            line_color=color,
            boxmean=True,
            opacity=0.4,
            legendgroup=name,
            offsetgroup=f"{i}_jour",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.2)",
        ))

    apply_base(fig, "Rayonnement nocturne vs diurne (10h-14h) par station [W/m²]",
               height=480)
    fig.update_layout(boxmode="group", yaxis_title="W/m²")
    fig.add_hline(y=0, line_color="#E74C3C", line_dash="dash",
                  annotation_text="Zero", annotation_font_color="#E74C3C")
    return fig


def fig_night_timeseries(stations: dict) -> go.Figure:
    t0 = pl.datetime(2023, 12, 15, 0, 0, 0, time_unit="us", time_zone="UTC")
    t1 = pl.datetime(2023, 12, 20, 0, 0, 0, time_unit="us", time_zone="UTC")

    fig = go.Figure()
    for i, (name, df) in enumerate(stations.items()):
        sub = df.filter(
            (pl.col("timestamp") >= t0) & (pl.col("timestamp") <= t1)
        ).to_pandas()

        fig.add_trace(go.Scatter(
            x=sub["timestamp"], y=sub["hist_radiation"],
            mode="lines",
            line=dict(color=COLORS[i % len(COLORS)], width=1.2),
            name=name,
        ))

    fig.add_hrect(y0=-50, y1=5, fillcolor="rgba(255,107,107,0.06)",
                  line_width=0, annotation_text="Zone attendue la nuit",
                  annotation_font_color="#E74C3C", annotation_font_size=10)

    apply_base(fig, "Serie temporelle rayonnement — 5 jours hiver (dec. 2023) [W/m²]",
               height=400)
    fig.update_layout(
        xaxis=dict(
            gridcolor=C_GRID, showline=True, linecolor="#30363D",
            rangeslider=dict(visible=True, bgcolor="#161B22", thickness=0.06),
            type="date",
        ),
        yaxis_title="W/m²"
    )
    return fig


def build_summary_table(stations: dict) -> str:
    rows = []
    rows.append(
        "<table style='width:100%;border-collapse:collapse;"
        "font-family:Courier New;font-size:12px;color:#E6EDF3'>"
    )
    rows.append(
        "<tr style='background:#21262D;color:#FFC000'>"
        "<th style='padding:8px 12px;text-align:left'>Station</th>"
        "<th style='padding:8px 12px'>Min nuit</th>"
        "<th style='padding:8px 12px'>Max nuit</th>"
        "<th style='padding:8px 12px'>Mediane nuit</th>"
        "<th style='padding:8px 12px'>p95 nuit</th>"
        "<th style='padding:8px 12px'>p99 nuit</th>"
        "<th style='padding:8px 12px'>Max jour</th>"
        "<th style='padding:8px 12px'>Ratio max_nuit/max_jour</th>"
        "<th style='padding:8px 12px'>Verdict</th>"
        "</tr>"
    )
    for i, (name, df) in enumerate(stations.items()):
        night = df.filter(
            (pl.col("timestamp").dt.hour() >= 22) |
            (pl.col("timestamp").dt.hour() <= 3)
        )["hist_radiation"].drop_nulls()

        day = df.filter(
            (pl.col("timestamp").dt.hour() >= 10) &
            (pl.col("timestamp").dt.hour() <= 14)
        )["hist_radiation"].drop_nulls()

        mn    = night.min()
        mx    = night.max()
        med   = night.median()
        p95   = night.quantile(0.95)
        p99   = night.quantile(0.99)
        mxd   = day.max()
        ratio = mx / mxd if mxd and mxd > 0 else 0

        if p99 < 5 and mx < 20:
            verdict = "Negligeable"
            vcolor  = "#27AE60"
        elif p99 < 30 and ratio < 0.05:
            verdict = "Faible bruit"
            vcolor  = "#F39C12"
        else:
            verdict = "A investiguer"
            vcolor  = "#E74C3C"

        bg = "#161B22" if i % 2 == 0 else "#0F1117"
        rows.append(
            f"<tr style='background:{bg}'>"
            f"<td style='padding:7px 12px;font-weight:bold'>{name}</td>"
            f"<td style='padding:7px 12px;text-align:center'>{mn:.2f}</td>"
            f"<td style='padding:7px 12px;text-align:center'>{mx:.2f}</td>"
            f"<td style='padding:7px 12px;text-align:center'>{med:.3f}</td>"
            f"<td style='padding:7px 12px;text-align:center'>{p95:.2f}</td>"
            f"<td style='padding:7px 12px;text-align:center'>{p99:.2f}</td>"
            f"<td style='padding:7px 12px;text-align:center'>{mxd:.1f}</td>"
            f"<td style='padding:7px 12px;text-align:center'>{ratio:.3f}</td>"
            f"<td style='padding:7px 12px;text-align:center;"
            f"color:{vcolor};font-weight:bold'>{verdict}</td>"
            f"</tr>"
        )
    rows.append("</table>")
    return "\n".join(rows)


def build_html(figures, summary_table) -> str:
    sections = [
        ("1. Distribution des valeurs nocturnes par station", figures[0]),
        ("2. Comparaison nuit vs plein jour (10h-14h)",       figures[1]),
        ("3. Serie temporelle — zoom hiver 5 jours",          figures[2]),
    ]
    html_parts = [f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="UTF-8">
<title>Analyse rayonnement nocturne</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:{BG}; color:{TEXT}; font-family:'Courier New',monospace; padding:24px; }}
  h1 {{ font-size:20px; color:{TEXT}; margin-bottom:4px; letter-spacing:2px; }}
  .subtitle {{ color:{SUBTEXT}; font-size:13px; margin-bottom:28px; }}
  .section {{ margin-bottom:28px; border:1px solid #21262D;
              border-radius:8px; overflow:hidden; }}
  .sec-title {{ background:#161B22; padding:10px 16px; font-size:13px;
                color:{SUBTEXT}; letter-spacing:1px; border-bottom:1px solid #21262D; }}
</style></head><body>
<h1>Analyse du rayonnement nocturne — MeteoSuisse</h1>
<div class="subtitle">Est-ce un vrai probleme ou du bruit negligeable ?</div>
"""]

    html_parts.append(f"""
<div class="section">
  <div class="sec-title">Synthese statistique — nuit (22h-4h) vs jour (10h-14h)</div>
  <div style="padding:16px;overflow-x:auto">{summary_table}</div>
</div>""")

    for i, (title, fig) in enumerate(sections):
        fig_html = pio.to_html(fig, include_plotlyjs=(i == 0),
                               full_html=False, config={"responsive": True})
        html_parts.append(f"""
<div class="section">
  <div class="sec-title">{title}</div>
  {fig_html}
</div>""")

    html_parts.append("</body></html>")
    return "".join(html_parts)


if __name__ == "__main__":
    stations = load_stations()
    figs = [
        fig_night_histogram(stations),
        fig_night_vs_day(stations),
        fig_night_timeseries(stations),
    ]
    summary = build_summary_table(stations)
    html = build_html(figs, summary)

    out = DATA_DIR / "eda_radiation_nuit.html"
    out.write_text(html, encoding="utf-8")