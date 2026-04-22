"""
Verification visuelle du calendrier
=====================================
Visualise le contenu du fichier calendar.parquet
"""

import math
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from config import DATA_DIR, FILE_CALENDAR

BG       = "#0F1117"
BG_PAPER = "#161B22"
TEXT     = "#E6EDF3"
SUBTEXT  = "#8B949E"
C_GRID   = "rgba(200,200,200,0.25)"

LAYOUT_BASE = dict(
    paper_bgcolor=BG_PAPER, plot_bgcolor=BG,
    font=dict(color=TEXT, family="Courier New, monospace", size=12),
    margin=dict(l=60, r=30, t=55, b=50),
)

if __name__ == "__main__":

    df = pl.read_parquet(FILE_CALENDAR)
    print(f"Shape : {df.shape[0]:,} lignes x {df.shape[1]} col")

    # ── Graphique 1 : distribution des heures locales ─────────────────────
    hour_counts = (df.group_by("hour").agg(pl.len().alias("count"))
                     .sort("hour").to_pandas())

    fig1 = go.Figure(go.Bar(
        x=hour_counts["hour"],
        y=hour_counts["count"],
        marker_color="#2E75B6",
        text=hour_counts["count"],
        textposition="outside",
        textfont=dict(size=9, color=TEXT),
    ))
    fig1.update_layout(
        **LAYOUT_BASE, height=360,
        title=dict(text="Distribution des heures locales (Europe/Zurich)",
                   font=dict(size=13, color=TEXT), x=0.01),
        xaxis=dict(gridcolor=C_GRID, showline=True, linecolor="#30363D",
                   tickmode="linear", dtick=1, title="Heure locale"),
        yaxis=dict(gridcolor=C_GRID, showline=True, linecolor="#30363D",
                   title="Nb points"),
    )

    # ── Graphique 2 : distribution jour de la semaine ─────────────────────
    dow_names  = ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"]
    dow_colors = ["#2E75B6","#3A8CC0","#4AA9CA","#5BC4D4",
                  "#FFC000","#ED7D31","#E74C3C"]
    dow_counts = (df.group_by("day_of_week").agg(pl.len().alias("count"))
                    .sort("day_of_week").to_pandas())

    fig2 = go.Figure(go.Bar(
        print(dow_counts["day_of_week"].to_list()),
        x=[dow_names[int(i)-1] for i in dow_counts["day_of_week"]],
        y=dow_counts["count"],
        marker_color=dow_colors,
        text=dow_counts["count"],
        textposition="outside",
        textfont=dict(size=10, color=TEXT),
    ))
    fig2.update_layout(
        **LAYOUT_BASE, height=360,
        title=dict(text="Distribution par jour de la semaine",
                   font=dict(size=13, color=TEXT), x=0.01),
        xaxis=dict(gridcolor=C_GRID, showline=True, linecolor="#30363D"),
        yaxis=dict(gridcolor=C_GRID, showline=True, linecolor="#30363D",
                   title="Nb points"),
    )

    # ── Graphique 3 : jours feries et vacances par mois ───────────────────
    monthly = (
        df.with_columns(
            pl.col("timestamp").dt.convert_time_zone("Europe/Zurich")
              .dt.strftime("%Y-%m").alias("month_str")
        )
        .group_by("month_str").agg([
            pl.col("is_holiday").sum().alias("n_holiday"),
            pl.col("is_school_holiday").sum().alias("n_school"),
            pl.col("is_weekend").sum().alias("n_weekend"),
            pl.len().alias("n_total"),
        ])
        .sort("month_str")
        .to_pandas()
    )

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=monthly["month_str"], y=monthly["n_holiday"],
        name="Jours feries", marker_color="#E74C3C", opacity=0.9,
    ))
    fig3.add_trace(go.Bar(
        x=monthly["month_str"], y=monthly["n_school"],
        name="Vacances scolaires", marker_color="#FFC000", opacity=0.9,
    ))
    fig3.add_trace(go.Bar(
        x=monthly["month_str"], y=monthly["n_weekend"],
        name="Weekend", marker_color="#2E75B6", opacity=0.6,
    ))
    fig3.update_layout(
        **LAYOUT_BASE, height=400,
        title=dict(text="Jours feries / vacances / weekend par mois",
                   font=dict(size=13, color=TEXT), x=0.01),
        barmode="group",
        xaxis=dict(gridcolor=C_GRID, showline=True, linecolor="#30363D",
                   tickangle=-45),
        yaxis=dict(gridcolor=C_GRID, showline=True, linecolor="#30363D",
                   title="Nb points 15min"),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="#30363D", borderwidth=1),
    )

    # ── Graphique 4 : encodages cycliques heure ───────────────────────────
    # Montrer que sin/cos encode correctement le cycle
    hours     = list(range(24))
    tau       = 2 * math.pi
    hour_sin  = [math.sin(h * tau / 24) for h in hours]
    hour_cos  = [math.cos(h * tau / 24) for h in hours]

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=hours, y=hour_sin,
        mode="lines+markers",
        line=dict(color="#2E75B6", width=2),
        marker=dict(size=6),
        name="hour_sin",
    ))
    fig4.add_trace(go.Scatter(
        x=hours, y=hour_cos,
        mode="lines+markers",
        line=dict(color="#FFC000", width=2),
        marker=dict(size=6),
        name="hour_cos",
    ))
    fig4.update_layout(
        **LAYOUT_BASE, height=360,
        title=dict(text="Encodage cyclique heure — sin/cos | 0h et 23h sont proches",
                   font=dict(size=13, color=TEXT), x=0.01),
        xaxis=dict(gridcolor=C_GRID, showline=True, linecolor="#30363D",
                   tickmode="linear", dtick=2, title="Heure locale"),
        yaxis=dict(gridcolor=C_GRID, showline=True, linecolor="#30363D",
                   title="Valeur", range=[-1.1, 1.1]),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="#30363D", borderwidth=1),
    )

    # ── Graphique 5 : encodage cyclique jour de l'annee ──────────────────
    doy_sample = (
        df.group_by("timestamp").agg([
            pl.col("day_of_year_sin").first(),
            pl.col("day_of_year_cos").first(),
        ])
        .sort("timestamp")
        .filter(
            pl.col("timestamp").dt.hour() == 12
        )
        .to_pandas()
    )

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=doy_sample["timestamp"],
        y=doy_sample["day_of_year_sin"],
        mode="lines",
        line=dict(color="#27AE60", width=1.5),
        name="day_of_year_sin",
    ))
    fig5.add_trace(go.Scatter(
        x=doy_sample["timestamp"],
        y=doy_sample["day_of_year_cos"],
        mode="lines",
        line=dict(color="#E74C3C", width=1.5),
        name="day_of_year_cos",
    ))
    fig5.update_layout(
        **LAYOUT_BASE, height=360,
        title=dict(text="Encodage cyclique jour de l'annee — sin/cos (valeurs a 12h)",
                   font=dict(size=13, color=TEXT), x=0.01),
        xaxis=dict(gridcolor=C_GRID, showline=True, linecolor="#30363D"),
        yaxis=dict(gridcolor=C_GRID, showline=True, linecolor="#30363D",
                   title="Valeur", range=[-1.1, 1.1]),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="#30363D", borderwidth=1),
    )

    # ── Graphique 6 : tableau apercu ─────────────────────────────────────
    sample = df.filter(
        pl.col("timestamp").dt.convert_time_zone("Europe/Zurich")
          .dt.hour() == 8
    ).head(10).to_pandas()
    sample["timestamp"] = sample["timestamp"].dt.strftime("%Y-%m-%d %H:%M UTC")

    fig6 = go.Figure(data=[go.Table(
        header=dict(
            values=list(sample.columns),
            fill_color="#21262D",
            font=dict(color="#FFC000", size=10, family="Courier New"),
            align="center",
            height=30,
        ),
        cells=dict(
            values=[sample[c] for c in sample.columns],
            fill_color=[["#161B22" if i % 2 == 0 else "#0F1117"
                         for i in range(len(sample))]],
            font=dict(color=TEXT, size=9, family="Courier New"),
            align="center",
            height=24,
        )
    )])
    fig6.update_layout(
        paper_bgcolor=BG_PAPER,
        height=380,
        title=dict(text="Apercu donnees calendrier (heures 8h locales)",
                   font=dict(size=13, color=TEXT), x=0.01),
        margin=dict(l=20, r=20, t=55, b=20),
    )

    # ── Export HTML ───────────────────────────────────────────────────────
    sections = [
        ("1. Distribution des heures locales",           fig1),
        ("2. Distribution par jour de la semaine",        fig2),
        ("3. Jours feries / vacances / weekend par mois", fig3),
        ("4. Encodage cyclique heure (sin/cos)",          fig4),
        ("5. Encodage cyclique jour de l'annee (sin/cos)",fig5),
        ("6. Apercu tableau calendrier",                  fig6),
    ]

    html_parts = [f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="UTF-8">
<title>Verification calendrier</title>
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
</style></head><body>
<h1>Verification calendrier</h1>
<div class="subtitle">
  105,120 lignes x 15 col | grille 15min UTC | heure locale Europe/Zurich
</div>
"""]

    for i, (title, fig) in enumerate(sections):
        fig_html = pio.to_html(
            fig, include_plotlyjs=(i == 0),
            full_html=False, config={"responsive": True}
        )
        html_parts.append(f"""
<div class="section">
  <div class="sec-title">{title}</div>
  {fig_html}
</div>""")

    html_parts.append("</body></html>")

    out = DATA_DIR / "verification_calendrier.html"
    out.write_text("".join(html_parts), encoding="utf-8")
    print(f"\n-> {out}")