"""
Verification visuelle normalisation previsions — tous les lags
===============================================================
"""

import polars as pl
import plotly.graph_objects as go
import plotly.io as pio

from config import DATA_DIR, FILE_METEO_PRED_CLEAN

FILE_NAME = "sion"
STATION   = "Sion"

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

COLORS = [
    "#E74C3C","#E67E22","#F39C12","#FFC000","#F1C40F",
    "#2ECC71","#27AE60","#1ABC9C","#16A085","#3498DB",
    "#2980B9","#2E75B6","#1F618D","#9B59B6","#8E44AD",
    "#7D3C98","#D35400","#CB4335","#BA4A00","#784212",
    "#1A5276","#154360","#0B5345","#145A32","#4A235A",
    "#2C3E50","#7F8C8D","#95A5A6","#BDC3C7","#AAB7B8",
    "#717D7E","#5D6D7E","#4D5656",
]

if __name__ == "__main__":

    df = pl.read_parquet(FILE_METEO_PRED_CLEAN(FILE_NAME))

    print(f"Shape : {df.shape[0]:,} lignes x {df.shape[1]} col")

    # Periode de visualisation : 3 jours d'ete
    t_start = pl.datetime(2023, 6, 15, 0, 0, 0, time_unit="us", time_zone="UTC")
    t_end   = pl.datetime(2023, 6, 18, 0, 0, 0, time_unit="us", time_zone="UTC")

    sub = df.filter(
        (pl.col("timestamp_target") >= t_start) &
        (pl.col("timestamp_target") <= t_end)
    ).to_pandas()

    # ── Graphique 1 : taux de remplissage par lag ─────────────────────────
    fill_lags, fill_pcts, fill_vals = [], [], []
    for lag in range(1, 34):
        col   = f"lag_{lag:02d}h_pred_temperature_ctrl"
        if col not in df.columns:
            continue
        n_val = df.shape[0] - df[col].is_null().sum()
        pct   = n_val / df.shape[0] * 100
        fill_lags.append(lag)
        fill_pcts.append(round(pct, 2))
        fill_vals.append(n_val)

    fig_fill = go.Figure(go.Bar(
        x=[f"lag_{l:02d}h" for l in fill_lags],
        y=fill_pcts,
        marker_color=[COLORS[i] for i in range(len(fill_lags))],
        text=[f"{p:.1f}%" for p in fill_pcts],
        textposition="outside",
        textfont=dict(size=9, color=TEXT),
        customdata=fill_vals,
        hovertemplate="%{x}<br>%{customdata:,} valeurs<br>%{y:.1f}%",
    ))
    fig_fill.update_layout(
        **LAYOUT_BASE,
        height=400,
        title=dict(text="Taux de remplissage par lag — temperature ctrl",
                   font=dict(size=13, color=TEXT), x=0.01),
        xaxis=dict(gridcolor=C_GRID, showline=True, linecolor="#30363D",
                   tickangle=-45, title="lag"),
        yaxis=dict(gridcolor=C_GRID, showline=True, linecolor="#30363D",
                   title="%", range=[0, 12]),
    )

    # ── Graphiques par variable : tous les lags superposes ────────────────
    variables = {
        "pred_temperature": ("Temperature ctrl [°C]",   "temperature"),
        "pred_radiation":   ("Radiation ctrl [W/m²]",   "radiation"),
        "pred_humidity":    ("Humidite ctrl [%]",        "humidity"),
        "pred_wind_speed":  ("Vent ctrl [m/s]",         "wind_speed"),
        "pred_pressure":    ("Pression ctrl [hPa]",     "pressure"),
        "pred_sunshine":    ("Ensoleillement ctrl [s]", "sunshine"),
    }

    figs_vars = []
    for var_col, (label, short) in variables.items():
        fig = go.Figure()
        for i, lag in enumerate(range(1, 34)):
            col = f"lag_{lag:02d}h_{var_col}_ctrl"
            if col not in sub.columns:
                continue
            fig.add_trace(go.Scatter(
                x=sub["timestamp_target"],
                y=sub[col],
                mode="markers",
                marker=dict(color=COLORS[i], size=5),
                name=f"lag_{lag:02d}h",
                connectgaps=False,
                hovertemplate=f"lag_{lag:02d}h<br>%{{x}}<br>%{{y:.2f}}",
            ))
        fig.update_layout(
            **LAYOUT_BASE,
            height=450,
            title=dict(
                text=f"{label} — tous les lags | 15-18 juin 2023",
                font=dict(size=13, color=TEXT), x=0.01
            ),
            xaxis=dict(gridcolor=C_GRID, showline=True, linecolor="#30363D",
                       title="timestamp_target"),
            yaxis=dict(gridcolor=C_GRID, showline=True, linecolor="#30363D",
                       title=label),
            legend=dict(
                bgcolor="rgba(0,0,0,0.4)", bordercolor="#30363D", borderwidth=1,
                font=dict(size=9),
                orientation="h",
            ),
        )
        figs_vars.append((f"{label} — tous les lags", fig))

    # ── Graphique incertitude : ctrl/q10/q90 pour quelques lags ──────────
    figs_unc = []
    for lag in [6, 12, 24, 33]:
        fig = go.Figure()

        col_q90  = f"lag_{lag:02d}h_pred_radiation_q90"
        col_q10  = f"lag_{lag:02d}h_pred_radiation_q10"
        col_ctrl = f"lag_{lag:02d}h_pred_radiation_ctrl"
        col_stde = f"lag_{lag:02d}h_pred_radiation_stde"

        if col_ctrl not in sub.columns:
            continue

        fig.add_trace(go.Scatter(
            x=sub["timestamp_target"], y=sub[col_q90],
            fill=None, mode="lines", line=dict(width=0),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=sub["timestamp_target"], y=sub[col_q10],
            fill="tonexty", mode="lines", line=dict(width=0),
            fillcolor="rgba(255,192,0,0.15)",
            name="q10-q90",
        ))
        fig.add_trace(go.Scatter(
            x=sub["timestamp_target"], y=sub[col_ctrl],
            mode="markers",
            marker=dict(color="#FFC000", size=5),
            name="ctrl",
        ))
        fig.add_trace(go.Scatter(
            x=sub["timestamp_target"], y=sub[col_stde],
            mode="markers",
            marker=dict(color="#E74C3C", size=4, symbol="diamond"),
            name="stde",
            yaxis="y2",
        ))
        fig.update_layout(
            **LAYOUT_BASE,
            height=420,
            title=dict(
                text=f"Radiation lag_{lag:02d}h — ctrl/q10/q90/stde | 15-18 juin 2023",
                font=dict(size=13, color=TEXT), x=0.01
            ),
            xaxis=dict(gridcolor=C_GRID, showline=True, linecolor="#30363D"),
            yaxis=dict(gridcolor=C_GRID, showline=True, linecolor="#30363D",
                       title="W/m²"),
            yaxis2=dict(
                title="stde",
                overlaying="y", side="right",
                showgrid=False,
            ),
            legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="#30363D", borderwidth=1),
        )
        figs_unc.append((f"Radiation lag_{lag:02d}h — incertitude ctrl/q10/q90/stde", fig))

    # ── Tableau apercu ────────────────────────────────────────────────────
    sample_cols = (
        ["timestamp_target"] +
        [f"lag_{lag:02d}h_pred_temperature_ctrl" for lag in range(1, 34)]
    )
    sample_cols = [c for c in sample_cols if c in df.columns]

    sample = df.filter(
        pl.col("lag_12h_pred_temperature_ctrl").is_not_null()
    ).filter(
        (pl.col("timestamp_target") >= t_start) &
        (pl.col("timestamp_target") <= t_end)
    ).select(sample_cols).head(20).to_pandas()

    sample["timestamp_target"] = sample["timestamp_target"].dt.strftime(
        "%Y-%m-%d %H:%M UTC"
    )

    col_labels = ["timestamp_target"] + [f"lag_{l:02d}h" for l in range(1, 34)]

    fig_tab = go.Figure(data=[go.Table(
        header=dict(
            values=col_labels,
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
            format=[None] + [".2f"] * 33,
        )
    )])
    fig_tab.update_layout(
        paper_bgcolor=BG_PAPER,
        height=550,
        title=dict(
            text="Tableau temperature ctrl — tous les lags | lignes non nulles",
            font=dict(size=13, color=TEXT), x=0.01
        ),
        margin=dict(l=20, r=20, t=55, b=20),
    )

    # ── Export HTML ───────────────────────────────────────────────────────
    sections = (
        [("Taux de remplissage par lag", fig_fill)] +
        figs_vars +
        figs_unc +
        [("Tableau temperature ctrl — tous les lags", fig_tab)]
    )

    html_parts = [f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="UTF-8">
<title>Verification previsions clean — {STATION}</title>
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
<h1>Verification normalisation previsions — {STATION}</h1>
<div class="subtitle">
  105,120 lignes x 1057 col | lag_01h a lag_33h | 8 variables x 4 sous-types | 15-18 juin 2023
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

    out = DATA_DIR / f"verification_pred_tous_lags_{FILE_NAME}.html"
    out.write_text("".join(html_parts), encoding="utf-8")
    print(f"\n-> {out}")