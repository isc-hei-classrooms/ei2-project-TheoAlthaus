"""
Verification visuelle du resampling 10min -> 15min
====================================================
Compare les valeurs brutes (10min) et les valeurs resamplees (15min)
sur quelques periodes incluant des trous et des zones interpolees.
"""

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from pathlib import Path

from config import DATA_DIR, STATIONS

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

# Periodes de verification : autour de grands trous connus
ZOOM_PERIODS = [
    # (label, t_start, t_end)
    ("Trou court 1.8h — 2023-02-11",
     pl.datetime(2023, 2, 11,  7,  0, 0, time_unit="us", time_zone="UTC"),
     pl.datetime(2023, 2, 11, 12,  0, 0, time_unit="us", time_zone="UTC")),

    ("Trou long 8.7h — 2023-03-08",
     pl.datetime(2023, 3,  8, 20,  0, 0, time_unit="us", time_zone="UTC"),
     pl.datetime(2023, 3,  9,  9,  0, 0, time_unit="us", time_zone="UTC")),

    ("Trou long 19.3h — 2023-09-01",
     pl.datetime(2023, 9,  1, 12,  0, 0, time_unit="us", time_zone="UTC"),
     pl.datetime(2023, 9,  2, 12,  0, 0, time_unit="us", time_zone="UTC")),

    ("Zone sans trou — 2023-06-15",
     pl.datetime(2023, 6, 15,  0,  0, 0, time_unit="us", time_zone="UTC"),
     pl.datetime(2023, 6, 15,  6,  0, 0, time_unit="us", time_zone="UTC")),
]

COLS_TO_CHECK = [
    ("hist_temperature", "Temperature [°C]",   "#E74C3C"),
    ("hist_radiation",   "Radiation [W/m²]",   "#FFC000"),
    ("hist_wind_speed",  "Vent [m/s]",         "#2E75B6"),
]


def verify_station(station: str, file_name: str) -> list[go.Figure]:
    path_raw   = DATA_DIR / f"meteo_{file_name}_hist_raw.parquet"
    path_clean = DATA_DIR / f"meteo_{file_name}_hist_clean.parquet"

    df_raw   = pl.read_parquet(path_raw).sort("timestamp")
    df_clean = pl.read_parquet(path_clean).sort("timestamp")

    figs = []

    for period_label, t_start, t_end in ZOOM_PERIODS:
        sub_raw   = df_raw.filter(
            (pl.col("timestamp") >= t_start) & (pl.col("timestamp") <= t_end)
        )
        sub_clean = df_clean.filter(
            (pl.col("timestamp") >= t_start) & (pl.col("timestamp") <= t_end)
        )

        n_cols = len([c for c, _, _ in COLS_TO_CHECK if c in df_raw.columns])
        if n_cols == 0:
            continue

        fig = make_subplots(
            rows=n_cols, cols=1,
            subplot_titles=[lbl for col, lbl, _ in COLS_TO_CHECK
                            if col in df_raw.columns],
            vertical_spacing=0.12,
        )

        row = 1
        for col, label, color in COLS_TO_CHECK:
            if col not in df_raw.columns:
                continue

            pd_raw   = sub_raw.select(["timestamp", col]).to_pandas()
            pd_clean = sub_clean.select(["timestamp", col]).to_pandas()

            # Valeurs brutes 10min
            fig.add_trace(go.Scatter(
                x=pd_raw["timestamp"], y=pd_raw[col],
                mode="markers+lines",
                marker=dict(color=color, size=6, symbol="circle"),
                line=dict(color=color, width=1, dash="dot"),
                name=f"{label} brut 10min",
                showlegend=(row == 1),
                legendgroup="raw",
            ), row=row, col=1)

            # Valeurs resamplees 15min
            fig.add_trace(go.Scatter(
                x=pd_clean["timestamp"], y=pd_clean[col],
                mode="markers+lines",
                marker=dict(color="#27AE60", size=5, symbol="diamond"),
                line=dict(color="#27AE60", width=1.5),
                name=f"{label} resample 15min",
                showlegend=(row == 1),
                legendgroup="clean",
            ), row=row, col=1)

            fig.update_yaxes(
                title_text=label,
                gridcolor=C_GRID, showline=True, linecolor="#30363D",
                row=row, col=1
            )
            row += 1

        fig.update_layout(
            **LAYOUT_BASE,
            height=300 * n_cols,
            title=dict(
                text=f"{station} — {period_label}",
                font=dict(size=13, color=TEXT), x=0.01
            ),
        )
        fig.update_xaxes(
            gridcolor=C_GRID, showline=True, linecolor="#30363D",
            tickformat="%H:%M\n%d/%m",
        )

        figs.append((f"{station} — {period_label}", fig))

    return figs


def build_html(all_sections: list) -> str:
    parts = [f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="UTF-8">
<title>Verification resampling</title>
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
<h1>Verification du resampling 10min -> 15min</h1>
<div class="subtitle">
  Points orange/rouge = valeurs brutes 10min |
  Points verts = valeurs resamplees 15min |
  Les trous doivent rester NaN (absence de points verts)
</div>
"""]

    for i, (title, fig) in enumerate(all_sections):
        fig_html = pio.to_html(
            fig, include_plotlyjs=(i == 0),
            full_html=False, config={"responsive": True}
        )
        parts.append(f"""
<div class="section">
  <div class="sec-title">{title}</div>
  {fig_html}
</div>""")

    parts.append("</body></html>")
    return "".join(parts)


if __name__ == "__main__":
    all_sections = []

    # Verifier uniquement Evionnaz pour commencer
    for station, file_name in STATIONS.items():
        if file_name != "evionnaz":
            continue
        print(f"Verification {station}...")
        figs = verify_station(station, file_name)
        all_sections.extend(figs)

    html = build_html(all_sections)
    out  = DATA_DIR / "verification_resampling.html"
    out.write_text(html, encoding="utf-8")
    print(f"-> {out}")