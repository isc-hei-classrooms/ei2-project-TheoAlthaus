"""
Analyse des correlations avec le load — Station de Sion uniquement
==================================================================
Genere : data/eda_correlations_sion.html

Dependances : pip install polars plotly pyarrow
Execution   : python eda_correlations_sion.py
"""

from pathlib import Path
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

DATA_DIR = Path("data")
MIN_ABS_R = 0.2   # seuil de pertinence minimum

# ── Palette ───────────────────────────────────────────────────────────────────
BG       = "#0F1117"
BG_PAPER = "#161B22"
TEXT     = "#E6EDF3"
SUBTEXT  = "#8B949E"
C_GRID   = "rgba(200,200,200,0.25)"

LAYOUT_BASE = dict(
    paper_bgcolor=BG_PAPER, plot_bgcolor=BG,
    font=dict(color=TEXT, family="Courier New, monospace", size=12),
    legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="#30363D", borderwidth=1),
    margin=dict(l=60, r=40, t=55, b=60),
)

def apply_base(fig, title, height=420):
    fig.update_layout(**LAYOUT_BASE, height=height,
        title=dict(text=title, font=dict(size=14, color=TEXT), x=0.01))
    fig.update_xaxes(gridcolor=C_GRID, showline=True, linecolor="#30363D")
    fig.update_yaxes(gridcolor=C_GRID, showline=True, linecolor="#30363D")
    return fig

def corr_color(r):
    if r is None or r != r:
        return "rgba(48,54,61,0.8)"
    a = abs(r)
    if r > 0:
        return f"rgba(46,117,182,{0.2 + a*0.8:.2f})"
    else:
        return f"rgba(231,76,60,{0.2 + a*0.8:.2f})"

def short_name(col: str) -> str:
    """Nom lisible depuis le nom de colonne."""
    return (col.replace("hist_", "")
               .replace("_sion", "")
               .replace("_", " ")
               .title())


# ════════════════════════════════════════════════════════════════════════════
# CHARGEMENT
# ════════════════════════════════════════════════════════════════════════════
def load_data() -> pl.DataFrame:
    """Charge Oiken + meteo Sion, joint sur timestamp, nettoie les _right."""

    # Oiken — priorite au fichier clean
    for name in ["oiken_clean.parquet", "oiken.parquet"]:
        p = DATA_DIR / name
        if p.exists():
            df_oiken = pl.read_parquet(p).sort("timestamp")
            print(f"  Oiken  : {p.name}  ({df_oiken.shape[0]:,} lignes)")
            break
    else:
        raise FileNotFoundError("Aucun fichier oiken*.parquet dans data/")

    # Garder load + PV, exclure forecast_load
    oiken_keep = ["timestamp", "load",
                  "pv_central_valais", "pv_sion", "pv_sierre",
                  "pv_remote", "pv_total"]
    oiken_keep = [c for c in oiken_keep if c in df_oiken.columns]
    df = df_oiken.select(oiken_keep)

    # Meteo Sion — priorite au fichier clean
    for pattern in ["meteo_sion_clean.parquet",
                    "meteo_sion_historical.parquet",
                    "meteo_sion.parquet"]:
        p = DATA_DIR / pattern
        if p.exists():
            df_sion = pl.read_parquet(p).sort("timestamp")
            hist_cols = [c for c in df_sion.columns if c.startswith("hist_")]
            if hist_cols:
                print(f"  Meteo  : {p.name}  ({df_sion.shape[0]:,} lignes)")
                print(f"  Cols   : {hist_cols}")
                df_sion = df_sion.select(["timestamp"] + hist_cols)
                break
    else:
        raise FileNotFoundError("Aucun fichier meteo_sion*.parquet dans data/")

    # Jointure
    df = df.join(df_sion, on="timestamp", how="left")

    # Supprimer toutes les colonnes _right (artefacts de jointure)
    right_cols = [c for c in df.columns if c.endswith("_right")]
    if right_cols:
        print(f"  Suppression colonnes _right : {right_cols}")
        df = df.drop(right_cols)

    # Supprimer les colonnes _missing (flags preprocessing, pas des features)
    missing_cols = [c for c in df.columns if c.endswith("_missing")]
    if missing_cols:
        df = df.drop(missing_cols)

    print(f"\n  DataFrame final : {df.shape[0]:,} lignes x {df.shape[1]} colonnes")
    print(f"  Colonnes : {df.columns}")
    return df.sort("timestamp")


# ════════════════════════════════════════════════════════════════════════════
# CHARGEMENT TOUTES STATIONS (pour la cross-correlation inter-stations)
# ════════════════════════════════════════════════════════════════════════════
def load_all_stations() -> dict[str, pl.DataFrame]:
    """
    Charge Oiken + chaque station meteo.
    Retourne {station_name: df_avec_load_et_hist_cols}.
    """
    # Oiken
    for name in ["oiken_clean.parquet", "oiken.parquet"]:
        p = DATA_DIR / name
        if p.exists():
            df_oiken = pl.read_parquet(p).sort("timestamp")
            break
    else:
        raise FileNotFoundError("Aucun fichier oiken*.parquet dans data/")

    load_df = df_oiken.select([c for c in
                               ["timestamp", "load"]
                               if c in df_oiken.columns])

    stations = {}
    meteo_files = sorted([
        p for p in DATA_DIR.glob("meteo_*.parquet")
        if "_predictions" not in p.name
    ])

    for p in meteo_files:
        # Nom de la station
        station = (p.stem
                   .replace("meteo_", "")
                   .replace("_clean", "")
                   .replace("_historical", "")
                   .replace("_", " ").title())

        df_m = pl.read_parquet(p).sort("timestamp")
        hist_cols = [c for c in df_m.columns if c.startswith("hist_")
                     and not c.endswith("_missing")]
        if not hist_cols:
            continue

        # Jointure avec le load
        df_joined = load_df.join(
            df_m.select(["timestamp"] + hist_cols),
            on="timestamp", how="left"
        )
        # Nettoyer les _right
        right_cols = [c for c in df_joined.columns if c.endswith("_right")]
        if right_cols:
            df_joined = df_joined.drop(right_cols)

        stations[station] = df_joined
        print(f"  {station:<25} {len(hist_cols)} variables historiques")

    return stations


# ════════════════════════════════════════════════════════════════════════════
# CROSS-CORRELATION TOUTES STATIONS
# ════════════════════════════════════════════════════════════════════════════
def fig_lagged_all_stations(stations: dict[str, pl.DataFrame],
                             variable: str = "hist_temperature",
                             max_lag_h: int = 24) -> go.Figure:
    """
    Pour une variable donnee, trace r(load[t], variable[t-lag]) pour chaque
    station sur lag 0 -> max_lag_h.
    Permet de voir si une station presente un pic de correlation a un lag
    different de Sion → effet corridor / decalage geographique.
    """
    STEP   = 4   # 4 x 15min = 1h
    lags_h = list(range(0, max_lag_h + 1))
    colors_list = ["#2E75B6","#ED7D31","#FFC000","#27AE60",
                   "#E74C3C","#9B59B6","#1ABC9C","#E67E22"]

    fig = go.Figure()
    found_any = False

    for i, (station, df) in enumerate(stations.items()):
        # Chercher la colonne de la variable (peu importe le suffixe)
        col = next((c for c in df.columns if c.startswith(variable)), None)
        if col is None:
            continue
        found_any = True

        corrs = []
        for lag_h in lags_h:
            try:
                # On decale le LOAD vers le futur (shift negatif) :
                # r( load[t + lag],  meteo_station[t] )
                # "La meteo de cette station predit-elle le load X heures plus tard ?"
                # Un pic a lag=3h signifie : cette station a 3h d'avance sur le load de Sion
                r = df.select([
                    pl.col("load").shift(-lag_h * STEP).alias("load_future"),
                    pl.col(col),
                ]).select(pl.corr("load_future", col)).item()
                corrs.append(round(r, 4) if r is not None and r == r else 0)
            except Exception:
                corrs.append(0)

        # Trouver le lag du pic
        peak_lag = lags_h[max(range(len(corrs)), key=lambda i: abs(corrs[i]))]
        peak_r   = corrs[peak_lag]

        c = colors_list[i % len(colors_list)]
        fig.add_trace(go.Scatter(
            x=lags_h, y=corrs,
            mode="lines+markers",
            line=dict(color=c, width=2),
            marker=dict(size=5),
            name=f"{station}  (pic lag={peak_lag}h, r={peak_r:+.3f})",
        ))
        # Marquer le pic
        fig.add_trace(go.Scatter(
            x=[peak_lag], y=[peak_r],
            mode="markers",
            marker=dict(color=c, size=10, symbol="star",
                        line=dict(color="white", width=1)),
            showlegend=False,
            hovertemplate=f"{station}<br>lag={peak_lag}h<br>r={peak_r:.3f}",
        ))

    if not found_any:
        fig.add_annotation(text=f"Variable '{variable}' absente de toutes les stations",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color=TEXT, size=14))

    var_label = variable.replace("hist_", "").replace("_", " ").title()
    apply_base(fig,
        f"Cross-correlation inter-stations — {var_label}  |  "
        f"r( load[t+lag], meteo_station[t] )  |  "
        f"Pic a lag=Xh → cette station a Xh d'avance sur le load  |  ⭐ = pic",
        height=480)
    fig.add_hline(y=0,    line_color=TEXT,      line_width=1, line_dash="dot")
    fig.add_hline(y=0.3,  line_color="#27AE60", line_width=1, line_dash="dot",
                  annotation_text="+0.3", annotation_font_color="#27AE60",
                  annotation_position="right")
    fig.add_hline(y=-0.3, line_color="#27AE60", line_width=1, line_dash="dot",
                  annotation_text="-0.3", annotation_font_color="#27AE60",
                  annotation_position="right")
    fig.update_layout(
        xaxis=dict(title="Decalage [heures]", tickmode="linear", dtick=2,
                   gridcolor=C_GRID, showline=True, linecolor="#30363D"),
        yaxis_title="r de Pearson",
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════
# CALCUL DES CORRELATIONS GLOBALES
# ════════════════════════════════════════════════════════════════════════════
def compute_global_corr(df: pl.DataFrame) -> list[tuple]:
    """Retourne [(col, r)] tries par |r| decroissant, |r| >= MIN_ABS_R."""
    results = []
    for col in df.columns:
        if col in ("timestamp", "load"):
            continue
        try:
            r = df.select(pl.corr("load", col)).item()
            if r is not None and r == r and abs(r) >= MIN_ABS_R:
                results.append((col, round(r, 4)))
        except Exception:
            pass
    results.sort(key=lambda x: abs(x[1]), reverse=True)
    return results


# ════════════════════════════════════════════════════════════════════════════
# 1. CORRELATIONS GLOBALES
# ════════════════════════════════════════════════════════════════════════════
def fig_global(results: list[tuple]) -> go.Figure:
    cols  = [short_name(c) for c, _ in results]
    corrs = [r for _, r in results]
    colors = [corr_color(r) for r in corrs]

    fig = go.Figure(go.Bar(
        x=corrs, y=cols,
        orientation="h",
        marker_color=colors,
        marker_line=dict(width=0),
        text=[f"{r:+.3f}" for r in corrs],
        textposition="outside",
        textfont=dict(size=11, color=TEXT),
    ))
    apply_base(fig, f"Correlations globales avec le load  (|r| >= {MIN_ABS_R})",
               height=max(360, len(results) * 30 + 80))
    fig.add_vline(x=0,    line_color=TEXT,      line_width=1)
    fig.add_vline(x=0.5,  line_color="#27AE60", line_dash="dot", line_width=1,
                  annotation_text="0.5", annotation_font_color="#27AE60",
                  annotation_position="top")
    fig.add_vline(x=-0.5, line_color="#27AE60", line_dash="dot", line_width=1,
                  annotation_text="-0.5", annotation_font_color="#27AE60",
                  annotation_position="top")
    fig.update_layout(xaxis=dict(range=[-1.05, 1.05], title="r de Pearson",
                                 gridcolor=C_GRID, showline=True, linecolor="#30363D"))
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=11))
    return fig


# ════════════════════════════════════════════════════════════════════════════
# 2. HEATMAP HORAIRE
# ════════════════════════════════════════════════════════════════════════════
def fig_hourly_heatmap(df: pl.DataFrame, results: list[tuple]) -> go.Figure:
    features = [c for c, _ in results]
    df_h = df.with_columns(pl.col("timestamp").dt.hour().alias("_hour"))
    hours = list(range(24))

    matrix = []
    for col in features:
        row = []
        for h in hours:
            sub = df_h.filter(pl.col("_hour") == h)
            try:
                r = sub.select(pl.corr("load", col)).item()
                row.append(round(r, 3) if r is not None and r == r else 0)
            except Exception:
                row.append(0)
        matrix.append(row)

    y_labels = [short_name(c) for c in features]

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=[f"{h:02d}h" for h in hours],
        y=y_labels,
        colorscale=[[0, "#E74C3C"], [0.5, "#161B22"], [1, "#2E75B6"]],
        zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in matrix],
        texttemplate="%{text}",
        textfont=dict(size=9),
        colorbar=dict(title="r", thickness=14, len=0.9,
                      tickvals=[-1, -0.5, 0, 0.5, 1]),
    ))
    apply_base(fig, "Correlations horaires — r par heure de la journee",
               height=max(350, len(features) * 32 + 100))
    fig.update_xaxes(tickangle=0)
    fig.update_yaxes(tickfont=dict(size=11))
    return fig


# ════════════════════════════════════════════════════════════════════════════
# 3. CROSS-CORRELATION (lag 0-24h)
# ════════════════════════════════════════════════════════════════════════════
def fig_lagged(df: pl.DataFrame, results: list[tuple]) -> go.Figure:
    # Toutes les features pertinentes
    features = [c for c, _ in results]
    STEP = 4   # 4 x 15min = 1h
    lags_h = list(range(0, 25))

    colors_list = ["#2E75B6","#ED7D31","#FFC000","#27AE60",
                   "#E74C3C","#9B59B6","#1ABC9C","#E67E22",
                   "#5B9BD5","#70AD47","#FF6B6B","#A29BFE"]

    fig = go.Figure()
    for i, col in enumerate(features):
        corrs = []
        for lag_h in lags_h:
            try:
                # Formule feature engineering : r( load[t], meteo[t-lag] )
                # "La meteo passee de Sion predit-elle le load present ?"
                # Decale la feature vers le passe (shift positif)
                r = df.select([
                    pl.col("load"),
                    pl.col(col).shift(lag_h * STEP).alias("lagged")
                ]).select(pl.corr("load", "lagged")).item()
                corrs.append(round(r, 4) if r is not None and r == r else 0)
            except Exception:
                corrs.append(0)

        fig.add_trace(go.Scatter(
            x=lags_h, y=corrs,
            mode="lines+markers",
            line=dict(color=colors_list[i % len(colors_list)], width=1.8),
            marker=dict(size=5),
            name=short_name(col),
        ))

    apply_base(fig, "Cross-correlation : r(load[t], feature[t - lag])  lag 0→24h",
               height=460)
    fig.add_hline(y=0, line_color=TEXT, line_width=1, line_dash="dot")
    fig.add_hline(y=0.3,  line_color="#27AE60", line_width=1, line_dash="dot",
                  annotation_text="+0.3", annotation_font_color="#27AE60",
                  annotation_position="right")
    fig.add_hline(y=-0.3, line_color="#27AE60", line_width=1, line_dash="dot",
                  annotation_text="-0.3", annotation_font_color="#27AE60",
                  annotation_position="right")
    fig.update_layout(
        xaxis=dict(title="Decalage [heures]", tickmode="linear", dtick=2,
                   gridcolor=C_GRID, showline=True, linecolor="#30363D"),
        yaxis_title="r de Pearson",
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════
# 4. CORRELATIONS PAR SAISON (version compacte)
# ════════════════════════════════════════════════════════════════════════════
def fig_seasonal(df: pl.DataFrame, results: list[tuple]) -> go.Figure:
    """
    Plus-value reelle : la relation temperature/load s'inverse entre ete et hiver
    (chauffage vs climatisation), et la correlation PV/load change aussi.
    """
    seasons = {
        "Hiver (DJF)":     [12, 1, 2],
        "Printemps (MAM)": [3,  4, 5],
        "Ete (JJA)":       [6,  7, 8],
        "Automne (SON)":   [9, 10, 11],
    }
    season_colors = ["#5B9BD5","#70AD47","#FFC000","#ED7D31"]
    features = [c for c, _ in results]
    x_labels = [short_name(c) for c in features]

    df_s = df.with_columns(pl.col("timestamp").dt.month().alias("_month"))

    fig = go.Figure()
    for (season_name, months), color in zip(seasons.items(), season_colors):
        sub = df_s.filter(pl.col("_month").is_in(months))
        corrs = []
        for col in features:
            try:
                r = sub.select(pl.corr("load", col)).item()
                corrs.append(round(r, 3) if r is not None and r == r else 0)
            except Exception:
                corrs.append(0)
        fig.add_trace(go.Bar(
            x=x_labels, y=corrs,
            name=season_name,
            marker_color=color, opacity=0.85,
        ))

    apply_base(fig, "Correlations par saison — la relation change-t-elle selon la periode ?",
               height=420)
    fig.update_layout(
        barmode="group",
        xaxis_tickangle=-35,
        yaxis=dict(title="r de Pearson", range=[-1.05, 1.05],
                   gridcolor=C_GRID, showline=True, linecolor="#30363D"),
    )
    fig.add_hline(y=0,    line_color=TEXT,      line_width=1)
    fig.add_hline(y=0.3,  line_color="#27AE60", line_dash="dot", line_width=1)
    fig.add_hline(y=-0.3, line_color="#27AE60", line_dash="dot", line_width=1)
    return fig



# ════════════════════════════════════════════════════════════════════════════
# CROSS-CORRELATION LOAD vs LOAD (autocorrelation)
# ════════════════════════════════════════════════════════════════════════════
def fig_load_autocorr(df: pl.DataFrame, max_lag_h: int = 168) -> go.Figure:
    """
    r( load[t], load[t-lag] ) pour lag = 0 -> 168h (1 semaine).
    Formule feature engineering : le load passe predit-il le load present ?
    Permet de valider quels lags valent la peine d'etre inclus comme features.

    Interpretation :
      Pic a lag=24h  -> load J-1 meme heure tres correlee -> feature load_lag_96
      Pic a lag=168h -> load J-7 meme heure correlee      -> feature load_lag_672
      Pic a lag=48h  -> load J-2 correlee                 -> feature optionnelle
    """
    STEP   = 4   # 4 x 15min = 1h
    lags_h = list(range(0, max_lag_h + 1))

    corrs = []
    for lag_h in lags_h:
        try:
            r = df.select([
                pl.col("load"),
                pl.col("load").shift(lag_h * STEP).alias("load_lagged")
            ]).select(pl.corr("load", "load_lagged")).item()
            corrs.append(round(r, 4) if r is not None and r == r else 0)
        except Exception:
            corrs.append(0)

    # Identifier les pics locaux (lag multiples de 24h)
    peaks_of_interest = [24, 48, 72, 96, 120, 144, 168]

    fig = go.Figure()

    # Courbe principale
    fig.add_trace(go.Scatter(
        x=lags_h, y=corrs,
        mode="lines",
        line=dict(color="#2E75B6", width=1.8),
        fill="tozeroy",
        fillcolor="rgba(46,117,182,0.08)",
        name="autocorrelation load",
    ))

    # Marquer les pics aux multiples de 24h
    for peak_h in peaks_of_interest:
        if peak_h <= max_lag_h:
            r_peak = corrs[peak_h]
            color  = "#FFC000" if abs(r_peak) >= 0.5 else                      "#F39C12" if abs(r_peak) >= 0.3 else "#8B949E"
            fig.add_trace(go.Scatter(
                x=[peak_h], y=[r_peak],
                mode="markers+text",
                marker=dict(color=color, size=10, symbol="diamond",
                            line=dict(color="white", width=1)),
                text=[f"J-{peak_h//24}\nr={r_peak:.2f}"],
                textposition="top center",
                textfont=dict(size=9, color=color),
                name=f"lag={peak_h}h (J-{peak_h//24})",
                showlegend=True,
            ))

    # Lignes de reference
    fig.add_hline(y=0,    line_color=TEXT,      line_width=1, line_dash="dot")
    fig.add_hline(y=0.5,  line_color="#27AE60", line_width=1, line_dash="dot",
                  annotation_text="seuil 0.5", annotation_font_color="#27AE60",
                  annotation_position="right")
    fig.add_hline(y=0.3,  line_color="#F39C12", line_width=1, line_dash="dot",
                  annotation_text="seuil 0.3", annotation_font_color="#F39C12",
                  annotation_position="right")

    # Lignes verticales aux jours
    for d in range(1, max_lag_h // 24 + 1):
        fig.add_vline(x=d * 24, line_color="rgba(200,200,200,0.15)",
                      line_width=1, line_dash="dot",
                      annotation_text=f"J-{d}" if d <= 7 else "",
                      annotation_font_color=SUBTEXT,
                      annotation_font_size=9,
                      annotation_position="top")

    apply_base(fig,
        "Autocorrelation du load — r(load[t], load[t-lag])  |  lag 0→168h (1 semaine)  |"
        "  ◆ = multiples de 24h  |  Pic fort -> ce lag vaut la peine d'etre une feature",
        height=460)
    fig.update_layout(
        xaxis=dict(title="Decalage [heures]", tickmode="linear", dtick=12,
                   gridcolor=C_GRID, showline=True, linecolor="#30363D"),
        yaxis=dict(title="r de Pearson", range=[-0.1, 1.05],
                   gridcolor=C_GRID, showline=True, linecolor="#30363D"),
    )
    return fig



# ════════════════════════════════════════════════════════════════════════════
# BOXPLOTS CALENDAIRES
# ════════════════════════════════════════════════════════════════════════════
def fig_calendar_boxplots(df_main: pl.DataFrame,
                           df_cal: pl.DataFrame) -> go.Figure:
    """
    Boxplots du load selon les variables calendaires :
      - Jour de la semaine (0=lundi ... 6=dimanche)
      - Heure de la journee (profil moyen)
      - Jour ferie vs normal
      - Vacances scolaires vs normal
      - Weekend vs semaine
    Utilise des violin plots pour voir la distribution complete.
    """
    # Joindre calendrier sur le timestamp (15min -> 15min)
    df = df_main.select(["timestamp", "load"]).join(
        df_cal.select([
            "timestamp", "day_of_week", "hour",
            "is_holiday", "is_school_holiday", "is_weekend", "month"
        ]),
        on="timestamp", how="left"
    )
    # Nettoyer _right
    df = df.drop([c for c in df.columns if c.endswith("_right")])

    DAY_NAMES = ["Lundi","Mardi","Mercredi","Jeudi",
                 "Vendredi","Samedi","Dimanche"]
    MONTH_NAMES = ["Jan","Fev","Mar","Avr","Mai","Jun",
                   "Jul","Aou","Sep","Oct","Nov","Dec"]
    COLORS_DOW = ["#2E75B6","#3A8CC0","#4AA9CA","#5BC4D4",
                  "#FFC000","#ED7D31","#E74C3C"]
    COLORS_MONTH = ["#1E4D8C","#2563AE","#2E75B6","#3A8CC0",
                    "#70AD47","#9DC45A","#FFC000","#F39C12",
                    "#ED7D31","#E74C3C","#C0392B","#1E4D8C"]

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "Load par jour de la semaine",
            "Load par mois",
            "Ferie vs Normal  |  Vacances vs Normal",
            "Load par heure (violin)",
            "Weekend vs Semaine",
            "Load : ferie + vacances combinees",
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.08,
    )

    # ── 1. Jour de la semaine ─────────────────────────────────────────────
    for dow in range(7):
        vals = df.filter(pl.col("day_of_week") == dow)["load"].drop_nulls()
        if vals.is_empty():
            continue
        fig.add_trace(go.Violin(
            y=vals.to_list(), name=DAY_NAMES[dow],
            box_visible=True, meanline_visible=True,
            marker_color=COLORS_DOW[dow], showlegend=False,
            x0=DAY_NAMES[dow],
        ), row=1, col=1)

    # ── 2. Mois ───────────────────────────────────────────────────────────
    for m in range(1, 13):
        vals = df.filter(pl.col("month") == m)["load"].drop_nulls()
        if vals.is_empty():
            continue
        fig.add_trace(go.Box(
            y=vals.sample(min(2000, len(vals)), seed=42).to_list(),
            name=MONTH_NAMES[m-1],
            marker_color=COLORS_MONTH[m-1],
            boxmean=True, showlegend=False,
            x0=MONTH_NAMES[m-1],
        ), row=1, col=2)

    # ── 3. Ferie / Vacances ───────────────────────────────────────────────
    groups = {
        "Normal":         df.filter((pl.col("is_holiday")==0) &
                                    (pl.col("is_school_holiday")==0)),
        "Ferie":          df.filter(pl.col("is_holiday")==1),
        "Vacances scol.": df.filter((pl.col("is_school_holiday")==1) &
                                    (pl.col("is_holiday")==0)),
    }
    colors_g = {"Normal": "#2E75B6", "Ferie": "#E74C3C",
                 "Vacances scol.": "#FFC000"}
    for gname, gdf in groups.items():
        vals = gdf["load"].drop_nulls()
        if vals.is_empty():
            continue
        fig.add_trace(go.Violin(
            y=vals.sample(min(5000, len(vals)), seed=42).to_list(),
            name=gname, x0=gname,
            box_visible=True, meanline_visible=True,
            marker_color=colors_g[gname], showlegend=False,
        ), row=1, col=3)

    # ── 4. Profil horaire (violin par heure) ─────────────────────────────
    for h in range(0, 24, 2):  # 1 violin toutes les 2h pour lisibilite
        vals = df.filter(pl.col("hour") == h)["load"].drop_nulls()
        if vals.is_empty():
            continue
        fig.add_trace(go.Violin(
            y=vals.sample(min(1000, len(vals)), seed=42).to_list(),
            name=f"{h:02d}h", x0=f"{h:02d}h",
            box_visible=False, meanline_visible=True,
            marker_color="#2E75B6", opacity=0.6, showlegend=False,
        ), row=2, col=1)

    # ── 5. Weekend vs Semaine ─────────────────────────────────────────────
    for is_we, label, color in [(0,"Semaine","#2E75B6"),(1,"Weekend","#ED7D31")]:
        vals = df.filter(pl.col("is_weekend") == is_we)["load"].drop_nulls()
        if vals.is_empty():
            continue
        fig.add_trace(go.Violin(
            y=vals.sample(min(8000, len(vals)), seed=42).to_list(),
            name=label, x0=label,
            box_visible=True, meanline_visible=True,
            marker_color=color, showlegend=False,
        ), row=2, col=2)

    # ── 6. Combinaisons ferie + vacances ──────────────────────────────────
    combos = {
        "Normal":           df.filter((pl.col("is_holiday")==0) &
                                      (pl.col("is_school_holiday")==0) &
                                      (pl.col("is_weekend")==0)),
        "Weekend":          df.filter((pl.col("is_weekend")==1) &
                                      (pl.col("is_holiday")==0)),
        "Vacances":         df.filter((pl.col("is_school_holiday")==1) &
                                      (pl.col("is_holiday")==0) &
                                      (pl.col("is_weekend")==0)),
        "Ferie":            df.filter(pl.col("is_holiday")==1),
        "Vac.+Weekend":     df.filter((pl.col("is_school_holiday")==1) &
                                      (pl.col("is_weekend")==1)),
    }
    combo_colors = {"Normal":"#2E75B6","Weekend":"#ED7D31",
                    "Vacances":"#FFC000","Ferie":"#E74C3C",
                    "Vac.+Weekend":"#9B59B6"}
    for cname, cdf in combos.items():
        vals = cdf["load"].drop_nulls()
        if vals.is_empty():
            continue
        fig.add_trace(go.Box(
            y=vals.sample(min(3000, len(vals)), seed=42).to_list(),
            name=cname, x0=cname,
            marker_color=combo_colors[cname],
            boxmean=True, showlegend=False,
        ), row=2, col=3)

    fig.update_layout(
        **LAYOUT_BASE, height=700,
        title=dict(
            text="Variables calendaires vs Load — boxplots et violin plots",
            font=dict(size=14, color=TEXT), x=0.01
        ),
    )
    fig.update_yaxes(gridcolor=C_GRID, showline=True, linecolor="#30363D")
    fig.update_xaxes(showline=True, linecolor="#30363D", tickangle=-20)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# MATRICE DE CORRELATION ENTRE FEATURES (multicollinearite)
# ════════════════════════════════════════════════════════════════════════════
def fig_feature_corr_matrix(df: pl.DataFrame,
                              df_cal: pl.DataFrame) -> go.Figure:
    """
    Matrice de correlation entre toutes les features (hors load).
    Permet de detecter la multicollinearite — features trop similaires
    qui n'apportent pas d'information supplementaire au modele.
    """
    # Joindre les features calendaires sin/cos
    cal_cols = ["timestamp", "hour_sin", "hour_cos",
                "day_of_year_sin", "day_of_year_cos",
                "month_sin", "month_cos",
                "day_of_week_sin", "day_of_week_cos",
                "is_weekend", "is_holiday", "is_school_holiday"]
    cal_cols = [c for c in cal_cols if c in df_cal.columns]

    df_full = df.join(df_cal.select(cal_cols), on="timestamp", how="left")
    df_full = df_full.drop([c for c in df_full.columns if c.endswith("_right")])

    # Toutes les features numeriques sauf load et timestamp
    feature_cols = [
        c for c in df_full.columns
        if c not in ("timestamp", "load")
        and not c.endswith("_missing")
        and df_full[c].dtype in (pl.Float64, pl.Float32,
                                  pl.Int64, pl.Int32, pl.Int8)
    ]

    n = len(feature_cols)
    labels = [short_name(c) for c in feature_cols]

    # Calculer la matrice n x n
    matrix = []
    for c1 in feature_cols:
        row = []
        for c2 in feature_cols:
            try:
                r = df_full.select(pl.corr(c1, c2)).item()
                row.append(round(r, 3) if r is not None and r == r else 0)
            except Exception:
                row.append(0)
        matrix.append(row)

    # Detecter les paires tres correlees (|r| > 0.85, hors diagonale)
    high_corr_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if abs(matrix[i][j]) > 0.85:
                high_corr_pairs.append(
                    (labels[i], labels[j], matrix[i][j])
                )

    print(f"  Paires tres correlees (|r|>0.85) : {len(high_corr_pairs)}")
    for a, b, r in high_corr_pairs:
        print(f"    {a:<30} <-> {b:<30} r={r:+.3f}")

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=labels, y=labels,
        colorscale=[[0,"#E74C3C"],[0.5,"#161B22"],[1,"#2E75B6"]],
        zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:.2f}" if abs(v) > 0.1 else "" for v in row]
              for row in matrix],
        texttemplate="%{text}",
        textfont=dict(size=8),
        colorbar=dict(title="r", thickness=14, len=0.9,
                      tickvals=[-1,-0.5,0,0.5,1]),
        hoverongaps=False,
    ))

    apply_base(fig,
        f"Matrice de correlation entre features — multicollinearite  |  "
        f"{len(high_corr_pairs)} paires |r|>0.85 detectees (voir console)",
        height=max(500, n * 28 + 120))
    fig.update_xaxes(tickangle=-40, tickfont=dict(size=9))
    fig.update_yaxes(tickfont=dict(size=9), autorange="reversed")
    return fig


# ════════════════════════════════════════════════════════════════════════════
# TABLE DE SYNTHESE
# ════════════════════════════════════════════════════════════════════════════
def build_summary_table(results: list[tuple]) -> str:
    rows = [
        "<table style='width:100%;border-collapse:collapse;"
        "font-family:Courier New;font-size:12px;color:#E6EDF3'>",
        "<tr style='background:#21262D;color:#FFC000'>"
        "<th style='padding:7px 14px;text-align:left'>Feature</th>"
        "<th style='padding:7px 14px'>r global</th>"
        "<th style='padding:7px 14px'>|r|</th>"
        "<th style='padding:7px 14px'>Sens</th>"
        "<th style='padding:7px 14px'>Pertinence ML</th>"
        "</tr>"
    ]
    for i, (col, r) in enumerate(results):
        bg = "#161B22" if i % 2 == 0 else "#0F1117"
        ar = abs(r)
        if ar >= 0.6:
            label, pc = "⭐⭐⭐ Tres forte", "#27AE60"
        elif ar >= 0.4:
            label, pc = "⭐⭐ Forte",       "#70AD47"
        elif ar >= 0.2:
            label, pc = "⭐ Moderee",       "#F39C12"
        else:
            label, pc = "✗ Negligeable",   "#E74C3C"
        rows.append(
            f"<tr style='background:{bg}'>"
            f"<td style='padding:6px 14px'>{short_name(col)}</td>"
            f"<td style='padding:6px 14px;text-align:center;"
            f"color:{'#2E75B6' if r>0 else '#E74C3C'}'>{r:+.4f}</td>"
            f"<td style='padding:6px 14px;text-align:center'>{ar:.4f}</td>"
            f"<td style='padding:6px 14px;text-align:center'>"
            f"{'positif' if r>0 else 'negatif'}</td>"
            f"<td style='padding:6px 14px;text-align:center;"
            f"color:{pc};font-weight:bold'>{label}</td>"
            f"</tr>"
        )
    rows.append("</table>")
    return "\n".join(rows)


# ════════════════════════════════════════════════════════════════════════════
# ASSEMBLAGE HTML
# ════════════════════════════════════════════════════════════════════════════
def build_html(sections: list, summary_table: str) -> str:
    parts = [f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="UTF-8">
<title>Correlations Load — Sion</title>
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
  .content {{ padding:16px; overflow-x:auto; }}
</style></head><body>
<h1>Correlations avec le Load — Station de Sion</h1>
<div class="subtitle">
  Features Oiken (PV) + MeteoSuisse Sion historique  |  |r| >= {MIN_ABS_R}  |  forecast exclu
</div>
"""]

    parts.append(f"""
<div class="section">
  <div class="sec-title">Synthese — features pertinentes triees par |r|</div>
  <div class="content">{summary_table}</div>
</div>""")

    for i, (title, fig) in enumerate(sections):
        fig_html = pio.to_html(fig, include_plotlyjs=(i == 0),
                               full_html=False, config={"responsive": True})
        parts.append(f"""
<div class="section">
  <div class="sec-title">{title}</div>
  {fig_html}
</div>""")

    parts.append("</body></html>")
    return "".join(parts)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ── Donnees Sion ──────────────────────────────────────────────────────
    print("Chargement des donnees Sion...")
    df = load_data()

    # ── Calendrier ────────────────────────────────────────────────────────
    df_cal = None
    for cal_name in ["calendar.parquet"]:
        p = DATA_DIR / cal_name
        if p.exists():
            df_cal = pl.read_parquet(p).sort("timestamp")
            print(f"  Calendrier : {p.name}  ({df_cal.shape[0]:,} lignes)")
            break
    if df_cal is None:
        print("  ⚠ calendar.parquet introuvable — boxplots et matrice de corr. sautes")

    print("\nCalcul des correlations globales...")
    results = compute_global_corr(df)
    print(f"  {len(results)} features avec |r| >= {MIN_ABS_R} :")
    for col, r in results:
        print(f"    {short_name(col):<30} r = {r:+.4f}")

    if not results:
        print("Aucune correlation suffisante trouvee.")
        exit(0)

    # ── Toutes les stations (pour cross-corr inter-stations) ──────────────
    print("\nChargement de toutes les stations...")
    all_stations = load_all_stations()

    # Variables disponibles dans toutes/la plupart des stations
    all_hist_vars = set()
    for df_s in all_stations.values():
        for c in df_s.columns:
            if c.startswith("hist_") and not c.endswith("_missing"):
                # Extraire le nom de base sans suffixe station
                base = "_".join(c.split("_")[:3])  # ex: hist_temperature
                all_hist_vars.add(base)

    print(f"  Variables historiques detectees : {sorted(all_hist_vars)}")

    # ── Generation des graphiques ─────────────────────────────────────────
    print("\nGeneration des graphiques...")

    inter_station_figs = []
    for var in sorted(all_hist_vars):
        var_label = var.replace("hist_", "").replace("_", " ").title()
        fig = fig_lagged_all_stations(all_stations, variable=var, max_lag_h=24)
        inter_station_figs.append(
            (f"Cross-correlation inter-stations — {var_label}", fig)
        )

    print("Calcul de l'autocorrelation du load (lag 0-168h)...")
    fig_autocorr = fig_load_autocorr(df, max_lag_h=168)

    # ── Figures dependantes du calendrier ─────────────────────────────────
    cal_sections = []
    if df_cal is not None:
        print("Generation des boxplots calendaires...")
        cal_sections.append((
            "Boxplots calendaires — load par jour/mois/ferie/vacances/weekend",
            fig_calendar_boxplots(df, df_cal)
        ))
        print("Generation de la matrice de correlation entre features...")
        cal_sections.append((
            "Matrice de correlation entre features — multicollinearite",
            fig_feature_corr_matrix(df, df_cal)
        ))

    sections = [
        ("1. Correlations globales — Sion",
            fig_global(results)),
        ("2. Heatmap horaire — r varie-t-il selon l'heure ? (Sion)",
            fig_hourly_heatmap(df, results)),
        ("3. Autocorrelation du load — r(load[t], load[t-lag])  lag 0-168h (1 semaine)",
            fig_autocorr),
        ("4. Feature lags meteo Sion — r(load[t], meteo_sion[t-lag]) "
         "| La meteo PASSEE de Sion predit-elle le load PRESENT ?",
            fig_lagged(df, results)),
        ("5. Correlations par saison — Sion",
            fig_seasonal(df, results)),
    ] + cal_sections + inter_station_figs
    # Sections 5+ = inter-stations : r(load[t+lag], meteo_station[t])
    # "La meteo ACTUELLE d'une station predit-elle le load FUTUR de Sion ?"
    # Pic a lag=Xh => cette station a Xh d'avance sur le load

    summary = build_summary_table(results)
    html = build_html(sections, summary)

    out = DATA_DIR / "eda_correlations_sion.html"
    out.write_text(html, encoding="utf-8")
    print(f"\n✅ {out}  ({out.stat().st_size/1024:.0f} KB)")
    print("Ouvrir dans un navigateur.")