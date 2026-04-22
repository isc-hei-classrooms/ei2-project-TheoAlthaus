"""
Visualisation interactive des predictions v4
=============================================
Lit  : data/models/wf_predictions_v4.parquet
       data/oiken_clean.parquet
Ecrit: data/visualisation_v4.html
"""

import polars as pl
import plotly.graph_objects as go
import plotly.io as pio
import json
from pathlib import Path

from config import DATA_DIR, FILE_OIKEN_CLEAN

TZ_LOCAL = "Europe/Zurich"
FILE_OUT = DATA_DIR / "visualisation_v4.html"

BG       = "#0F1117"
BG_PAPER = "#161B22"
TEXT     = "#E6EDF3"
SUBTEXT  = "#8B949E"
C_GRID   = "rgba(200,200,200,0.15)"

if __name__ == "__main__":

    # ── Chargement ────────────────────────────────────────────────────────
    print("Chargement des donnees...")
    df_pred = pl.read_parquet(DATA_DIR / "models" / "wf_predictions_v4.parquet")
    oiken   = pl.read_parquet(FILE_OIKEN_CLEAN).sort("timestamp")

    # Ajouter heure locale
    df_pred = df_pred.with_columns(
        pl.col("timestamp")
          .dt.convert_time_zone(TZ_LOCAL)
          .alias("ts_local")
    ).with_columns(
        pl.col("ts_local").dt.date().alias("date_locale"),
        pl.col("ts_local").dt.strftime("%Y-%m-%d").alias("date_str"),
    )

    # ── Calculer MAE par jour et par modele ───────────────────────────────
    print("Calcul MAE par jour...")

    models_list = df_pred["model"].unique().sort().to_list()

    # MAE oiken par jour (depuis oiken_clean)
    oiken_mae = (
        df_pred.filter(pl.col("model") == models_list[0])
        .with_columns(
            (pl.col("true") - pl.col("forecast")).abs().alias("err_oiken")
        )
        .group_by("date_str")
        .agg([
            pl.col("err_oiken").mean().alias("mae_oiken"),
            pl.col("split").first(),
            pl.col("date_locale").first(),
        ])
        .sort("date_str")
    )

    # MAE par modele par jour
    mae_per_model = {}
    for model_name in models_list:
        sub = df_pred.filter(pl.col("model") == model_name)
        mae_day = (
            sub.with_columns(
                (pl.col("true") - pl.col("pred")).abs().alias("err")
            )
            .group_by("date_str")
            .agg(pl.col("err").mean().alias("mae"))
            .sort("date_str")
        )
        mae_per_model[model_name] = dict(zip(
            mae_day["date_str"].to_list(),
            mae_day["mae"].to_list()
        ))

    oiken_mae_dict = dict(zip(
        oiken_mae["date_str"].to_list(),
        oiken_mae["mae_oiken"].to_list()
    ))

    # ── Preparer les donnees par jour pour le graphique ───────────────────
    # Prendre le meilleur modele (xgboost) pour la visualisation principale
    best_model = "xgboost"
    df_best = df_pred.filter(pl.col("model") == best_model).sort("ts_local")

    # Construire les series de donnees
    all_dates = sorted(df_best["date_str"].unique().to_list())

    print(f"  Nombre de jours : {len(all_dates)}")
    print(f"  Modeles : {models_list}")

    # Donnees par jour : {date: {ts, true, pred, forecast}}
    day_data = {}
    for date_str in all_dates:
        sub = df_best.filter(pl.col("date_str") == date_str).sort("ts_local")
        day_data[date_str] = {
            "ts":       [str(t) for t in sub["ts_local"].to_list()],
            "true":     sub["true"].to_list(),
            "pred":     sub["pred"].to_list(),
            "forecast": sub["forecast"].to_list(),
            "split":    sub["split"][0],
        }

    # Donnees multi-modeles par jour
    multimodel_data = {}
    for model_name in models_list:
        sub_m = df_pred.filter(pl.col("model") == model_name).sort("ts_local")
        for date_str in all_dates:
            sub = sub_m.filter(pl.col("date_str") == date_str).sort("ts_local")
            if sub.shape[0] == 0:
                continue
            if date_str not in multimodel_data:
                multimodel_data[date_str] = {}
            multimodel_data[date_str][model_name] = sub["pred"].to_list()

    # MAE par jour pour le calendrier
    calendar_data = []
    for date_str in all_dates:
        mae_best  = mae_per_model.get(best_model, {}).get(date_str, None)
        mae_oiken = oiken_mae_dict.get(date_str, None)
        split     = day_data[date_str]["split"]
        calendar_data.append({
            "date":      date_str,
            "mae_model": round(mae_best, 5)  if mae_best  is not None else None,
            "mae_oiken": round(mae_oiken, 5) if mae_oiken is not None else None,
            "split":     split,
            "better":    (mae_best < mae_oiken)
                         if mae_best is not None and mae_oiken is not None
                         else None,
        })

    # ── Serialiser en JSON ────────────────────────────────────────────────
    day_data_json       = json.dumps(day_data)
    multimodel_data_json = json.dumps(multimodel_data)
    calendar_data_json  = json.dumps(calendar_data)
    models_list_json    = json.dumps(models_list)

    # ── HTML ──────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Visualisation predictions v4</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.26.0/plotly.min.js"></script>
<style>
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ background:{BG}; color:{TEXT}; font-family:'Courier New',monospace; padding:16px; }}
h1 {{ font-size:18px; letter-spacing:2px; margin-bottom:4px; }}
.subtitle {{ color:{SUBTEXT}; font-size:12px; margin-bottom:20px; }}

/* Layout */
.layout {{ display:grid; grid-template-columns:1fr 340px; gap:16px; }}
.left {{ display:flex; flex-direction:column; gap:12px; }}
.right {{ display:flex; flex-direction:column; gap:12px; }}

/* Cards */
.card {{ background:{BG_PAPER}; border:1px solid #21262D; border-radius:8px; padding:14px; }}
.card-title {{ font-size:11px; color:{SUBTEXT}; letter-spacing:1px; margin-bottom:10px; text-transform:uppercase; }}

/* Navigation */
.nav {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; }}
.btn {{ background:#21262D; border:1px solid #30363D; color:{TEXT}; padding:5px 12px;
        border-radius:4px; cursor:pointer; font-family:inherit; font-size:11px; }}
.btn:hover {{ background:#30363D; }}
.btn.active {{ background:#2E75B6; border-color:#2E75B6; }}
.btn-nav {{ background:none; border:1px solid #30363D; color:{TEXT}; padding:4px 10px;
            border-radius:4px; cursor:pointer; font-size:14px; }}
.btn-nav:hover {{ background:#21262D; }}
#date-label {{ font-size:13px; color:{TEXT}; min-width:120px; text-align:center; }}
.mode-btn {{ background:#21262D; border:1px solid #30363D; color:{SUBTEXT}; padding:4px 10px;
             border-radius:4px; cursor:pointer; font-size:11px; }}
.mode-btn.active {{ background:#2E75B6; border-color:#2E75B6; color:{TEXT}; }}

/* Metriques */
.metrics {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; }}
.metric {{ background:#21262D; border-radius:6px; padding:10px; text-align:center; }}
.metric-val {{ font-size:18px; font-weight:bold; }}
.metric-lbl {{ font-size:10px; color:{SUBTEXT}; margin-top:2px; }}
.better {{ color:#27AE60; }}
.worse  {{ color:#E74C3C; }}
.equal  {{ color:{SUBTEXT}; }}

/* Calendrier */
.calendar {{ width:100%; border-collapse:collapse; }}
.calendar th {{ font-size:10px; color:{SUBTEXT}; padding:3px; text-align:center; }}
.calendar td {{ width:40px; height:40px; text-align:center; cursor:pointer;
                border-radius:4px; font-size:10px; position:relative; }}
.calendar td:hover {{ border:1px solid #30363D; }}
.calendar td.selected {{ border:2px solid #2E75B6 !important; }}
.calendar td.empty {{ cursor:default; }}
.cal-day {{ font-size:10px; color:{SUBTEXT}; }}
.cal-mae {{ font-size:9px; margin-top:1px; }}
.cal-nav {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }}
.cal-nav span {{ font-size:12px; }}
.legend {{ display:flex; gap:12px; flex-wrap:wrap; margin-top:8px; }}
.legend-item {{ display:flex; align-items:center; gap:4px; font-size:10px; color:{SUBTEXT}; }}
.legend-dot {{ width:10px; height:10px; border-radius:2px; }}

/* Modeles toggle */
.model-toggles {{ display:flex; gap:6px; flex-wrap:wrap; }}
.model-toggle {{ padding:3px 8px; border-radius:4px; cursor:pointer; font-size:10px;
                 border:1px solid #30363D; background:#21262D; color:{SUBTEXT}; }}
.model-toggle.active {{ color:{TEXT}; }}

/* Split badge */
.split-badge {{ display:inline-block; padding:2px 8px; border-radius:4px; font-size:10px; margin-left:8px; }}
.split-train {{ background:#1a3a1a; color:#27AE60; }}
.split-val   {{ background:#1a2a3a; color:#2E75B6; }}
.split-test  {{ background:#3a1a1a; color:#E74C3C; }}
</style>
</head>
<body>

<h1>Visualisation predictions — Walk-forward v4</h1>
<div class="subtitle">
  XGBoost walk-forward | heure locale Europe/Zurich | comparaison avec Oiken forecast
</div>

<div class="layout">

  <!-- COLONNE GAUCHE -->
  <div class="left">

    <!-- Navigation -->
    <div class="card">
      <div class="card-title">Navigation</div>
      <div class="nav">
        <button class="btn-nav" onclick="navigate(-1)">&#8592;</button>
        <span id="date-label">—</span>
        <button class="btn-nav" onclick="navigate(1)">&#8594;</button>
        &nbsp;
        <button class="mode-btn active" id="mode-day" onclick="setMode('day')">Jour</button>
        <button class="mode-btn" id="mode-week" onclick="setMode('week')">Semaine</button>
        &nbsp;
        <span style="font-size:11px;color:{SUBTEXT}">Modeles :</span>
        <div class="model-toggles" id="model-toggles"></div>
      </div>
    </div>

    <!-- Graphique principal -->
    <div class="card">
      <div class="card-title">Courbe de charge
        <span id="split-badge" class="split-badge"></span>
      </div>
      <div id="chart-main" style="height:380px;"></div>
    </div>

    <!-- Metriques du jour/semaine -->
    <div class="card">
      <div class="card-title">Metriques <span id="period-label"></span></div>
      <div class="metrics" id="metrics-grid"></div>
    </div>

  </div>

  <!-- COLONNE DROITE -->
  <div class="right">

    <!-- Calendrier -->
    <div class="card">
      <div class="card-title">Calendrier MAE</div>
      <div class="cal-nav">
        <button class="btn-nav" onclick="prevMonth()">&#8592;</button>
        <span id="cal-month-label">—</span>
        <button class="btn-nav" onclick="nextMonth()">&#8594;</button>
      </div>
      <table class="calendar" id="calendar-table"></table>
      <div class="legend">
        <div class="legend-item"><div class="legend-dot" style="background:#27AE60"></div> Meilleur qu'Oiken</div>
        <div class="legend-item"><div class="legend-dot" style="background:#E74C3C"></div> Moins bon qu'Oiken</div>
        <div class="legend-item"><div class="legend-dot" style="background:#21262D"></div> Val</div>
        <div class="legend-item"><div class="legend-dot" style="background:#2C1F1F"></div> Test</div>
      </div>
    </div>

    <!-- Stats globales -->
    <div class="card">
      <div class="card-title">Stats globales (walk-forward)</div>
      <div id="global-stats"></div>
    </div>

  </div>
</div>

<script>
const DAY_DATA        = {day_data_json};
const MULTIMODEL_DATA = {multimodel_data_json};
const CALENDAR_DATA   = {calendar_data_json};
const MODELS_LIST     = {models_list_json};
const BEST_MODEL      = "{best_model}";

const MODEL_COLORS = {{
  "random_forest": "#E74C3C",
  "lightgbm":      "#FFC000",
  "xgboost":       "#2E75B6",
}};
const TRUE_COLOR     = "#27AE60";
const FORECAST_COLOR = "#9B59B6";

// State
let allDates     = Object.keys(DAY_DATA).sort();
let calData      = {{}};
CALENDAR_DATA.forEach(d => {{ calData[d.date] = d; }});

let currentIdx   = 0;
let currentMode  = "day";
let calYear      = null;
let calMonth     = null;
let activeModels = new Set([BEST_MODEL]);

// Init
function init() {{
  // Trouver premier jour val
  let firstVal = allDates.find(d => DAY_DATA[d].split === "val") || allDates[0];
  currentIdx   = allDates.indexOf(firstVal);

  // Init calendrier sur le premier mois val
  let parts = firstVal.split("-");
  calYear   = parseInt(parts[0]);
  calMonth  = parseInt(parts[1]) - 1;

  // Creer les toggles modeles
  let togglesDiv = document.getElementById("model-toggles");
  MODELS_LIST.forEach(m => {{
    let btn = document.createElement("div");
    btn.className = "model-toggle" + (activeModels.has(m) ? " active" : "");
    btn.style.borderColor = MODEL_COLORS[m] || "#888";
    if (activeModels.has(m)) btn.style.background = (MODEL_COLORS[m] || "#888") + "33";
    btn.textContent = m.replace("_", " ");
    btn.onclick = () => toggleModel(m, btn);
    togglesDiv.appendChild(btn);
  }});

  // Stats globales
  renderGlobalStats();
  renderCalendar();
  renderAll();
}}

function toggleModel(name, btn) {{
  if (activeModels.has(name)) {{
    if (activeModels.size === 1) return;
    activeModels.delete(name);
    btn.classList.remove("active");
    btn.style.background = "#21262D";
  }} else {{
    activeModels.add(name);
    btn.classList.add("active");
    btn.style.background = (MODEL_COLORS[name] || "#888") + "33";
  }}
  renderAll();
}}

function setMode(mode) {{
  currentMode = mode;
  document.getElementById("mode-day").classList.toggle("active", mode === "day");
  document.getElementById("mode-week").classList.toggle("active", mode === "week");
  renderAll();
}}

function navigate(dir) {{
  let step = currentMode === "week" ? 7 : 1;
  currentIdx = Math.max(0, Math.min(allDates.length - 1, currentIdx + dir * step));
  renderAll();
  // Sync calendrier
  let d = allDates[currentIdx].split("-");
  calYear  = parseInt(d[0]);
  calMonth = parseInt(d[1]) - 1;
  renderCalendar();
}}

function renderAll() {{
  if (currentMode === "day") {{
    renderDay(allDates[currentIdx]);
  }} else {{
    renderWeek(currentIdx);
  }}
}}

function getDatesForWeek(startIdx) {{
  let dates = [];
  for (let i = startIdx; i < Math.min(startIdx + 7, allDates.length); i++) {{
    dates.push(allDates[i]);
  }}
  return dates;
}}

function renderDay(dateStr) {{
  let data = DAY_DATA[dateStr];
  if (!data) return;

  document.getElementById("date-label").textContent = dateStr;
  document.getElementById("period-label").textContent = "— " + dateStr;

  // Split badge
  let badge = document.getElementById("split-badge");
  badge.textContent = data.split;
  badge.className = "split-badge split-" + data.split;

  let traces = [];

  // Load reel
  traces.push({{
    x: data.ts, y: data.true,
    name: "Load réel",
    line: {{ color: TRUE_COLOR, width: 2 }},
    type: "scatter", mode: "lines",
  }});

  // Forecast Oiken
  traces.push({{
    x: data.ts, y: data.forecast,
    name: "Oiken forecast",
    line: {{ color: FORECAST_COLOR, width: 1.5, dash: "dot" }},
    type: "scatter", mode: "lines",
  }});

  // Modeles actifs
  let mmData = MULTIMODEL_DATA[dateStr] || {{}};
  activeModels.forEach(m => {{
    if (mmData[m]) {{
      traces.push({{
        x: data.ts, y: mmData[m],
        name: m.replace("_", " "),
        line: {{ color: MODEL_COLORS[m] || "#888", width: 1.5 }},
        type: "scatter", mode: "lines",
      }});
    }}
  }});

  Plotly.react("chart-main", traces, getLayout(dateStr), {{responsive: true}});
  renderMetrics([dateStr]);
}}

function renderWeek(startIdx) {{
  let dates  = getDatesForWeek(startIdx);
  let label  = dates[0] + " → " + dates[dates.length - 1];
  document.getElementById("date-label").textContent = label;
  document.getElementById("period-label").textContent = "— semaine";

  // Badge split du premier jour
  let firstData = DAY_DATA[dates[0]];
  let badge     = document.getElementById("split-badge");
  if (firstData) {{
    badge.textContent = firstData.split;
    badge.className   = "split-badge split-" + firstData.split;
  }}

  let allTs   = [];
  let allTrue = [];
  let allFc   = [];
  let modelPreds = {{}};
  activeModels.forEach(m => {{ modelPreds[m] = []; }});

  dates.forEach(d => {{
    let data = DAY_DATA[d];
    if (!data) return;
    allTs   = allTs.concat(data.ts);
    allTrue = allTrue.concat(data.true);
    allFc   = allFc.concat(data.forecast);
    let mm  = MULTIMODEL_DATA[d] || {{}};
    activeModels.forEach(m => {{
      modelPreds[m] = modelPreds[m].concat(mm[m] || new Array(data.ts.length).fill(null));
    }});
  }});

  let traces = [
    {{ x: allTs, y: allTrue, name: "Load réel",
       line: {{ color: TRUE_COLOR, width: 2 }}, type: "scatter", mode: "lines" }},
    {{ x: allTs, y: allFc, name: "Oiken forecast",
       line: {{ color: FORECAST_COLOR, width: 1.5, dash: "dot" }}, type: "scatter", mode: "lines" }},
  ];
  activeModels.forEach(m => {{
    traces.push({{
      x: allTs, y: modelPreds[m], name: m.replace("_", " "),
      line: {{ color: MODEL_COLORS[m] || "#888", width: 1.5 }},
      type: "scatter", mode: "lines",
    }});
  }});

  // Lignes de separation entre jours
  let shapes = [];
  let dayBoundaries = [];
  dates.forEach((d, i) => {{
    if (i > 0 && DAY_DATA[d]) {{
      dayBoundaries.push(DAY_DATA[d].ts[0]);
    }}
  }});
  dayBoundaries.forEach(ts => {{
    shapes.push({{
      type: "line", x0: ts, x1: ts, y0: 0, y1: 1,
      xref: "x", yref: "paper",
      line: {{ color: "rgba(200,200,200,0.1)", width: 1, dash: "dot" }},
    }});
  }});

  let layout = getLayout(label);
  layout.shapes = shapes;
  Plotly.react("chart-main", traces, layout, {{responsive: true}});
  renderMetrics(dates);
}}

function getLayout(title) {{
  return {{
    paper_bgcolor: "{BG_PAPER}", plot_bgcolor: "{BG}",
    font: {{ color: "{TEXT}", family: "Courier New", size: 11 }},
    margin: {{ l:50, r:20, t:30, b:50 }},
    xaxis: {{ gridcolor: "{C_GRID}", showline:true, linecolor:"#30363D",
              tickformat:"%H:%M\\n%d/%m" }},
    yaxis: {{ gridcolor: "{C_GRID}", showline:true, linecolor:"#30363D",
              title:"Load normalisé" }},
    legend: {{ bgcolor:"rgba(0,0,0,0.4)", bordercolor:"#30363D", borderwidth:1,
               font:{{size:10}} }},
    hovermode: "x unified",
  }};
}}

function renderMetrics(dates) {{
  let trueAll = [], predAll = [], fcAll = [];
  dates.forEach(d => {{
    let data = DAY_DATA[d];
    let mm   = MULTIMODEL_DATA[d] || {{}};
    if (!data) return;
    trueAll = trueAll.concat(data.true);
    fcAll   = fcAll.concat(data.forecast);
    // Meilleur modele actif
    let bestActive = [...activeModels][0];
    predAll = predAll.concat(mm[bestActive] || []);
  }});

  let maeModel  = mae(trueAll, predAll);
  let maeOiken  = mae(trueAll, fcAll);
  let rmseModel = rmse(trueAll, predAll);

  let diff    = maeModel - maeOiken;
  let diffPct = maeOiken > 0 ? (diff / maeOiken * 100) : 0;
  let cls     = diff < 0 ? "better" : (diff > 0 ? "worse" : "equal");
  let sign    = diff < 0 ? "▼ " : "▲ ";

  document.getElementById("metrics-grid").innerHTML = `
    <div class="metric">
      <div class="metric-val">${{maeModel.toFixed(4)}}</div>
      <div class="metric-lbl">MAE modele</div>
    </div>
    <div class="metric">
      <div class="metric-val">${{maeOiken.toFixed(4)}}</div>
      <div class="metric-lbl">MAE Oiken</div>
    </div>
    <div class="metric">
      <div class="metric-val ${{cls}}">${{sign}}${{Math.abs(diffPct).toFixed(1)}}%</div>
      <div class="metric-lbl">vs Oiken</div>
    </div>
    <div class="metric">
      <div class="metric-val">${{rmseModel.toFixed(4)}}</div>
      <div class="metric-lbl">RMSE modele</div>
    </div>
    <div class="metric">
      <div class="metric-val">${{dates.length}}</div>
      <div class="metric-lbl">jours</div>
    </div>
    <div class="metric">
      <div class="metric-val">${{trueAll.length}}</div>
      <div class="metric-lbl">points</div>
    </div>
  `;
}}

function mae(y_true, y_pred) {{
  let s = 0, n = 0;
  for (let i = 0; i < y_true.length; i++) {{
    if (y_true[i] != null && y_pred[i] != null) {{
      s += Math.abs(y_true[i] - y_pred[i]);
      n++;
    }}
  }}
  return n > 0 ? s / n : 0;
}}

function rmse(y_true, y_pred) {{
  let s = 0, n = 0;
  for (let i = 0; i < y_true.length; i++) {{
    if (y_true[i] != null && y_pred[i] != null) {{
      s += (y_true[i] - y_pred[i]) ** 2;
      n++;
    }}
  }}
  return n > 0 ? Math.sqrt(s / n) : 0;
}}

// ── Calendrier ──────────────────────────────────────────────────────────────
function renderCalendar() {{
  let label = new Date(calYear, calMonth, 1)
    .toLocaleString("fr-CH", {{month:"long", year:"numeric"}});
  document.getElementById("cal-month-label").textContent = label;

  let table    = document.getElementById("calendar-table");
  let daysInM  = new Date(calYear, calMonth + 1, 0).getDate();
  let firstDay = (new Date(calYear, calMonth, 1).getDay() + 6) % 7; // Lundi=0

  let html = "<tr>";
  ["Lu","Ma","Me","Je","Ve","Sa","Di"].forEach(d => {{
    html += `<th>${{d}}</th>`;
  }});
  html += "</tr><tr>";

  // Cases vides debut
  for (let i = 0; i < firstDay; i++) html += "<td class='empty'></td>";

  let col = firstDay;
  for (let day = 1; day <= daysInM; day++) {{
    let dateStr = `${{calYear}}-${{String(calMonth+1).padStart(2,"0")}}-${{String(day).padStart(2,"0")}}`;
    let info    = calData[dateStr];

    if (!info) {{
      html += `<td class="empty"><div class="cal-day">${{day}}</div></td>`;
    }} else {{
      let bg      = info.split === "test" ? "#2C1F1F" : "#1a2030";
      let dot     = info.better === true  ? "#27AE60" :
                    info.better === false ? "#E74C3C" : "#888";
      let maeStr  = info.mae_model != null ? info.mae_model.toFixed(3) : "—";
      let isSelected = allDates[currentIdx] === dateStr ? " selected" : "";

      html += `<td style="background:${{bg}}" class="${{isSelected}}"
                  onclick="jumpToDate('${{dateStr}}')">
        <div class="cal-day">${{day}}</div>
        <div class="cal-mae" style="color:${{dot}}">${{maeStr}}</div>
      </td>`;
    }}

    col++;
    if (col % 7 === 0 && day < daysInM) html += "</tr><tr>";
  }}

  // Cases vides fin
  let remaining = (7 - (col % 7)) % 7;
  for (let i = 0; i < remaining; i++) html += "<td class='empty'></td>";
  html += "</tr>";

  table.innerHTML = html;
}}

function jumpToDate(dateStr) {{
  let idx = allDates.indexOf(dateStr);
  if (idx >= 0) {{
    currentIdx = idx;
    renderAll();
    let d = dateStr.split("-");
    calYear  = parseInt(d[0]);
    calMonth = parseInt(d[1]) - 1;
    renderCalendar();
  }}
}}

function prevMonth() {{
  calMonth--;
  if (calMonth < 0) {{ calMonth = 11; calYear--; }}
  renderCalendar();
}}

function nextMonth() {{
  calMonth++;
  if (calMonth > 11) {{ calMonth = 0; calYear++; }}
  renderCalendar();
}}

// ── Stats globales ───────────────────────────────────────────────────────────
function renderGlobalStats() {{
  let splits = ["val", "test"];
  let html   = "";
  splits.forEach(split => {{
    let dates_s = allDates.filter(d => DAY_DATA[d].split === split);
    let trueAll = [], predAll = [], fcAll = [];
    dates_s.forEach(d => {{
      let data = DAY_DATA[d];
      let mm   = MULTIMODEL_DATA[d] || {{}};
      trueAll = trueAll.concat(data.true);
      fcAll   = fcAll.concat(data.forecast);
      predAll = predAll.concat(mm[BEST_MODEL] || []);
    }});
    let maeM = mae(trueAll, predAll);
    let maeO = mae(trueAll, fcAll);
    let diff = ((maeM - maeO) / maeO * 100).toFixed(1);
    let cls  = maeM < maeO ? "better" : "worse";
    let sign = maeM < maeO ? "▼ " : "▲ ";
    html += `
      <div style="margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid #21262D">
        <div style="font-size:11px;color:{SUBTEXT};margin-bottom:6px;text-transform:uppercase">
          ${{split}} (${{dates_s.length}} jours)
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px">
          <div class="metric">
            <div class="metric-val">${{maeM.toFixed(4)}}</div>
            <div class="metric-lbl">MAE ${{BEST_MODEL}}</div>
          </div>
          <div class="metric">
            <div class="metric-val">${{maeO.toFixed(4)}}</div>
            <div class="metric-lbl">MAE Oiken</div>
          </div>
        </div>
        <div style="text-align:center;margin-top:6px;font-size:12px" class="${{cls}}">
          ${{sign}}${{Math.abs(diff)}}% vs Oiken
        </div>
      </div>`;
  }});
  document.getElementById("global-stats").innerHTML = html;
}}

init();
</script>
</body>
</html>"""

    # ── Sauvegarde ────────────────────────────────────────────────────────
    FILE_OUT.write_text(html, encoding="utf-8")
    size_mb = FILE_OUT.stat().st_size / 1024 / 1024
    print(f"\n-> {FILE_OUT}")
    print(f"   {size_mb:.1f} MB")