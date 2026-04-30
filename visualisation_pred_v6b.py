"""
Visualisation sliding window — LightGBM + Optuna
==================================================
Lit  : data/models/sw_results_lgb.parquet
       data/models/sw_predictions_lgb.parquet
       data/models/sw_hyperparams_lgb.parquet
Ecrit: data/visualisation_sw_lgb.html

Visualisation :
  - Comparaison MAE par mois (modele vs Oiken)
  - Courbe de charge jour/semaine/mois
  - Evolution des hyperparametres Optuna
  - Calendrier MAE
"""

import polars as pl
import json

from config import DATA_DIR

FILE_OUT = DATA_DIR / "visualisation_sw_lgb.html"
TZ_LOCAL = "Europe/Zurich"

BG       = "#0F1117"
BG_PAPER = "#161B22"
TEXT     = "#E6EDF3"
SUBTEXT  = "#8B949E"

if __name__ == "__main__":

    print("Chargement des donnees...")
    results = pl.read_parquet(DATA_DIR / "models" / "sw_results_lgb.parquet")
    preds   = pl.read_parquet(DATA_DIR / "models" / "sw_predictions_lgb.parquet")
    params  = pl.read_parquet(DATA_DIR / "models" / "sw_hyperparams_lgb.parquet")

    print(f"  Iterations : {results.shape[0]}")
    print(f"  Predictions : {preds.shape[0]:,}")

    # ── Ajouter heure locale aux predictions ──────────────────────────────
    preds = preds.with_columns(
        pl.col("timestamp")
          .dt.convert_time_zone(TZ_LOCAL)
          .alias("ts_local")
    ).with_columns([
        pl.col("ts_local").dt.date().alias("date_locale"),
        pl.col("ts_local").dt.strftime("%Y-%m-%d").alias("date_str"),
    ])

    all_dates = sorted(preds["date_str"].unique().to_list())
    print(f"  Jours : {len(all_dates)}")

    # ── MAE par jour ──────────────────────────────────────────────────────
    print("Calcul MAE par jour...")
    mae_model_dict = {}
    mae_oiken_dict = {}

    for row in (
        preds
        .filter(pl.col("oiken_valid") == True)
        .with_columns([
            (pl.col("true") - pl.col("pred")).abs().alias("err_model"),
            (pl.col("true") - pl.col("forecast")).abs().alias("err_oiken"),
        ])
        .group_by("date_str")
        .agg([
            pl.col("err_model").mean().alias("mae_model"),
            pl.col("err_oiken").mean().alias("mae_oiken"),
        ])
        .iter_rows(named=True)
    ):
        mae_model_dict[row["date_str"]] = row["mae_model"]
        mae_oiken_dict[row["date_str"]] = row["mae_oiken"]

    # ── Donnees par jour ──────────────────────────────────────────────────
    print("Construction donnees par jour...")
    day_data = {}
    for date_str in all_dates:
        sub = preds.filter(pl.col("date_str") == date_str).sort("ts_local")
        if sub.shape[0] == 0:
            continue
        day_data[date_str] = {
            "ts":         [str(t) for t in sub["ts_local"].to_list()],
            "true":       sub["true"].to_list(),
            "pred":       sub["pred"].to_list(),
            "forecast":   sub["forecast"].to_list(),
            "test_month": sub["test_month"][0],
            "valid":      sub["oiken_valid"].to_list(),
        }

    # ── Calendrier ────────────────────────────────────────────────────────
    calendar_data = []
    for date_str in all_dates:
        mae_m = mae_model_dict.get(date_str, None)
        mae_o = mae_oiken_dict.get(date_str, None)
        calendar_data.append({
            "date":        date_str,
            "mae_model":   round(mae_m, 5) if mae_m is not None else None,
            "mae_oiken":   round(mae_o, 5) if mae_o is not None else None,
            "test_month":  day_data[date_str]["test_month"],
            "better":      (mae_m < mae_o)
                           if mae_m is not None and mae_o is not None
                           else None,
        })

    # ── Resultats par mois ────────────────────────────────────────────────
    monthly_results = []
    for row in results.iter_rows(named=True):
        monthly_results.append({
            "iter":        row["iter"],
            "test_month":  row["test_month"],
            "mae_model":   round(row["mae_model"], 5),
            "mae_oiken":   round(row["mae_oiken"], 5),
            "diff_pct":    round(row["diff_pct"], 1),
            "n_test":      row["n_test"],
            "n_excl":      row["n_excl"],
        })

    # ── Hyperparametres ───────────────────────────────────────────────────
    hp_data = []
    for row in params.iter_rows(named=True):
        hp_data.append({
            "iter":             row["iter"],
            "test_month":       row["test_month"],
            "mae_val":          round(row["mae_val"], 5),
            "n_estimators":     row["n_estimators"],
            "learning_rate":    round(row["learning_rate"], 5),
            "max_depth":        row["max_depth"],
            "num_leaves":       row["num_leaves"],
            "min_child_samples":row["min_child_samples"],
            "subsample":        round(row["subsample"], 3),
            "colsample_bytree": round(row["colsample_bytree"], 3),
            "reg_alpha":        round(row["reg_alpha"], 6),
            "reg_lambda":       round(row["reg_lambda"], 6),
        })

    # ── JSON ──────────────────────────────────────────────────────────────
    day_data_json       = json.dumps(day_data)
    calendar_data_json  = json.dumps(calendar_data)
    monthly_results_json= json.dumps(monthly_results)
    hp_data_json        = json.dumps(hp_data)
    all_dates_json      = json.dumps(all_dates)

    # ── HTML ──────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Sliding window — LightGBM + Optuna</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.26.0/plotly.min.js"></script>
<style>
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ background:{BG}; color:{TEXT}; font-family:'Courier New',monospace; padding:16px; }}
h1 {{ font-size:18px; letter-spacing:2px; margin-bottom:4px; }}
.subtitle {{ color:{SUBTEXT}; font-size:12px; margin-bottom:20px; }}
.layout {{ display:grid; grid-template-columns:1fr 340px; gap:16px; }}
.left  {{ display:flex; flex-direction:column; gap:12px; }}
.right {{ display:flex; flex-direction:column; gap:12px; }}
.card {{ background:{BG_PAPER}; border:1px solid #21262D;
         border-radius:8px; padding:14px; }}
.card-title {{ font-size:11px; color:{SUBTEXT}; letter-spacing:1px;
               margin-bottom:10px; text-transform:uppercase; }}
.nav {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; }}
.btn-nav {{ background:none; border:1px solid #30363D; color:{TEXT};
            padding:4px 10px; border-radius:4px; cursor:pointer; font-size:14px; }}
.btn-nav:hover {{ background:#21262D; }}
#date-label {{ font-size:13px; color:{TEXT}; min-width:160px; text-align:center; }}
.mode-btn {{ background:#21262D; border:1px solid #30363D; color:{SUBTEXT};
             padding:4px 10px; border-radius:4px; cursor:pointer; font-size:11px; }}
.mode-btn.active {{ background:#2E75B6; border-color:#2E75B6; color:{TEXT}; }}
.metrics {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; }}
.metric {{ background:#21262D; border-radius:6px; padding:10px; text-align:center; }}
.metric-val {{ font-size:18px; font-weight:bold; }}
.metric-lbl {{ font-size:10px; color:{SUBTEXT}; margin-top:2px; }}
.better {{ color:#27AE60; }}
.worse  {{ color:#E74C3C; }}
.calendar {{ width:100%; border-collapse:collapse; }}
.calendar th {{ font-size:10px; color:{SUBTEXT}; padding:3px; text-align:center; }}
.calendar td {{ width:40px; height:40px; text-align:center; cursor:pointer;
                border-radius:4px; font-size:10px; }}
.calendar td:hover {{ border:1px solid #30363D; }}
.calendar td.selected {{ border:2px solid #2E75B6 !important; }}
.calendar td.empty {{ cursor:default; }}
.cal-day {{ font-size:10px; color:{SUBTEXT}; }}
.cal-mae {{ font-size:9px; margin-top:1px; }}
.cal-nav {{ display:flex; justify-content:space-between;
            align-items:center; margin-bottom:8px; }}
.legend {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:8px; }}
.legend-item {{ display:flex; align-items:center; gap:4px;
                font-size:10px; color:{SUBTEXT}; }}
.legend-dot {{ width:10px; height:10px; border-radius:2px; }}
.hp-table {{ width:100%; border-collapse:collapse; font-size:10px; }}
.hp-table th {{ color:{SUBTEXT}; padding:4px 6px; text-align:left;
                border-bottom:1px solid #21262D; white-space:nowrap; }}
.hp-table td {{ padding:3px 6px; border-bottom:1px solid #0D1117; }}
.hp-table tr:hover td {{ background:#21262D; }}
.month-badge {{ display:inline-block; padding:2px 8px; border-radius:4px;
                font-size:10px; background:#1a2a3a; color:#2E75B6;
                margin-left:8px; }}
</style>
</head>
<body>

<h1>Sliding window — LightGBM + Optuna</h1>
<div class="subtitle">
  22 mois train | 4 mois val (Optuna 30 essais) | 1 mois test |
  10 iterations | features v6
</div>

<div class="layout">
  <div class="left">

    <!-- Navigation -->
    <div class="card">
      <div class="card-title">Navigation</div>
      <div class="nav">
        <button class="btn-nav" onclick="navigate(-1)">&#8592;</button>
        <span id="date-label">—</span>
        <button class="btn-nav" onclick="navigate(1)">&#8594;</button>
        &nbsp;
        <button class="mode-btn active" id="mode-day"
                onclick="setMode('day')">Jour</button>
        <button class="mode-btn" id="mode-week"
                onclick="setMode('week')">Semaine</button>
        <button class="mode-btn" id="mode-month"
                onclick="setMode('month')">Mois</button>
        <span id="month-badge" class="month-badge"></span>
      </div>
    </div>

    <!-- Graphique courbe de charge -->
    <div class="card">
      <div class="card-title">Courbe de charge
        <span id="period-label" style="color:{TEXT}"></span>
      </div>
      <div id="chart-main" style="height:380px;"></div>
    </div>

    <!-- Metriques -->
    <div class="card">
      <div class="card-title">Metriques</div>
      <div class="metrics" id="metrics-grid"></div>
    </div>

    <!-- MAE par mois -->
    <div class="card">
      <div class="card-title">MAE par mois — LightGBM vs Oiken</div>
      <div id="chart-monthly" style="height:280px;"></div>
    </div>

    <!-- Hyperparametres -->
    <div class="card">
      <div class="card-title">Hyperparametres Optuna par iteration</div>
      <div id="chart-hp" style="height:280px;"></div>
      <div style="overflow-x:auto;margin-top:12px;">
        <table class="hp-table" id="hp-table"></table>
      </div>
    </div>

  </div>

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
        <div class="legend-item">
          <div class="legend-dot" style="background:#27AE60"></div>
          Meilleur qu'Oiken
        </div>
        <div class="legend-item">
          <div class="legend-dot" style="background:#E74C3C"></div>
          Moins bon
        </div>
      </div>
    </div>

    <!-- Stats globales -->
    <div class="card">
      <div class="card-title">Resume global</div>
      <div id="global-stats"></div>
    </div>

    <!-- Detail mois selectionne -->
    <div class="card">
      <div class="card-title">Detail iteration selectionnee</div>
      <div id="iter-detail" style="font-size:11px;"></div>
    </div>

  </div>
</div>

<script>
const DAY_DATA        = {day_data_json};
const CALENDAR_DATA   = {calendar_data_json};
const MONTHLY_RESULTS = {monthly_results_json};
const HP_DATA         = {hp_data_json};
const ALL_DATES       = {all_dates_json};

const TRUE_COLOR     = "#27AE60";
const FORECAST_COLOR = "#9B59B6";
const MODEL_COLOR    = "#FFC000";

let allDates    = ALL_DATES;
let calData     = {{}};
CALENDAR_DATA.forEach(d => {{ calData[d.date] = d; }});
let currentIdx  = 0;
let currentMode = "day";
let calYear     = null;
let calMonth    = null;

function init() {{
  let parts = allDates[0].split("-");
  calYear   = parseInt(parts[0]);
  calMonth  = parseInt(parts[1]) - 1;

  renderMonthlyChart();
  renderHpChart();
  renderHpTable();
  renderGlobalStats();
  renderCalendar();
  renderAll();
}}

function setMode(mode) {{
  currentMode = mode;
  ["day","week","month"].forEach(m => {{
    document.getElementById("mode-"+m).classList.toggle("active", m===mode);
  }});
  renderAll();
}}

function navigate(dir) {{
  if (currentMode === "month") {{
    let months  = [...new Set(allDates.map(d=>d.substring(0,7)))].sort();
    let curM    = allDates[currentIdx].substring(0,7);
    let idx     = months.indexOf(curM);
    let newM    = months[Math.max(0,Math.min(months.length-1,idx+dir))];
    let firstD  = allDates.find(d=>d.startsWith(newM));
    if (firstD) currentIdx = allDates.indexOf(firstD);
  }} else {{
    let step   = currentMode === "week" ? 7 : 1;
    currentIdx = Math.max(0, Math.min(allDates.length-1, currentIdx+dir*step));
  }}
  let d = allDates[currentIdx].split("-");
  calYear  = parseInt(d[0]);
  calMonth = parseInt(d[1]) - 1;
  renderCalendar();
  renderAll();
}}

function renderAll() {{
  if (currentMode === "day")   renderDay(allDates[currentIdx]);
  if (currentMode === "week")  renderMultiDay(getWeekDates(currentIdx));
  if (currentMode === "month") renderMultiDay(getMonthDates(allDates[currentIdx]));
}}

function getWeekDates(startIdx) {{
  let dates = [];
  for (let i=startIdx; i<Math.min(startIdx+7,allDates.length); i++)
    dates.push(allDates[i]);
  return dates;
}}

function getMonthDates(dateStr) {{
  let m = dateStr.substring(0,7);
  return allDates.filter(d=>d.startsWith(m));
}}

function renderDay(dateStr) {{
  let data = DAY_DATA[dateStr];
  if (!data) return;

  document.getElementById("date-label").textContent   = dateStr;
  document.getElementById("period-label").textContent = "— " + dateStr;
  document.getElementById("month-badge").textContent  = "test " + data.test_month;

  let traces = [
    {{ x:data.ts, y:data.true, name:"Load réel",
       line:{{color:TRUE_COLOR,width:2}}, type:"scatter", mode:"lines" }},
    {{ x:data.ts, y:data.forecast, name:"Oiken forecast",
       line:{{color:FORECAST_COLOR,width:1.5,dash:"dot"}},
       type:"scatter", mode:"lines" }},
    {{ x:data.ts, y:data.pred, name:"LightGBM",
       line:{{color:MODEL_COLOR,width:1.5}},
       type:"scatter", mode:"lines" }},
  ];

  Plotly.react("chart-main", traces, getLayout(), {{responsive:true}});
  renderMetrics([dateStr]);
  renderIterDetail(data.test_month);
}}

function renderMultiDay(dates) {{
  if (dates.length === 0) return;
  let label = currentMode==="month"
    ? dates[0].substring(0,7)
    : dates[0]+" → "+dates[dates.length-1];
  document.getElementById("date-label").textContent   = label;
  document.getElementById("period-label").textContent =
    currentMode==="month" ? "— mois" : "— semaine";
  document.getElementById("month-badge").textContent  =
    "test " + (DAY_DATA[dates[0]]?.test_month || "");

  let allTs=[], allTrue=[], allFc=[], allPred=[];
  dates.forEach(d => {{
    let data = DAY_DATA[d]; if (!data) return;
    allTs   = allTs.concat(data.ts);
    allTrue = allTrue.concat(data.true);
    allFc   = allFc.concat(data.forecast);
    allPred = allPred.concat(data.pred);
  }});

  let traces = [
    {{ x:allTs, y:allTrue, name:"Load réel",
       line:{{color:TRUE_COLOR,width:2}}, type:"scatter", mode:"lines" }},
    {{ x:allTs, y:allFc, name:"Oiken forecast",
       line:{{color:FORECAST_COLOR,width:1.5,dash:"dot"}},
       type:"scatter", mode:"lines" }},
    {{ x:allTs, y:allPred, name:"LightGBM",
       line:{{color:MODEL_COLOR,width:1.5}},
       type:"scatter", mode:"lines" }},
  ];

  let shapes = [];
  if (dates.length <= 31) {{
    dates.slice(1).forEach(d => {{
      if (DAY_DATA[d]) shapes.push({{
        type:"line", x0:DAY_DATA[d].ts[0], x1:DAY_DATA[d].ts[0],
        y0:0, y1:1, xref:"x", yref:"paper",
        line:{{color:"rgba(200,200,200,0.08)",width:1,dash:"dot"}},
      }});
    }});
  }}
  let layout = getLayout();
  layout.shapes = shapes;
  Plotly.react("chart-main", traces, layout, {{responsive:true}});
  renderMetrics(dates);
  renderIterDetail(DAY_DATA[dates[0]]?.test_month);
}}

function getLayout() {{
  return {{
    paper_bgcolor:"{BG_PAPER}", plot_bgcolor:"{BG}",
    font:{{color:"{TEXT}",family:"Courier New",size:11}},
    margin:{{l:50,r:20,t:20,b:50}},
    xaxis:{{gridcolor:"rgba(200,200,200,0.1)",showline:true,
            linecolor:"#30363D",tickformat:"%H:%M\\n%d/%m"}},
    yaxis:{{gridcolor:"rgba(200,200,200,0.1)",showline:true,
            linecolor:"#30363D",title:"Load normalisé"}},
    legend:{{bgcolor:"rgba(0,0,0,0.4)",bordercolor:"#30363D",
             borderwidth:1,font:{{size:10}}}},
    hovermode:"x unified",
  }};
}}

function renderMetrics(dates) {{
  let trueAll=[], predAll=[], fcAll=[];
  dates.forEach(d => {{
    let data = DAY_DATA[d]; if (!data) return;
    trueAll = trueAll.concat(data.true);
    predAll = predAll.concat(data.pred);
    fcAll   = fcAll.concat(data.forecast);
  }});

  let maeM  = mae(trueAll, predAll);
  let maeO  = mae(trueAll, fcAll);
  let rmseM = rmse(trueAll, predAll);
  let diff  = maeM - maeO;
  let pct   = maeO > 0 ? (diff/maeO*100) : 0;
  let cls   = diff < 0 ? "better" : "worse";
  let sign  = diff < 0 ? "▼ " : "▲ ";

  document.getElementById("metrics-grid").innerHTML = `
    <div class="metric">
      <div class="metric-val">${{maeM.toFixed(4)}}</div>
      <div class="metric-lbl">MAE LightGBM</div>
    </div>
    <div class="metric">
      <div class="metric-val">${{maeO.toFixed(4)}}</div>
      <div class="metric-lbl">MAE Oiken</div>
    </div>
    <div class="metric">
      <div class="metric-val ${{cls}}">${{sign}}${{Math.abs(pct).toFixed(1)}}%</div>
      <div class="metric-lbl">vs Oiken</div>
    </div>
    <div class="metric">
      <div class="metric-val">${{rmseM.toFixed(4)}}</div>
      <div class="metric-lbl">RMSE</div>
    </div>
    <div class="metric">
      <div class="metric-val">${{dates.length}}</div>
      <div class="metric-lbl">jours</div>
    </div>
    <div class="metric">
      <div class="metric-val">${{trueAll.length}}</div>
      <div class="metric-lbl">points</div>
    </div>`;
}}

function mae(yt,yp) {{
  let s=0,n=0;
  for(let i=0;i<yt.length;i++)
    if(yt[i]!=null&&yp[i]!=null){{s+=Math.abs(yt[i]-yp[i]);n++;}}
  return n>0?s/n:0;
}}
function rmse(yt,yp) {{
  let s=0,n=0;
  for(let i=0;i<yt.length;i++)
    if(yt[i]!=null&&yp[i]!=null){{s+=(yt[i]-yp[i])**2;n++;}}
  return n>0?Math.sqrt(s/n):0;
}}

// ── Graphique MAE mensuel ─────────────────────────────────────────────────
function renderMonthlyChart() {{
  let months   = MONTHLY_RESULTS.map(r=>r.test_month);
  let maeModel = MONTHLY_RESULTS.map(r=>r.mae_model);
  let maeOiken = MONTHLY_RESULTS.map(r=>r.mae_oiken);
  let colors   = MONTHLY_RESULTS.map(r=>r.diff_pct<0?"#27AE60":"#E74C3C");

  Plotly.react("chart-monthly", [
    {{
      x:months, y:maeOiken, name:"Oiken forecast",
      line:{{color:FORECAST_COLOR,width:2,dash:"dot"}},
      type:"scatter", mode:"lines+markers",
      marker:{{size:6}},
    }},
    {{
      x:months, y:maeModel, name:"LightGBM",
      line:{{color:MODEL_COLOR,width:2}},
      type:"scatter", mode:"lines+markers",
      marker:{{size:8, color:colors}},
    }},
  ], {{
    paper_bgcolor:"{BG_PAPER}", plot_bgcolor:"{BG}",
    font:{{color:"{TEXT}",family:"Courier New",size:11}},
    margin:{{l:50,r:20,t:20,b:60}},
    xaxis:{{gridcolor:"rgba(200,200,200,0.1)",tickangle:-45}},
    yaxis:{{gridcolor:"rgba(200,200,200,0.1)",title:"MAE"}},
    legend:{{bgcolor:"rgba(0,0,0,0.4)",bordercolor:"#30363D",borderwidth:1}},
    hovermode:"x unified",
  }}, {{responsive:true}});
}}

// ── Graphique hyperparametres ─────────────────────────────────────────────
function renderHpChart() {{
  let months = HP_DATA.map(r=>r.test_month);
  Plotly.react("chart-hp", [
    {{
      x:months, y:HP_DATA.map(r=>r.n_estimators), name:"n_estimators",
      line:{{color:"#2E75B6",width:2}}, type:"scatter", mode:"lines+markers",
      marker:{{size:6}}, yaxis:"y1",
    }},
    {{
      x:months, y:HP_DATA.map(r=>r.learning_rate), name:"learning_rate",
      line:{{color:"#E74C3C",width:2}}, type:"scatter", mode:"lines+markers",
      marker:{{size:6}}, yaxis:"y2",
    }},
    {{
      x:months, y:HP_DATA.map(r=>r.num_leaves), name:"num_leaves",
      line:{{color:"#27AE60",width:2}}, type:"scatter", mode:"lines+markers",
      marker:{{size:6}}, yaxis:"y1",
    }},
  ], {{
    paper_bgcolor:"{BG_PAPER}", plot_bgcolor:"{BG}",
    font:{{color:"{TEXT}",family:"Courier New",size:11}},
    margin:{{l:50,r:60,t:20,b:60}},
    xaxis:{{gridcolor:"rgba(200,200,200,0.1)",tickangle:-45}},
    yaxis:{{gridcolor:"rgba(200,200,200,0.1)",title:"n_estimators / num_leaves"}},
    yaxis2:{{title:"learning_rate",overlaying:"y",side:"right",
             gridcolor:"rgba(0,0,0,0)"}},
    legend:{{bgcolor:"rgba(0,0,0,0.4)",bordercolor:"#30363D",borderwidth:1}},
    hovermode:"x unified",
  }}, {{responsive:true}});
}}

// ── Table hyperparametres ─────────────────────────────────────────────────
function renderHpTable() {{
  let html = `<tr>
    <th>Mois test</th><th>MAE val</th><th>n_est</th>
    <th>lr</th><th>depth</th><th>leaves</th>
    <th>min_child</th><th>subsample</th><th>colsample</th>
  </tr>`;
  HP_DATA.forEach(r => {{
    html += `<tr>
      <td style="color:{TEXT}">${{r.test_month}}</td>
      <td>${{r.mae_val.toFixed(4)}}</td>
      <td>${{r.n_estimators}}</td>
      <td>${{r.learning_rate.toFixed(4)}}</td>
      <td>${{r.max_depth}}</td>
      <td>${{r.num_leaves}}</td>
      <td>${{r.min_child_samples}}</td>
      <td>${{r.subsample.toFixed(3)}}</td>
      <td>${{r.colsample_bytree.toFixed(3)}}</td>
    </tr>`;
  }});
  document.getElementById("hp-table").innerHTML = html;
}}

// ── Stats globales ────────────────────────────────────────────────────────
function renderGlobalStats() {{
  let n_better = MONTHLY_RESULTS.filter(r=>r.diff_pct<0).length;
  let n_total  = MONTHLY_RESULTS.length;
  let mae_mean = MONTHLY_RESULTS.reduce((a,r)=>a+r.mae_model,0)/n_total;
  let oik_mean = MONTHLY_RESULTS.reduce((a,r)=>a+r.mae_oiken,0)/n_total;
  let diff     = ((mae_mean-oik_mean)/oik_mean*100).toFixed(1);
  let cls      = mae_mean<oik_mean?"better":"worse";
  let sign     = mae_mean<oik_mean?"▼ ":"▲ ";

  document.getElementById("global-stats").innerHTML = `
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:10px">
      <div class="metric">
        <div class="metric-val">${{mae_mean.toFixed(4)}}</div>
        <div class="metric-lbl">MAE moy LightGBM</div>
      </div>
      <div class="metric">
        <div class="metric-val">${{oik_mean.toFixed(4)}}</div>
        <div class="metric-lbl">MAE moy Oiken</div>
      </div>
    </div>
    <div style="text-align:center;font-size:13px;margin-bottom:10px"
         class="${{cls}}">${{sign}}${{Math.abs(diff)}}% vs Oiken</div>
    <div style="text-align:center;font-size:11px;color:{SUBTEXT}">
      Mois meilleurs qu'Oiken : ${{n_better}}/${{n_total}}
    </div>`;
}}

// ── Detail iteration ──────────────────────────────────────────────────────
function renderIterDetail(testMonth) {{
  if (!testMonth) return;
  let res = MONTHLY_RESULTS.find(r=>r.test_month===testMonth);
  let hp  = HP_DATA.find(r=>r.test_month===testMonth);
  if (!res || !hp) return;

  let cls  = res.diff_pct<0?"better":"worse";
  let sign = res.diff_pct<0?"▼ ":"▲ ";

  document.getElementById("iter-detail").innerHTML = `
    <div style="margin-bottom:8px;font-size:12px;color:{TEXT}">
      Iter ${{res.iter}} — test ${{testMonth}}
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:8px">
      <div class="metric">
        <div class="metric-val" style="font-size:14px">${{res.mae_model.toFixed(4)}}</div>
        <div class="metric-lbl">MAE test</div>
      </div>
      <div class="metric">
        <div class="metric-val" style="font-size:14px">${{res.mae_oiken.toFixed(4)}}</div>
        <div class="metric-lbl">MAE Oiken</div>
      </div>
    </div>
    <div style="text-align:center;font-size:12px;margin-bottom:10px"
         class="${{cls}}">${{sign}}${{Math.abs(res.diff_pct)}}% vs Oiken</div>
    <div style="font-size:10px;color:{SUBTEXT};line-height:1.8">
      <div>MAE val (Optuna) : ${{hp.mae_val.toFixed(4)}}</div>
      <div>n_estimators     : ${{hp.n_estimators}}</div>
      <div>learning_rate    : ${{hp.learning_rate.toFixed(4)}}</div>
      <div>max_depth        : ${{hp.max_depth}}</div>
      <div>num_leaves       : ${{hp.num_leaves}}</div>
      <div>min_child_samples: ${{hp.min_child_samples}}</div>
      <div>subsample        : ${{hp.subsample.toFixed(3)}}</div>
      <div>colsample_bytree : ${{hp.colsample_bytree.toFixed(3)}}</div>
      <div>n_test pts       : ${{res.n_test.toLocaleString()}}</div>
      <div>n_excl pts       : ${{res.n_excl.toLocaleString()}}</div>
    </div>`;
}}

// ── Calendrier ───────────────────────────────────────────────────────────
function renderCalendar() {{
  let label = new Date(calYear,calMonth,1)
    .toLocaleString("fr-CH",{{month:"long",year:"numeric"}});
  document.getElementById("cal-month-label").textContent = label;

  let daysInM  = new Date(calYear,calMonth+1,0).getDate();
  let firstDay = (new Date(calYear,calMonth,1).getDay()+6)%7;
  let table    = document.getElementById("calendar-table");

  let html = "<tr>";
  ["Lu","Ma","Me","Je","Ve","Sa","Di"].forEach(d=>{{html+=`<th>${{d}}</th>`;}});
  html += "</tr><tr>";

  for(let i=0;i<firstDay;i++) html+="<td class='empty'></td>";

  let col=firstDay;
  for(let day=1;day<=daysInM;day++) {{
    let ds   = `${{calYear}}-${{String(calMonth+1).padStart(2,"0")}}-${{String(day).padStart(2,"0")}}`;
    let info = calData[ds];

    if(!info) {{
      html+=`<td class="empty"><div class="cal-day">${{day}}</div></td>`;
    }} else {{
      let dot = info.better===true  ? "#27AE60" :
                info.better===false ? "#E74C3C" : "#888";
      let maeStr = info.mae_model!=null ? info.mae_model.toFixed(3) : "—";
      let sel    = allDates[currentIdx]===ds ? " selected" : "";
      html+=`<td style="background:#1a2030" class="${{sel}}"
               onclick="jumpToDate('${{ds}}')">
        <div class="cal-day">${{day}}</div>
        <div class="cal-mae" style="color:${{dot}}">${{maeStr}}</div>
      </td>`;
    }}
    col++;
    if(col%7===0&&day<daysInM) html+="</tr><tr>";
  }}
  let rem=(7-col%7)%7;
  for(let i=0;i<rem;i++) html+="<td class='empty'></td>";
  html+="</tr>";
  table.innerHTML=html;
}}

function jumpToDate(ds) {{
  let idx=allDates.indexOf(ds);
  if(idx>=0) {{
    currentIdx=idx;
    currentMode="day";
    ["day","week","month"].forEach(m=>{{
      document.getElementById("mode-"+m).classList.toggle("active",m==="day");
    }});
    renderAll();
    let d=ds.split("-");
    calYear=parseInt(d[0]);
    calMonth=parseInt(d[1])-1;
    renderCalendar();
  }}
}}
function prevMonth() {{
  calMonth--; if(calMonth<0){{calMonth=11;calYear--;}}
  renderCalendar();
}}
function nextMonth() {{
  calMonth++; if(calMonth>11){{calMonth=0;calYear++;}}
  renderCalendar();
}}

init();
</script>
</body>
</html>"""

    FILE_OUT.write_text(html, encoding="utf-8")
    size_mb = FILE_OUT.stat().st_size / 1024 / 1024
    print(f"\n-> {FILE_OUT}")
    print(f"   {size_mb:.1f} MB")