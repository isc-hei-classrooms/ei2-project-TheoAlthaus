"""
Exploration visuelle — Oiken clean + golden
=============================================
Lit  : data/oiken_clean.parquet
       golden_data/oiken_golden_clean.parquet
Ecrit: golden_data/exploration_oiken.html

Visualisation interactive :
  - Courbe de charge (load + forecast)
  - Production PV (central valais, sion, sierre, remote)
  - Navigation jour / semaine / mois
  - Calendrier
  - Stats par periode
"""

import polars as pl
import json
from pathlib import Path

from config import DATA_DIR, FILE_OIKEN_CLEAN

GOLDEN_DIR = DATA_DIR.parent / "golden_data"
FILE_OUT   = GOLDEN_DIR / "exploration_oiken.html"
TZ_LOCAL   = "Europe/Zurich"

BG       = "#0F1117"
BG_PAPER = "#161B22"
TEXT     = "#E6EDF3"
SUBTEXT  = "#8B949E"

if __name__ == "__main__":

    print("Chargement des donnees...")

    # Charger et combiner oiken_clean + oiken_golden
    oiken_train = pl.read_parquet(FILE_OIKEN_CLEAN).sort("timestamp")
    oiken_golden = pl.read_parquet(
        GOLDEN_DIR / "oiken_golden_clean.parquet"
    ).sort("timestamp")

    # Combiner les deux
    oiken = pl.concat([
        oiken_train,
        oiken_golden.select(oiken_train.columns)
    ]).sort("timestamp").unique(subset=["timestamp"], keep="first")

    print(f"  Total : {oiken.shape[0]:,} lignes")
    print(f"  Periode : {oiken['timestamp'].min()} -> "
          f"{oiken['timestamp'].max()}")

    # Ajouter heure locale
    oiken = oiken.with_columns(
        pl.col("timestamp")
          .dt.convert_time_zone(TZ_LOCAL)
          .alias("ts_local")
    ).with_columns([
        pl.col("ts_local").dt.date().alias("date_locale"),
        pl.col("ts_local").dt.strftime("%Y-%m-%d").alias("date_str"),
        pl.col("ts_local").dt.strftime("%Y-%m").alias("month_str"),
    ])

    all_dates  = sorted(oiken["date_str"].unique().to_list())
    all_months = sorted(oiken["month_str"].unique().to_list())

    # Marquer les donnees golden
    golden_start_str = "2025-09-30"

    print(f"  Jours train  : "
          f"{len([d for d in all_dates if d < golden_start_str])}")
    print(f"  Jours golden : "
          f"{len([d for d in all_dates if d >= golden_start_str])}")

    # ── Donnees par jour ──────────────────────────────────────────────────
    print("Construction des donnees par jour...")

    day_data = {}
    for date_str in all_dates:
        sub = oiken.filter(pl.col("date_str") == date_str).sort("ts_local")
        if sub.shape[0] == 0:
            continue

        def to_list_safe(col):
            return [None if v is None else round(float(v), 5)
                    for v in sub[col].to_list()]

        day_data[date_str] = {
            "ts":               [str(t) for t in sub["ts_local"].to_list()],
            "load":             to_list_safe("load"),
            "forecast_load":    to_list_safe("forecast_load"),
            "pv_central_valais":to_list_safe("pv_central_valais"),
            "pv_sion":          to_list_safe("pv_sion"),
            "pv_sierre":        to_list_safe("pv_sierre"),
            "pv_remote":        to_list_safe("pv_remote"),
            "is_golden":        date_str >= golden_start_str,
        }

    # ── Stats par mois ────────────────────────────────────────────────────
    monthly_stats = {}
    for row in (
        oiken.with_columns([
            (pl.col("load") - pl.col("forecast_load")).abs().alias("err_fc"),
        ])
        .group_by("month_str")
        .agg([
            pl.col("load").mean().alias("mean_load"),
            pl.col("load").std().alias("std_load"),
            pl.col("pv_central_valais").mean().alias("mean_pv_central"),
            pl.col("pv_sion").mean().alias("mean_pv_sion"),
            pl.col("pv_sierre").mean().alias("mean_pv_sierre"),
            pl.col("err_fc").mean().alias("mae_oiken"),
            pl.len().alias("n_pts"),
        ])
        .sort("month_str")
        .iter_rows(named=True)
    ):
        monthly_stats[row["month_str"]] = {
            k: round(float(v), 5) if isinstance(v, float) else v
            for k, v in row.items()
            if k != "month_str"
        }

    # ── JSON ──────────────────────────────────────────────────────────────
    day_data_json      = json.dumps(day_data)
    all_months_json    = json.dumps(all_months)
    monthly_stats_json = json.dumps(monthly_stats)

    # ── HTML ──────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Exploration Oiken</title>
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
.layer-btn {{ background:#21262D; border:1px solid #30363D; color:{SUBTEXT};
              padding:3px 8px; border-radius:4px; cursor:pointer; font-size:10px; }}
.layer-btn.active {{ color:{TEXT}; }}
.metrics {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
.metric {{ background:#21262D; border-radius:6px; padding:10px; text-align:center; }}
.metric-val {{ font-size:16px; font-weight:bold; }}
.metric-lbl {{ font-size:10px; color:{SUBTEXT}; margin-top:2px; }}
.golden-badge {{ display:inline-block; padding:2px 8px; border-radius:4px;
                 font-size:10px; background:#2a1f3a; color:#9B59B6;
                 margin-left:8px; }}
.train-badge  {{ display:inline-block; padding:2px 8px; border-radius:4px;
                 font-size:10px; background:#1a2a3a; color:#2E75B6;
                 margin-left:8px; }}
.calendar {{ width:100%; border-collapse:collapse; }}
.calendar th {{ font-size:10px; color:{SUBTEXT}; padding:3px; text-align:center; }}
.calendar td {{ width:40px; height:36px; text-align:center; cursor:pointer;
                border-radius:4px; font-size:10px; }}
.calendar td:hover {{ border:1px solid #30363D; }}
.calendar td.selected {{ border:2px solid #2E75B6 !important; }}
.calendar td.empty {{ cursor:default; }}
.calendar td.golden {{ background:#1a1225 !important; }}
.cal-day {{ font-size:10px; color:{SUBTEXT}; }}
.cal-nav {{ display:flex; justify-content:space-between;
            align-items:center; margin-bottom:8px; }}
.legend {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:8px; }}
.legend-item {{ display:flex; align-items:center; gap:4px;
                font-size:10px; color:{SUBTEXT}; }}
.legend-dot {{ width:10px; height:10px; border-radius:2px; }}
</style>
</head>
<body>

<h1>Exploration Oiken — train + golden</h1>
<div class="subtitle">
  Load, forecast Oiken et production PV | 2022-10-01 → 2026-04-22 |
  heure locale Europe/Zurich
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
        &nbsp;
        <span id="period-badge"></span>
      </div>
      <div class="nav" style="margin-top:8px">
        <span style="font-size:10px;color:{SUBTEXT}">Couches :</span>
        <div class="layer-btn active" id="layer-load"
             onclick="toggleLayer('load', this)">Load</div>
        <div class="layer-btn active" id="layer-forecast"
             onclick="toggleLayer('forecast', this)">Forecast Oiken</div>
        <div class="layer-btn active" id="layer-pv-central"
             onclick="toggleLayer('pv-central', this)">PV Central Valais</div>
        <div class="layer-btn active" id="layer-pv-sion"
             onclick="toggleLayer('pv-sion', this)">PV Sion</div>
        <div class="layer-btn active" id="layer-pv-sierre"
             onclick="toggleLayer('pv-sierre', this)">PV Sierre</div>
        <div class="layer-btn active" id="layer-pv-remote"
             onclick="toggleLayer('pv-remote', this)">PV Remote</div>
      </div>
    </div>

    <!-- Graphique load + forecast -->
    <div class="card">
      <div class="card-title">Load et forecast Oiken</div>
      <div id="chart-load" style="height:300px;"></div>
    </div>

    <!-- Graphique PV -->
    <div class="card">
      <div class="card-title">Production PV (kWh)</div>
      <div id="chart-pv" style="height:250px;"></div>
    </div>

    <!-- Stats du jour/semaine/mois -->
    <div class="card">
      <div class="card-title">Stats de la periode</div>
      <div class="metrics" id="metrics-grid"></div>
    </div>

    <!-- Evolution mensuelle -->
    <div class="card">
      <div class="card-title">Evolution mensuelle — MAE Oiken et PV moyen</div>
      <div id="chart-monthly" style="height:300px;"></div>
    </div>

  </div>

  <div class="right">

    <!-- Calendrier -->
    <div class="card">
      <div class="card-title">Calendrier</div>
      <div class="cal-nav">
        <button class="btn-nav" onclick="prevMonth()">&#8592;</button>
        <span id="cal-month-label">—</span>
        <button class="btn-nav" onclick="nextMonth()">&#8594;</button>
      </div>
      <table class="calendar" id="calendar-table"></table>
      <div class="legend">
        <div class="legend-item">
          <div class="legend-dot" style="background:#1a2a3a"></div>
          Donnees train
        </div>
        <div class="legend-item">
          <div class="legend-dot" style="background:#1a1225"></div>
          Golden dataset
        </div>
      </div>
    </div>

    <!-- Info jour -->
    <div class="card">
      <div class="card-title">Info journee</div>
      <div id="day-info" style="font-size:11px;color:{SUBTEXT}">
        Cliquez sur un jour dans le calendrier
      </div>
    </div>

  </div>
</div>

<script>
const DAY_DATA       = {day_data_json};
const ALL_MONTHS     = {all_months_json};
const MONTHLY_STATS  = {monthly_stats_json};
const GOLDEN_START   = "2025-09-30";

const COLORS = {{
  load:       "#27AE60",
  forecast:   "#9B59B6",
  pv_central: "#FFC000",
  pv_sion:    "#2E75B6",
  pv_sierre:  "#E74C3C",
  pv_remote:  "#1ABC9C",
}};

let allDates    = Object.keys(DAY_DATA).sort();
let currentIdx  = allDates.indexOf(GOLDEN_START) >= 0
                  ? allDates.indexOf(GOLDEN_START) : 0;
let currentMode = "day";
let calYear     = null;
let calMonth    = null;

let activeLayers = new Set([
  "load","forecast","pv-central","pv-sion","pv-sierre","pv-remote"
]);

function init() {{
  let d = allDates[currentIdx].split("-");
  calYear  = parseInt(d[0]);
  calMonth = parseInt(d[1]) - 1;
  renderMonthlyChart();
  renderCalendar();
  renderAll();
}}

function toggleLayer(name, btn) {{
  if (activeLayers.has(name)) {{
    activeLayers.delete(name);
    btn.classList.remove("active");
  }} else {{
    activeLayers.add(name);
    btn.classList.add("active");
  }}
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
    let months  = ALL_MONTHS;
    let curM    = allDates[currentIdx].substring(0,7);
    let idx     = months.indexOf(curM);
    let newM    = months[Math.max(0,Math.min(months.length-1,idx+dir))];
    let firstD  = allDates.find(d => d.startsWith(newM));
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

function getDates() {{
  if (currentMode === "day")
    return [allDates[currentIdx]];
  if (currentMode === "week") {{
    let dates = [];
    for (let i=currentIdx; i<Math.min(currentIdx+7,allDates.length); i++)
      dates.push(allDates[i]);
    return dates;
  }}
  let m = allDates[currentIdx].substring(0,7);
  return allDates.filter(d => d.startsWith(m));
}}

function renderAll() {{
  let dates = getDates();
  let label = currentMode === "day"   ? dates[0] :
              currentMode === "week"  ? dates[0]+" → "+dates[dates.length-1] :
              dates[0].substring(0,7);
  document.getElementById("date-label").textContent = label;

  // Badge train/golden
  let badge = document.getElementById("period-badge");
  let isGolden = dates.some(d => d >= GOLDEN_START);
  let isTrain  = dates.some(d => d < GOLDEN_START);
  if (isGolden && isTrain)
    badge.innerHTML = `<span class="train-badge">Train</span>
                       <span class="golden-badge">Golden</span>`;
  else if (isGolden)
    badge.innerHTML = `<span class="golden-badge">Golden</span>`;
  else
    badge.innerHTML = `<span class="train-badge">Train</span>`;

  // Assembler les donnees
  let allTs=[], loadArr=[], fcArr=[], pvCArr=[], pvSArr=[], pvSiArr=[], pvRArr=[];
  dates.forEach(d => {{
    let data = DAY_DATA[d]; if (!data) return;
    allTs   = allTs.concat(data.ts);
    loadArr = loadArr.concat(data.load);
    fcArr   = fcArr.concat(data.forecast_load);
    pvCArr  = pvCArr.concat(data.pv_central_valais);
    pvSArr  = pvSArr.concat(data.pv_sion);
    pvSiArr = pvSiArr.concat(data.pv_sierre);
    pvRArr  = pvRArr.concat(data.pv_remote);
  }});

  // Separateurs de jours
  let shapes = [];
  if (dates.length > 1 && dates.length <= 31) {{
    dates.slice(1).forEach(d => {{
      if (DAY_DATA[d]) shapes.push({{
        type:"line", x0:DAY_DATA[d].ts[0], x1:DAY_DATA[d].ts[0],
        y0:0, y1:1, xref:"x", yref:"paper",
        line:{{color:"rgba(200,200,200,0.08)",width:1,dash:"dot"}},
      }});
    }});
  }}

  let baseLayout = {{
    paper_bgcolor:"{BG_PAPER}", plot_bgcolor:"{BG}",
    font:{{color:"{TEXT}",family:"Courier New",size:11}},
    margin:{{l:50,r:20,t:20,b:50}},
    xaxis:{{gridcolor:"rgba(200,200,200,0.1)",showline:true,
            linecolor:"#30363D",tickformat:"%H:%M\\n%d/%m"}},
    legend:{{bgcolor:"rgba(0,0,0,0.4)",bordercolor:"#30363D",
             borderwidth:1,font:{{size:10}}}},
    hovermode:"x unified",
    shapes: shapes,
  }};

  // ── Graphique load ───────────────────────────────────────────────────
  let tracesLoad = [];
  if (activeLayers.has("load"))
    tracesLoad.push({{
      x:allTs, y:loadArr, name:"Load réel",
      line:{{color:COLORS.load,width:2}},
      type:"scatter", mode:"lines",
    }});
  if (activeLayers.has("forecast"))
    tracesLoad.push({{
      x:allTs, y:fcArr, name:"Forecast Oiken",
      line:{{color:COLORS.forecast,width:1.5,dash:"dot"}},
      type:"scatter", mode:"lines",
    }});

  Plotly.react("chart-load", tracesLoad, {{
    ...baseLayout,
    yaxis:{{...baseLayout.xaxis,
            gridcolor:"rgba(200,200,200,0.1)",
            title:"Load normalisé"}},
    height:300,
  }}, {{responsive:true}});

  // ── Graphique PV ─────────────────────────────────────────────────────
  let tracesPV = [];
  if (activeLayers.has("pv-central"))
    tracesPV.push({{
      x:allTs, y:pvCArr, name:"PV Central Valais",
      line:{{color:COLORS.pv_central,width:1.5}},
      type:"scatter", mode:"lines",
    }});
  if (activeLayers.has("pv-sion"))
    tracesPV.push({{
      x:allTs, y:pvSArr, name:"PV Sion",
      line:{{color:COLORS.pv_sion,width:1.5}},
      type:"scatter", mode:"lines",
    }});
  if (activeLayers.has("pv-sierre"))
    tracesPV.push({{
      x:allTs, y:pvSiArr, name:"PV Sierre",
      line:{{color:COLORS.pv_sierre,width:1.5}},
      type:"scatter", mode:"lines",
    }});
  if (activeLayers.has("pv-remote"))
    tracesPV.push({{
      x:allTs, y:pvRArr, name:"PV Remote",
      line:{{color:COLORS.pv_remote,width:1.5}},
      type:"scatter", mode:"lines",
    }});

  Plotly.react("chart-pv", tracesPV, {{
    ...baseLayout,
    yaxis:{{...baseLayout.xaxis,
            gridcolor:"rgba(200,200,200,0.1)",
            title:"Production (kWh)"}},
    height:250,
  }}, {{responsive:true}});

  // ── Stats ─────────────────────────────────────────────────────────────
  let validLoad = loadArr.filter(v => v!=null);
  let validPvC  = pvCArr.filter(v => v!=null);
  let errFc     = loadArr.map((v,i) =>
    v!=null&&fcArr[i]!=null ? Math.abs(v-fcArr[i]) : null
  ).filter(v=>v!=null);

  let meanLoad  = validLoad.reduce((a,b)=>a+b,0)/validLoad.length;
  let meanPvC   = validPvC.reduce((a,b)=>a+b,0)/validPvC.length;
  let maeOiken  = errFc.reduce((a,b)=>a+b,0)/errFc.length;
  let maxPvC    = Math.max(...validPvC);

  document.getElementById("metrics-grid").innerHTML = `
    <div class="metric">
      <div class="metric-val">${{meanLoad.toFixed(3)}}</div>
      <div class="metric-lbl">Load moyen</div>
    </div>
    <div class="metric">
      <div class="metric-val">${{maeOiken.toFixed(4)}}</div>
      <div class="metric-lbl">MAE Oiken</div>
    </div>
    <div class="metric">
      <div class="metric-val">${{meanPvC.toFixed(1)}}</div>
      <div class="metric-lbl">PV central moy (kWh)</div>
    </div>
    <div class="metric">
      <div class="metric-val">${{maxPvC.toFixed(1)}}</div>
      <div class="metric-lbl">PV central max (kWh)</div>
    </div>
    <div class="metric">
      <div class="metric-val">${{dates.length}}</div>
      <div class="metric-lbl">jours</div>
    </div>
    <div class="metric">
      <div class="metric-val">${{allTs.length}}</div>
      <div class="metric-lbl">points</div>
    </div>`;

  renderDayInfo(dates[0]);
}}

function renderDayInfo(dateStr) {{
  let data = DAY_DATA[dateStr];
  if (!data) return;

  let validPvC = data.pv_central_valais.filter(v=>v!=null);
  let maxPv    = Math.max(...validPvC).toFixed(1);
  let sumPv    = validPvC.reduce((a,b)=>a+b,0).toFixed(1);
  let isGolden = data.is_golden;
  let badge    = isGolden
    ? `<span class="golden-badge">Golden</span>`
    : `<span class="train-badge">Train</span>`;

  document.getElementById("day-info").innerHTML = `
    <div style="margin-bottom:8px">${{dateStr}} ${{badge}}</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px">
      <div class="metric">
        <div class="metric-val" style="font-size:13px">${{maxPv}}</div>
        <div class="metric-lbl">PV central max</div>
      </div>
      <div class="metric">
        <div class="metric-val" style="font-size:13px">${{sumPv}}</div>
        <div class="metric-lbl">PV central total</div>
      </div>
    </div>`;
}}

// ── Graphique mensuel ────────────────────────────────────────────────────
function renderMonthlyChart() {{
  let months   = ALL_MONTHS;
  let maeOiken = months.map(m => MONTHLY_STATS[m]?.mae_oiken ?? null);
  let pvCentral= months.map(m => MONTHLY_STATS[m]?.mean_pv_central ?? null);
  let isGolden = months.map(m => m >= "2025-09");

  Plotly.react("chart-monthly", [
    {{
      x: months, y: maeOiken, name: "MAE Oiken",
      line: {{color:COLORS.forecast,width:2}},
      type:"scatter", mode:"lines+markers",
      marker:{{size:6,color:months.map(m=>m>="2025-09"?"#9B59B6":"#6C3483")}},
      yaxis:"y1",
    }},
    {{
      x: months, y: pvCentral, name: "PV Central Valais moy",
      line: {{color:COLORS.pv_central,width:2}},
      type:"scatter", mode:"lines+markers",
      marker:{{size:6}},
      yaxis:"y2",
    }},
  ], {{
    paper_bgcolor:"{BG_PAPER}", plot_bgcolor:"{BG}",
    font:{{color:"{TEXT}",family:"Courier New",size:11}},
    margin:{{l:50,r:60,t:20,b:60}},
    xaxis:{{gridcolor:"rgba(200,200,200,0.1)",tickangle:-45}},
    yaxis:{{gridcolor:"rgba(200,200,200,0.1)",title:"MAE Oiken",
            titlefont:{{color:COLORS.forecast}}}},
    yaxis2:{{title:"PV moy (kWh)",overlaying:"y",side:"right",
             titlefont:{{color:COLORS.pv_central}},
             gridcolor:"rgba(0,0,0,0)"}},
    legend:{{bgcolor:"rgba(0,0,0,0.4)",bordercolor:"#30363D",borderwidth:1}},
    hovermode:"x unified",
    shapes:[{{
      type:"rect",x0:"2025-09",x1:months[months.length-1],
      y0:0,y1:1,xref:"x",yref:"paper",
      fillcolor:"rgba(155,89,182,0.05)",line:{{width:0}},
    }}],
  }}, {{responsive:true}});
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
    let data = DAY_DATA[ds];
    let sel  = allDates[currentIdx]===ds ? " selected" : "";
    let goldenCls = (data && data.is_golden) ? " golden" : "";
    let bg   = (data && data.is_golden) ? "#1a1225" : "#1a2030";

    if (!data) {{
      html+=`<td class="empty${{goldenCls}}">
        <div class="cal-day">${{day}}</div></td>`;
    }} else {{
      html+=`<td style="background:${{bg}}" class="${{sel}}${{goldenCls}}"
               onclick="jumpToDate('${{ds}}')">
        <div class="cal-day">${{day}}</div>
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
    ["day","week","month"].forEach(m => {{
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