"""
Visualisation interactive des predictions v6
=============================================
Lit  : data/models/wf_predictions_v6.parquet
       data/oiken_clean.parquet
Ecrit: data/visualisation_v6.html
"""

import polars as pl
import json

from config import DATA_DIR, FILE_OIKEN_CLEAN

TZ_LOCAL = "Europe/Zurich"
FILE_OUT = DATA_DIR / "visualisation_v6.html"

BG       = "#0F1117"
BG_PAPER = "#161B22"
TEXT     = "#E6EDF3"
SUBTEXT  = "#8B949E"

if __name__ == "__main__":

    print("Chargement des donnees...")
    df_pred = pl.read_parquet(DATA_DIR / "models" / "wf_predictions_v6.parquet")

    df_pred = df_pred.with_columns(
        pl.col("timestamp")
          .dt.convert_time_zone(TZ_LOCAL)
          .alias("ts_local")
    ).with_columns([
        pl.col("ts_local").dt.date().alias("date_locale"),
        pl.col("ts_local").dt.strftime("%Y-%m-%d").alias("date_str"),
    ])

    models_list = df_pred["model"].unique().sort().to_list()
    all_dates   = sorted(df_pred["date_str"].unique().to_list())

    print(f"  Modeles : {models_list}")
    print(f"  Jours   : {len(all_dates)}")

    # ── MAE par jour ──────────────────────────────────────────────────────
    print("Calcul MAE par jour...")

    oiken_mae_dict = {}
    for row in (
        df_pred.filter(pl.col("model") == models_list[0])
        .with_columns((pl.col("true") - pl.col("forecast")).abs().alias("err"))
        .group_by("date_str")
        .agg(pl.col("err").mean().alias("mae_oiken"))
        .iter_rows(named=True)
    ):
        oiken_mae_dict[row["date_str"]] = row["mae_oiken"]

    mae_per_model = {}
    for model_name in models_list:
        d = {}
        for row in (
            df_pred.filter(pl.col("model") == model_name)
            .with_columns((pl.col("true") - pl.col("pred")).abs().alias("err"))
            .group_by("date_str")
            .agg(pl.col("err").mean().alias("mae"))
            .iter_rows(named=True)
        ):
            d[row["date_str"]] = row["mae"]
        mae_per_model[model_name] = d

    # ── Donnees par jour ──────────────────────────────────────────────────
    best_model = "xgboost"
    df_best    = df_pred.filter(pl.col("model") == best_model).sort("ts_local")

    day_data = {}
    for date_str in all_dates:
        sub = df_best.filter(pl.col("date_str") == date_str).sort("ts_local")
        if sub.shape[0] == 0:
            continue
        day_data[date_str] = {
            "ts":       [str(t) for t in sub["ts_local"].to_list()],
            "true":     sub["true"].to_list(),
            "pred":     sub["pred"].to_list(),
            "forecast": sub["forecast"].to_list(),
            "split":    sub["split"][0],
        }

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

    calendar_data = []
    for date_str in all_dates:
        if date_str not in day_data:
            continue
        mae_best  = mae_per_model.get(best_model, {}).get(date_str, None)
        mae_oiken = oiken_mae_dict.get(date_str, None)
        calendar_data.append({
            "date":      date_str,
            "mae_model": round(mae_best,  5) if mae_best  is not None else None,
            "mae_oiken": round(mae_oiken, 5) if mae_oiken is not None else None,
            "split":     day_data[date_str]["split"],
            "better":    (mae_best < mae_oiken)
                         if mae_best is not None and mae_oiken is not None
                         else None,
        })

    bug_dates = [
        d for d in all_dates
        if d >= "2025-09-13" and d <= "2025-09-17"
    ]

    # ── JSON ──────────────────────────────────────────────────────────────
    day_data_json        = json.dumps(day_data)
    multimodel_data_json = json.dumps(multimodel_data)
    calendar_data_json   = json.dumps(calendar_data)
    models_list_json     = json.dumps(models_list)
    bug_dates_json       = json.dumps(bug_dates)

    # ── HTML ──────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Visualisation predictions v6</title>
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
.model-toggles {{ display:flex; gap:6px; flex-wrap:wrap; }}
.model-toggle {{ padding:3px 8px; border-radius:4px; cursor:pointer;
                 font-size:10px; border:1px solid #30363D;
                 background:#21262D; color:{SUBTEXT}; }}
.model-toggle.active {{ color:{TEXT}; }}
.metrics {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; }}
.metric {{ background:#21262D; border-radius:6px; padding:10px; text-align:center; }}
.metric-val {{ font-size:18px; font-weight:bold; }}
.metric-lbl {{ font-size:10px; color:{SUBTEXT}; margin-top:2px; }}
.better {{ color:#27AE60; }}
.worse  {{ color:#E74C3C; }}
.split-badge {{ display:inline-block; padding:2px 8px; border-radius:4px;
                font-size:10px; margin-left:8px; }}
.split-train {{ background:#1a3a1a; color:#27AE60; }}
.split-val   {{ background:#1a2a3a; color:#2E75B6; }}
.split-test  {{ background:#3a1a1a; color:#E74C3C; }}
.calendar {{ width:100%; border-collapse:collapse; }}
.calendar th {{ font-size:10px; color:{SUBTEXT}; padding:3px; text-align:center; }}
.calendar td {{ width:40px; height:40px; text-align:center; cursor:pointer;
                border-radius:4px; font-size:10px; }}
.calendar td:hover {{ border:1px solid #30363D; }}
.calendar td.selected {{ border:2px solid #2E75B6 !important; }}
.calendar td.empty {{ cursor:default; }}
.calendar td.bugged {{ opacity:0.4; }}
.cal-day {{ font-size:10px; color:{SUBTEXT}; }}
.cal-mae {{ font-size:9px; margin-top:1px; }}
.cal-nav {{ display:flex; justify-content:space-between;
            align-items:center; margin-bottom:8px; }}
.legend {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:8px; }}
.legend-item {{ display:flex; align-items:center; gap:4px;
                font-size:10px; color:{SUBTEXT}; }}
.legend-dot {{ width:10px; height:10px; border-radius:2px; }}
</style>
</head>
<body>

<h1>Visualisation predictions — Walk-forward v6</h1>
<div class="subtitle">
  XGBoost + LightGBM | PV + meteo + calendrier | heure locale Europe/Zurich |
  comparaison Oiken forecast | periode buguee 13-17 sept 2025 exclue
</div>

<div class="layout">
  <div class="left">

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
        &nbsp;
        <span style="font-size:11px;color:{SUBTEXT}">Modeles :</span>
        <div class="model-toggles" id="model-toggles"></div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Courbe de charge
        <span id="split-badge" class="split-badge"></span>
      </div>
      <div id="chart-main" style="height:400px;"></div>
    </div>

    <div class="card">
      <div class="card-title">Metriques
        <span id="period-label" style="color:{TEXT}"></span>
      </div>
      <div class="metrics" id="metrics-grid"></div>
    </div>

  </div>

  <div class="right">

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
        <div class="legend-item">
          <div class="legend-dot" style="background:#555"></div>
          Oiken bugue
        </div>
      </div>
    </div>

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
const BUG_DATES       = new Set({bug_dates_json});
const BEST_MODEL      = "{best_model}";

const MODEL_COLORS = {{
  "lightgbm": "#FFC000",
  "xgboost":  "#2E75B6",
}};
const TRUE_COLOR     = "#27AE60";
const FORECAST_COLOR = "#9B59B6";

let allDates     = Object.keys(DAY_DATA).sort();
let calData      = {{}};
CALENDAR_DATA.forEach(d => {{ calData[d.date] = d; }});
let currentIdx   = 0;
let currentMode  = "day";
let calYear      = null;
let calMonth     = null;
let activeModels = new Set(MODELS_LIST);

function init() {{
  let firstVal = allDates.find(d => DAY_DATA[d].split === "val") || allDates[0];
  currentIdx   = allDates.indexOf(firstVal);
  let parts    = firstVal.split("-");
  calYear      = parseInt(parts[0]);
  calMonth     = parseInt(parts[1]) - 1;

  let div = document.getElementById("model-toggles");
  MODELS_LIST.forEach(m => {{
    let btn = document.createElement("div");
    btn.className = "model-toggle active";
    btn.style.borderColor = MODEL_COLORS[m] || "#888";
    btn.style.background  = (MODEL_COLORS[m] || "#888") + "33";
    btn.style.color       = "{TEXT}";
    btn.textContent       = m;
    btn.onclick = () => toggleModel(m, btn);
    div.appendChild(btn);
  }});

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
    btn.style.color      = "{SUBTEXT}";
  }} else {{
    activeModels.add(name);
    btn.classList.add("active");
    btn.style.background = (MODEL_COLORS[name] || "#888") + "33";
    btn.style.color      = "{TEXT}";
  }}
  renderAll();
}}

function setMode(mode) {{
  currentMode = mode;
  document.getElementById("mode-day").classList.toggle("active", mode==="day");
  document.getElementById("mode-week").classList.toggle("active", mode==="week");
  renderAll();
}}

function navigate(dir) {{
  let step   = currentMode === "week" ? 7 : 1;
  currentIdx = Math.max(0, Math.min(allDates.length-1, currentIdx+dir*step));
  renderAll();
  let d = allDates[currentIdx].split("-");
  calYear  = parseInt(d[0]);
  calMonth = parseInt(d[1]) - 1;
  renderCalendar();
}}

function renderAll() {{
  currentMode === "day"
    ? renderDay(allDates[currentIdx])
    : renderWeek(currentIdx);
}}

function renderDay(dateStr) {{
  let data = DAY_DATA[dateStr];
  if (!data) return;

  document.getElementById("date-label").textContent    = dateStr;
  document.getElementById("period-label").textContent  = "— " + dateStr;
  let badge     = document.getElementById("split-badge");
  badge.textContent = data.split;
  badge.className   = "split-badge split-" + data.split;

  let traces = [
    {{ x:data.ts, y:data.true, name:"Load réel",
       line:{{color:TRUE_COLOR,width:2}}, type:"scatter", mode:"lines" }},
    {{ x:data.ts, y:data.forecast, name:"Oiken forecast",
       line:{{color:FORECAST_COLOR,width:1.5,dash:"dot"}},
       type:"scatter", mode:"lines" }},
  ];
  let mm = MULTIMODEL_DATA[dateStr] || {{}};
  activeModels.forEach(m => {{
    if (mm[m]) traces.push({{
      x:data.ts, y:mm[m], name:m,
      line:{{color:MODEL_COLORS[m]||"#888",width:1.5}},
      type:"scatter", mode:"lines",
    }});
  }});

  Plotly.react("chart-main", traces, getLayout(), {{responsive:true}});
  renderMetrics([dateStr]);
}}

function renderWeek(startIdx) {{
  let dates = [];
  for (let i=startIdx; i<Math.min(startIdx+7,allDates.length); i++)
    dates.push(allDates[i]);

  document.getElementById("date-label").textContent =
    dates[0] + " → " + dates[dates.length-1];
  document.getElementById("period-label").textContent = "— semaine";

  let allTs=[], allTrue=[], allFc=[];
  let mp={{}};
  activeModels.forEach(m => {{ mp[m]=[]; }});

  dates.forEach(d => {{
    let data = DAY_DATA[d]; if (!data) return;
    allTs   = allTs.concat(data.ts);
    allTrue = allTrue.concat(data.true);
    allFc   = allFc.concat(data.forecast);
    let mm  = MULTIMODEL_DATA[d] || {{}};
    activeModels.forEach(m => {{
      mp[m] = mp[m].concat(mm[m] || new Array(data.ts.length).fill(null));
    }});
  }});

  let traces = [
    {{ x:allTs, y:allTrue, name:"Load réel",
       line:{{color:TRUE_COLOR,width:2}}, type:"scatter", mode:"lines" }},
    {{ x:allTs, y:allFc, name:"Oiken forecast",
       line:{{color:FORECAST_COLOR,width:1.5,dash:"dot"}},
       type:"scatter", mode:"lines" }},
  ];
  activeModels.forEach(m => {{
    traces.push({{
      x:allTs, y:mp[m], name:m,
      line:{{color:MODEL_COLORS[m]||"#888",width:1.5}},
      type:"scatter", mode:"lines",
    }});
  }});

  let shapes = [];
  dates.slice(1).forEach(d => {{
    if (DAY_DATA[d]) shapes.push({{
      type:"line", x0:DAY_DATA[d].ts[0], x1:DAY_DATA[d].ts[0],
      y0:0, y1:1, xref:"x", yref:"paper",
      line:{{color:"rgba(200,200,200,0.1)",width:1,dash:"dot"}},
    }});
  }});

  let layout   = getLayout();
  layout.shapes = shapes;
  Plotly.react("chart-main", traces, layout, {{responsive:true}});

  let firstData = DAY_DATA[dates[0]];
  if (firstData) {{
    let badge     = document.getElementById("split-badge");
    badge.textContent = firstData.split;
    badge.className   = "split-badge split-" + firstData.split;
  }}
  renderMetrics(dates);
}}

function getLayout() {{
  return {{
    paper_bgcolor:"{BG_PAPER}", plot_bgcolor:"{BG}",
    font:{{color:"{TEXT}",family:"Courier New",size:11}},
    margin:{{l:50,r:20,t:30,b:50}},
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
    let data = DAY_DATA[d];
    let mm   = MULTIMODEL_DATA[d] || {{}};
    if (!data) return;
    trueAll = trueAll.concat(data.true);
    fcAll   = fcAll.concat(data.forecast);
    let bestActive = [...activeModels][0];
    predAll = predAll.concat(mm[bestActive] || []);
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
      <div class="metric-lbl">MAE modele</div>
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
      <div class="metric-lbl">RMSE modele</div>
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
    let isBug = BUG_DATES.has(ds);

    if(!info) {{
      html+=`<td class="empty"><div class="cal-day">${{day}}</div></td>`;
    }} else {{
      let bg  = info.split==="test" ? "#2C1F1F" : "#1a2030";
      let dot = isBug ? "#555" :
                info.better===true ? "#27AE60" :
                info.better===false ? "#E74C3C" : "#888";
      let maeStr = info.mae_model!=null ? info.mae_model.toFixed(3) : "—";
      let sel    = allDates[currentIdx]===ds ? " selected" : "";
      let bugCls = isBug ? " bugged" : "";
      html+=`<td style="background:${{bg}}" class="${{sel}}${{bugCls}}"
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

function renderGlobalStats() {{
  let html="";
  ["val","test"].forEach(split => {{
    let dates_s = allDates.filter(d=>DAY_DATA[d]&&DAY_DATA[d].split===split);
    let trueAll=[],predAll=[],fcAll=[];
    dates_s.forEach(d=>{{
      let data=DAY_DATA[d];
      let mm=MULTIMODEL_DATA[d]||{{}};
      trueAll=trueAll.concat(data.true);
      fcAll=fcAll.concat(data.forecast);
      predAll=predAll.concat(mm[BEST_MODEL]||[]);
    }});
    let maeM=mae(trueAll,predAll);
    let maeO=mae(trueAll,fcAll);
    let diff=((maeM-maeO)/maeO*100).toFixed(1);
    let cls=maeM<maeO?"better":"worse";
    let sign=maeM<maeO?"▼ ":"▲ ";
    html+=`
      <div style="margin-bottom:10px;padding-bottom:10px;
                  border-bottom:1px solid #21262D">
        <div style="font-size:11px;color:{SUBTEXT};margin-bottom:6px;
                    text-transform:uppercase">${{split}}</div>
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
        <div style="text-align:center;margin-top:6px;font-size:12px"
             class="${{cls}}">${{sign}}${{Math.abs(diff)}}% vs Oiken</div>
      </div>`;
  }});
  document.getElementById("global-stats").innerHTML=html;
}}

init();
</script>
</body>
</html>"""

    FILE_OUT.write_text(html, encoding="utf-8")
    size_mb = FILE_OUT.stat().st_size / 1024 / 1024
    print(f"\n-> {FILE_OUT}")
    print(f"   {size_mb:.1f} MB")