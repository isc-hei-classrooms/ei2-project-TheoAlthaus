"""
Analyse importance des features — v4
======================================
Lit  : data/models/xgboost_v4.pkl
       data/models/lightgbm_v4.pkl
       data/models/random_forest_v4.pkl
       data/features_v4.parquet
Ecrit: data/feature_importance_v4.html
"""

import pickle
import json
import numpy as np
import polars as pl
from pathlib import Path

from config import DATA_DIR

MODELS_DIR = DATA_DIR / "models"
FILE_FEAT  = DATA_DIR / "features_v4.parquet"
FILE_OUT   = DATA_DIR / "feature_importance_v4.html"

BG       = "#0F1117"
BG_PAPER = "#161B22"
TEXT     = "#E6EDF3"
SUBTEXT  = "#8B949E"

# Couleurs par categorie
CAT_COLORS = {
    "calendaire":   "#2E75B6",
    "lag_load":     "#27AE60",
    "lag_meteo":    "#FFC000",
    "prev_meteo":   "#E74C3C",
}


def get_category(feat_name: str) -> str:
    if feat_name.startswith("lag_") and "pred_" in feat_name:
        return "prev_meteo"
    elif feat_name.startswith("feat_load"):
        return "lag_load"
    elif feat_name.startswith("feat_"):
        return "lag_meteo"
    else:
        return "calendaire"


def get_group(feat_name: str) -> str:
    """Groupe semantique pour affichage."""
    if "temperature" in feat_name:
        return "temperature"
    elif "radiation" in feat_name:
        return "radiation"
    elif "sunshine" in feat_name:
        return "sunshine"
    elif "humidity" in feat_name:
        return "humidity"
    elif "load" in feat_name:
        return "load"
    elif "hour" in feat_name:
        return "heure"
    elif "day_of_week" in feat_name:
        return "jour_semaine"
    elif "month" in feat_name:
        return "mois"
    elif "day_of_year" in feat_name:
        return "jour_annee"
    elif "weekend" in feat_name:
        return "weekend"
    elif "holiday" in feat_name:
        return "ferie_vacances"
    elif "school" in feat_name:
        return "vacances_scolaires"
    else:
        return "autre"


if __name__ == "__main__":

    # ── Chargement features ───────────────────────────────────────────────
    print("Chargement features v4...")
    df = pl.read_parquet(FILE_FEAT)
    feat_cols = [c for c in df.columns
                 if c not in ("timestamp", "date_locale", "target", "split")]
    print(f"  {len(feat_cols)} features")

    # ── Chargement modeles ────────────────────────────────────────────────
    models = {}
    for name in ["xgboost", "lightgbm", "random_forest"]:
        path = MODELS_DIR / f"{name}_v4.pkl"
        if path.exists():
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
            print(f"  {name} charge")

    # ── Extraction importances ────────────────────────────────────────────
    importances = {}
    for name, model in models.items():
        print(f"\nExtraction importances {name}...")

        if name == "xgboost":
            # Gain = contribution moyenne au gain d'information
            imp = model.get_booster().get_score(importance_type="gain")
            # Normaliser
            total = sum(imp.values())
            imp_norm = {k: v / total for k, v in imp.items()}
            # Certaines features peuvent etre absentes (jamais utilisees)
            imp_full = {f: imp_norm.get(f, 0.0) for f in feat_cols}

        elif name == "lightgbm":
            imp_vals = model.booster_.feature_importance(importance_type="gain")
            total    = imp_vals.sum()
            imp_full = {
                f: float(v / total) if total > 0 else 0.0
                for f, v in zip(feat_cols, imp_vals)
            }

        elif name == "random_forest":
            imp_vals = model.feature_importances_
            imp_full = {
                f: float(v)
                for f, v in zip(feat_cols, imp_vals)
            }

        importances[name] = imp_full
        n_nonzero = sum(1 for v in imp_full.values() if v > 0)
        print(f"  Features utilisees : {n_nonzero}/{len(feat_cols)}")

    # ── Construire le tableau de donnees ──────────────────────────────────
    rows = []
    for feat in feat_cols:
        cat   = get_category(feat)
        group = get_group(feat)
        row   = {
            "feature":  feat,
            "category": cat,
            "group":    group,
            "color":    CAT_COLORS[cat],
        }
        for name in models:
            row[name] = round(importances[name].get(feat, 0.0), 6)
        rows.append(row)

    # Trier par importance XGBoost decroissante
    rows.sort(key=lambda r: r.get("xgboost", 0), reverse=True)

    # ── Stats par categorie ───────────────────────────────────────────────
    cat_stats = {}
    for cat in CAT_COLORS:
        cat_rows = [r for r in rows if r["category"] == cat]
        cat_stats[cat] = {
            name: round(sum(r[name] for r in cat_rows), 4)
            for name in models
        }
        cat_stats[cat]["n_features"] = len(cat_rows)

    # ── Stats par groupe ──────────────────────────────────────────────────
    group_stats = {}
    for row in rows:
        g = row["group"]
        if g not in group_stats:
            group_stats[g] = {
                "category": row["category"],
                "color":    row["color"],
                "n":        0,
            }
            for name in models:
                group_stats[g][name] = 0.0
        group_stats[g]["n"] += 1
        for name in models:
            group_stats[g][name] += row[name]
    # Trier par xgboost
    group_stats_sorted = sorted(
        group_stats.items(),
        key=lambda x: x[1].get("xgboost", 0),
        reverse=True
    )

    # ── Serialiser ────────────────────────────────────────────────────────
    rows_json        = json.dumps(rows)
    cat_stats_json   = json.dumps(cat_stats)
    group_stats_json = json.dumps(group_stats_sorted)
    models_json      = json.dumps(list(models.keys()))
    cat_colors_json  = json.dumps(CAT_COLORS)

    # ── HTML ──────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Feature importance v4</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.26.0/plotly.min.js"></script>
<style>
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ background:{BG}; color:{TEXT}; font-family:'Courier New',monospace; padding:16px; }}
h1 {{ font-size:18px; letter-spacing:2px; margin-bottom:4px; }}
.subtitle {{ color:{SUBTEXT}; font-size:12px; margin-bottom:20px; }}
.card {{ background:{BG_PAPER}; border:1px solid #21262D; border-radius:8px;
         padding:14px; margin-bottom:16px; }}
.card-title {{ font-size:11px; color:{SUBTEXT}; letter-spacing:1px;
               margin-bottom:12px; text-transform:uppercase; }}
.tabs {{ display:flex; gap:8px; margin-bottom:16px; flex-wrap:wrap; }}
.tab {{ background:#21262D; border:1px solid #30363D; color:{SUBTEXT};
        padding:5px 14px; border-radius:4px; cursor:pointer;
        font-size:11px; font-family:inherit; }}
.tab.active {{ background:#2E75B6; border-color:#2E75B6; color:{TEXT}; }}
.section {{ display:none; }}
.section.active {{ display:block; }}

/* Table */
.feat-table {{ width:100%; border-collapse:collapse; font-size:11px; }}
.feat-table th {{ color:{SUBTEXT}; padding:6px 8px; text-align:left;
                  border-bottom:1px solid #21262D; cursor:pointer;
                  user-select:none; white-space:nowrap; }}
.feat-table th:hover {{ color:{TEXT}; }}
.feat-table td {{ padding:5px 8px; border-bottom:1px solid #0D1117; }}
.feat-table tr:hover td {{ background:#21262D; }}
.bar-cell {{ width:120px; }}
.bar-bg {{ background:#0D1117; border-radius:2px; height:12px; }}
.bar-fill {{ height:12px; border-radius:2px; transition:width 0.3s; }}
.cat-badge {{ display:inline-block; padding:1px 6px; border-radius:3px;
              font-size:9px; }}

/* Filtres */
.filters {{ display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px; align-items:center; }}
.filter-btn {{ background:#21262D; border:1px solid #30363D; color:{SUBTEXT};
               padding:3px 10px; border-radius:4px; cursor:pointer; font-size:10px; }}
.filter-btn.active {{ color:{TEXT}; }}
.search-input {{ background:#21262D; border:1px solid #30363D; color:{TEXT};
                 padding:4px 10px; border-radius:4px; font-size:11px;
                 font-family:inherit; outline:none; width:200px; }}
.search-input::placeholder {{ color:{SUBTEXT}; }}
.model-select {{ background:#21262D; border:1px solid #30363D; color:{TEXT};
                 padding:4px 10px; border-radius:4px; font-size:11px;
                 font-family:inherit; cursor:pointer; }}
</style>
</head>
<body>

<h1>Feature importance — v4</h1>
<div class="subtitle">
  Importance normalisee (gain) | XGBoost, LightGBM, Random Forest
</div>

<div class="tabs">
  <button class="tab active" onclick="showSection('overview')">Vue d'ensemble</button>
  <button class="tab" onclick="showSection('categories')">Par categorie</button>
  <button class="tab" onclick="showSection('groups')">Par groupe</button>
  <button class="tab" onclick="showSection('detail')">Detail features</button>
</div>

<!-- VUE D'ENSEMBLE -->
<div class="section active" id="section-overview">
  <div class="card">
    <div class="card-title">Top 30 features — XGBoost (gain normalise)</div>
    <div id="chart-top30" style="height:600px;"></div>
  </div>
  <div class="card">
    <div class="card-title">Comparaison top 20 features — 3 modeles</div>
    <div id="chart-compare" style="height:500px;"></div>
  </div>
</div>

<!-- PAR CATEGORIE -->
<div class="section" id="section-categories">
  <div class="card">
    <div class="card-title">Importance totale par categorie</div>
    <div id="chart-cat" style="height:350px;"></div>
  </div>
  <div class="card">
    <div class="card-title">Repartition par categorie (camembert)</div>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;">
      <div id="chart-pie-xgb" style="height:300px;"></div>
      <div id="chart-pie-lgb" style="height:300px;"></div>
      <div id="chart-pie-rf"  style="height:300px;"></div>
    </div>
  </div>
</div>

<!-- PAR GROUPE -->
<div class="section" id="section-groups">
  <div class="card">
    <div class="card-title">Importance par groupe semantique</div>
    <div id="chart-groups" style="height:450px;"></div>
  </div>
</div>

<!-- DETAIL FEATURES -->
<div class="section" id="section-detail">
  <div class="card">
    <div class="card-title">Toutes les features</div>
    <div class="filters">
      <input class="search-input" type="text" id="search-input"
             placeholder="Rechercher une feature..." oninput="filterTable()">
      <select class="model-select" id="sort-model" onchange="sortTable()">
        <option value="xgboost">Trier par XGBoost</option>
        <option value="lightgbm">Trier par LightGBM</option>
        <option value="random_forest">Trier par Random Forest</option>
      </select>
      <span style="font-size:10px;color:{SUBTEXT}">Categorie :</span>
      <button class="filter-btn active" onclick="filterCat('all', this)">Toutes</button>
      <button class="filter-btn" onclick="filterCat('calendaire', this)"
              style="border-color:#2E75B6">Calendaire</button>
      <button class="filter-btn" onclick="filterCat('lag_load', this)"
              style="border-color:#27AE60">Lag load</button>
      <button class="filter-btn" onclick="filterCat('lag_meteo', this)"
              style="border-color:#FFC000">Lag meteo</button>
      <button class="filter-btn" onclick="filterCat('prev_meteo', this)"
              style="border-color:#E74C3C">Prev meteo</button>
    </div>
    <div style="overflow-x:auto;max-height:600px;overflow-y:auto;">
      <table class="feat-table" id="feat-table">
        <thead>
          <tr>
            <th onclick="sortByCol('rank')">#</th>
            <th onclick="sortByCol('feature')">Feature</th>
            <th onclick="sortByCol('category')">Categorie</th>
            <th onclick="sortByCol('group')">Groupe</th>
            <th onclick="sortByCol('xgboost')">XGBoost</th>
            <th class="bar-cell">XGB bar</th>
            <th onclick="sortByCol('lightgbm')">LightGBM</th>
            <th class="bar-cell">LGB bar</th>
            <th onclick="sortByCol('random_forest')">RF</th>
            <th class="bar-cell">RF bar</th>
          </tr>
        </thead>
        <tbody id="feat-tbody"></tbody>
      </table>
    </div>
    <div style="margin-top:8px;font-size:10px;color:{SUBTEXT}" id="row-count"></div>
  </div>
</div>

<script>
const ROWS        = {rows_json};
const CAT_STATS   = {cat_stats_json};
const GROUP_STATS = {group_stats_json};
const MODELS      = {models_json};
const CAT_COLORS  = {cat_colors_json};

const LAYOUT_BASE = {{
  paper_bgcolor:"{BG_PAPER}", plot_bgcolor:"{BG}",
  font:{{color:"{TEXT}", family:"Courier New", size:11}},
  margin:{{l:200, r:30, t:30, b:50}},
}};

let currentCatFilter = "all";
let sortCol          = "xgboost";
let sortAsc          = false;

// ── Sections ──────────────────────────────────────────────────────────────
function showSection(name) {{
  document.querySelectorAll(".section").forEach(s => s.classList.remove("active"));
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.getElementById("section-" + name).classList.add("active");
  event.target.classList.add("active");
  if (name === "overview")   renderOverview();
  if (name === "categories") renderCategories();
  if (name === "groups")     renderGroups();
  if (name === "detail")     renderTable();
}}

// ── Overview ──────────────────────────────────────────────────────────────
function renderOverview() {{
  let top30 = ROWS.slice(0, 30);

  // Top 30 XGBoost
  Plotly.react("chart-top30", [{{
    x: top30.map(r => r.xgboost),
    y: top30.map(r => r.feature),
    type: "bar", orientation: "h",
    marker: {{ color: top30.map(r => r.color), opacity: 0.85 }},
    text: top30.map(r => (r.xgboost * 100).toFixed(2) + "%"),
    textposition: "outside",
    textfont: {{ size: 10 }},
  }}], {{
    ...LAYOUT_BASE,
    margin: {{l:220, r:80, t:30, b:40}},
    xaxis: {{ gridcolor:"rgba(200,200,200,0.1)", title:"Importance (gain normalise)" }},
    yaxis: {{ autorange:"reversed" }},
    height: 600,
  }}, {{responsive:true}});

  // Comparaison 3 modeles top 20
  let top20 = ROWS.slice(0, 20);
  let traces = MODELS.map(m => ({{
    x: top20.map(r => r[m]),
    y: top20.map(r => r.feature),
    name: m.replace("_", " "),
    type: "bar", orientation: "h",
    marker: {{ color: m === "xgboost" ? "#2E75B6" :
                       m === "lightgbm" ? "#FFC000" : "#E74C3C",
               opacity: 0.8 }},
  }}));
  Plotly.react("chart-compare", traces, {{
    ...LAYOUT_BASE,
    barmode: "group",
    margin: {{l:220, r:30, t:30, b:40}},
    xaxis: {{ gridcolor:"rgba(200,200,200,0.1)" }},
    yaxis: {{ autorange:"reversed" }},
    height: 500,
    legend: {{ bgcolor:"rgba(0,0,0,0.4)", bordercolor:"#30363D", borderwidth:1 }},
  }}, {{responsive:true}});
}}

// ── Categories ────────────────────────────────────────────────────────────
function renderCategories() {{
  let cats  = Object.keys(CAT_COLORS);
  let cnames = {{"calendaire":"Calendaire","lag_load":"Lag load",
                 "lag_meteo":"Lag meteo hist","prev_meteo":"Prev meteo"}};

  // Bar par categorie
  let traces = MODELS.map(m => ({{
    x: cats.map(c => CAT_STATS[c][m]),
    y: cats.map(c => cnames[c] || c),
    name: m.replace("_"," "),
    type:"bar", orientation:"h",
    marker:{{ color: m==="xgboost"?"#2E75B6":m==="lightgbm"?"#FFC000":"#E74C3C",
               opacity:0.8 }},
  }}));
  Plotly.react("chart-cat", traces, {{
    ...LAYOUT_BASE, barmode:"group",
    margin:{{l:130,r:30,t:30,b:40}}, height:350,
    xaxis:{{gridcolor:"rgba(200,200,200,0.1)",title:"Importance totale"}},
    yaxis:{{autorange:"reversed"}},
    legend:{{bgcolor:"rgba(0,0,0,0.4)",bordercolor:"#30363D",borderwidth:1}},
  }}, {{responsive:true}});

  // Camemberts
  [{{"id":"chart-pie-xgb","model":"xgboost","title":"XGBoost"}},
   {{"id":"chart-pie-lgb","model":"lightgbm","title":"LightGBM"}},
   {{"id":"chart-pie-rf","model":"random_forest","title":"Random Forest"}}
  ].forEach(cfg => {{
    Plotly.react(cfg.id, [{{
      values: cats.map(c => CAT_STATS[c][cfg.model]),
      labels: cats.map(c => cnames[c] || c),
      type:"pie",
      marker:{{colors: cats.map(c => CAT_COLORS[c])}},
      textinfo:"label+percent",
      textfont:{{size:10}},
      hole:0.3,
    }}], {{
      paper_bgcolor:"{BG_PAPER}",
      font:{{color:"{TEXT}",family:"Courier New",size:10}},
      margin:{{l:10,r:10,t:30,b:10}},
      title:{{text:cfg.title,font:{{size:12}}}},
      showlegend:false,
    }}, {{responsive:true}});
  }});
}}

// ── Groupes ───────────────────────────────────────────────────────────────
function renderGroups() {{
  let groups = GROUP_STATS.map(([name, info]) => ({{
    name, ...info
  }}));

  let traces = MODELS.map(m => ({{
    x: groups.map(g => g[m]),
    y: groups.map(g => g.name),
    name: m.replace("_"," "),
    type:"bar", orientation:"h",
    marker:{{ color: m==="xgboost"?"#2E75B6":m==="lightgbm"?"#FFC000":"#E74C3C",
               opacity:0.8 }},
  }}));
  Plotly.react("chart-groups", traces, {{
    ...LAYOUT_BASE, barmode:"group",
    margin:{{l:160,r:30,t:30,b:40}}, height:450,
    xaxis:{{gridcolor:"rgba(200,200,200,0.1)",title:"Importance totale"}},
    yaxis:{{autorange:"reversed"}},
    legend:{{bgcolor:"rgba(0,0,0,0.4)",bordercolor:"#30363D",borderwidth:1}},
  }}, {{responsive:true}});
}}

// ── Table detail ──────────────────────────────────────────────────────────
function renderTable() {{
  let search  = (document.getElementById("search-input")?.value || "").toLowerCase();
  let filtered = ROWS.filter(r => {{
    let catOk  = currentCatFilter === "all" || r.category === currentCatFilter;
    let srchOk = r.feature.toLowerCase().includes(search);
    return catOk && srchOk;
  }});

  filtered.sort((a, b) => {{
    let av = a[sortCol] ?? 0;
    let bv = b[sortCol] ?? 0;
    if (typeof av === "string") return sortAsc ? av.localeCompare(bv) : bv.localeCompare(av);
    return sortAsc ? av - bv : bv - av;
  }});

  let maxVals = {{}};
  MODELS.forEach(m => {{
    maxVals[m] = Math.max(...ROWS.map(r => r[m] || 0));
  }});

  let cnames = {{"calendaire":"Calendaire","lag_load":"Lag load",
                 "lag_meteo":"Lag meteo","prev_meteo":"Prev meteo"}};

  let html = "";
  filtered.forEach((r, i) => {{
    let rank = ROWS.indexOf(r) + 1;
    html += `<tr>
      <td style="color:{SUBTEXT}">${{rank}}</td>
      <td style="font-size:10px;max-width:220px;word-break:break-all">${{r.feature}}</td>
      <td><span class="cat-badge"
          style="background:${{r.color}}22;color:${{r.color}};border:1px solid ${{r.color}}44">
        ${{cnames[r.category] || r.category}}
      </span></td>
      <td style="color:{SUBTEXT};font-size:10px">${{r.group}}</td>`;

    MODELS.forEach(m => {{
      let val  = r[m] || 0;
      let pct  = maxVals[m] > 0 ? (val / maxVals[m] * 100) : 0;
      let col  = m === "xgboost" ? "#2E75B6" : m === "lightgbm" ? "#FFC000" : "#E74C3C";
      html += `<td style="text-align:right">${{(val*100).toFixed(3)}}%</td>
               <td class="bar-cell">
                 <div class="bar-bg">
                   <div class="bar-fill" style="width:${{pct}}%;background:${{col}}"></div>
                 </div>
               </td>`;
    }});
    html += "</tr>";
  }});

  document.getElementById("feat-tbody").innerHTML = html;
  document.getElementById("row-count").textContent =
    `${{filtered.length}} features affichees sur ${{ROWS.length}} total`;
}}

function filterTable() {{ renderTable(); }}

function filterCat(cat, btn) {{
  currentCatFilter = cat;
  document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  renderTable();
}}

function sortByCol(col) {{
  if (sortCol === col) sortAsc = !sortAsc;
  else {{ sortCol = col; sortAsc = false; }}
  renderTable();
}}

function sortTable() {{
  sortCol = document.getElementById("sort-model").value;
  sortAsc = false;
  renderTable();
}}

// Init
renderOverview();
</script>
</body>
</html>"""

    FILE_OUT.write_text(html, encoding="utf-8")
    size_mb = FILE_OUT.stat().st_size / 1024 / 1024
    print(f"\n-> {FILE_OUT}")
    print(f"   {size_mb:.1f} MB")

    # ── Resume console ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("TOP 10 FEATURES PAR MODELE")
    print(f"{'='*60}")
    for name in models:
        print(f"\n  {name.upper()} :")
        top10 = sorted(rows, key=lambda r: r.get(name, 0), reverse=True)[:10]
        for i, r in enumerate(top10, 1):
            print(f"    {i:>2}. {r['feature']:<45} {r[name]*100:.3f}%")