"""
Feature importance — sliding window LightGBM toutes iterations
================================================================
Lit  : data/features_v6.parquet
       data/models/sw_hyperparams_lgb.parquet
Ecrit: data/feature_importance_sw_lgb_all.html

Reentraine le modele de chaque iteration avec les hyperparametres
Optuna sauvegardes, puis calcule l'importance des features.
Permet de visualiser l'evolution de l'importance par iteration.
"""

import json
import datetime
import polars as pl
import numpy as np
import lightgbm as lgb

from config import DATA_DIR

MODELS_DIR = DATA_DIR / "models"
FILE_FEAT  = DATA_DIR / "features_v6.parquet"
FILE_OUT   = DATA_DIR / "feature_importance_sw_lgb_all.html"

TZ_LOCAL = "Europe/Zurich"

BG       = "#0F1117"
BG_PAPER = "#161B22"
TEXT     = "#E6EDF3"
SUBTEXT  = "#8B949E"

CAT_COLORS = {
    "calendaire": "#2E75B6",
    "lag_load":   "#27AE60",
    "lag_pv":     "#E74C3C",
    "lag_meteo":  "#FFC000",
    "prev_meteo": "#9B59B6",
}

TRAIN_MONTHS = 22
VAL_MONTHS   = 4


def add_months(d: datetime.date, n: int) -> datetime.date:
    month = d.month + n
    year  = d.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    return datetime.date(year, month, 1)


def get_category(feat: str) -> str:
    if feat.startswith("lag_") and "pred_" in feat:
        return "prev_meteo"
    elif "pv_" in feat:
        return "lag_pv"
    elif feat.startswith("feat_load"):
        return "lag_load"
    elif feat.startswith("feat_"):
        return "lag_meteo"
    else:
        return "calendaire"


def get_group(feat: str) -> str:
    if "temperature" in feat: return "temperature"
    if "radiation"   in feat: return "radiation"
    if "sunshine"    in feat: return "sunshine"
    if "humidity"    in feat: return "humidity"
    if "pv_central"  in feat: return "pv_central_valais"
    if "pv_sierre"   in feat: return "pv_sierre"
    if "pv_sion"     in feat: return "pv_sion"
    if "load"        in feat: return "load"
    if "hour"        in feat: return "heure"
    if "day_of_week" in feat: return "jour_semaine"
    if "month"       in feat: return "mois"
    if "day_of_year" in feat: return "jour_annee"
    if "weekend"     in feat: return "weekend"
    if "holiday"     in feat: return "ferie"
    if "school"      in feat: return "vacances_scolaires"
    return "autre"


if __name__ == "__main__":

    # ── Chargement ────────────────────────────────────────────────────────
    print("Chargement features v6...")
    df = pl.read_parquet(FILE_FEAT).sort("date_locale")
    feat_cols = [c for c in df.columns
                 if c not in ("timestamp", "date_locale", "target", "split")]
    print(f"  {len(feat_cols)} features")

    print("Chargement hyperparametres SW...")
    params_df = pl.read_parquet(MODELS_DIR / "sw_hyperparams_lgb.parquet").sort("iter")
    n_iters   = params_df.shape[0]
    print(f"  {n_iters} iterations")

    # ── Boucle sur toutes les iterations ─────────────────────────────────
    all_iter_data = []

    for row_params in params_df.iter_rows(named=True):
        iter_num       = row_params["iter"]
        test_month_str = row_params["test_month"]

        print(f"\nIter {iter_num:>2}/{n_iters} — test {test_month_str}...")

        # Fenetres
        parts       = test_month_str.split("-")
        test_start  = datetime.date(int(parts[0]), int(parts[1]), 1)
        val_start   = add_months(test_start, -VAL_MONTHS)
        train_start = add_months(val_start,  -TRAIN_MONTHS)

        # Filtrer train+val
        df_trainval = df.filter(
            (pl.col("date_locale") >= train_start) &
            (pl.col("date_locale") < test_start)
        )
        X = df_trainval.select(feat_cols).to_numpy()
        y = df_trainval["target"].to_numpy()
        print(f"  Train+val : {X.shape[0]:,} pts ({X.shape[0]//96} jours)")

        # Hyperparametres
        best_params = {
            "n_estimators":      row_params["n_estimators"],
            "learning_rate":     row_params["learning_rate"],
            "max_depth":         row_params["max_depth"],
            "num_leaves":        row_params["num_leaves"],
            "min_child_samples": row_params["min_child_samples"],
            "subsample":         row_params["subsample"],
            "colsample_bytree":  row_params["colsample_bytree"],
            "reg_alpha":         row_params["reg_alpha"],
            "reg_lambda":        row_params["reg_lambda"],
            "n_jobs":            -1,
            "random_state":      42,
            "verbose":           -1,
        }

        # Entrainement
        model = lgb.LGBMRegressor(**best_params)
        model.fit(X, y)

        # Importances
        imp_vals = model.booster_.feature_importance(importance_type="gain")
        total    = imp_vals.sum() or 1
        importances = {
            f: float(v / total)
            for f, v in zip(feat_cols, imp_vals)
        }

        # Stats par categorie
        cat_totals = {}
        for cat in CAT_COLORS:
            cat_totals[cat] = round(sum(
                importances.get(f, 0)
                for f in feat_cols
                if get_category(f) == cat
            ), 4)

        # Stats par groupe
        group_totals = {}
        for f in feat_cols:
            g = get_group(f)
            if g not in group_totals:
                group_totals[g] = {
                    "importance": 0.0,
                    "category":   get_category(f),
                    "color":      CAT_COLORS[get_category(f)],
                }
            group_totals[g]["importance"] += importances.get(f, 0)

        # Features triees par importance
        feat_sorted = sorted(
            [{"feature": f,
              "importance": round(importances[f], 6),
              "category":   get_category(f),
              "group":      get_group(f),
              "color":      CAT_COLORS[get_category(f)]}
             for f in feat_cols],
            key=lambda r: r["importance"], reverse=True
        )

        all_iter_data.append({
            "iter":        iter_num,
            "test_month":  test_month_str,
            "train_start": str(train_start),
            "val_start":   str(val_start),
            "mae_val":     round(row_params["mae_val"], 5),
            "n_trainval":  int(X.shape[0]),
            "features":    feat_sorted,
            "cat_totals":  cat_totals,
            "group_totals":sorted(
                [{"group": g, **v} for g, v in group_totals.items()],
                key=lambda x: x["importance"], reverse=True
            ),
            "params": {k: v for k, v in best_params.items()
                       if k not in ("n_jobs", "random_state", "verbose")},
        })

        n_nz = sum(1 for v in importances.values() if v > 0)
        print(f"  Features utilisees : {n_nz}/{len(feat_cols)}")
        print(f"  Top 3 : " + ", ".join(
            f"{r['feature']} ({r['importance']*100:.2f}%)"
            for r in feat_sorted[:3]
        ))

    # ── Resume console ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("EVOLUTION IMPORTANCE PAR CATEGORIE")
    print(f"{'='*60}")
    print(f"  {'Iter':<6} {'Mois test':<12} {'Lag load':>10} "
          f"{'Lag PV':>10} {'Prev meteo':>10} {'Calendaire':>10}")
    print(f"  {'-'*60}")
    for d in all_iter_data:
        print(f"  {d['iter']:<6} {d['test_month']:<12} "
              f"{d['cat_totals']['lag_load']*100:>9.1f}% "
              f"{d['cat_totals']['lag_pv']*100:>9.1f}% "
              f"{d['cat_totals']['prev_meteo']*100:>9.1f}% "
              f"{d['cat_totals']['calendaire']*100:>9.1f}%")

    # ── JSON ──────────────────────────────────────────────────────────────
    all_iter_json  = json.dumps(all_iter_data)
    cat_colors_json = json.dumps(CAT_COLORS)

    # ── HTML ──────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Feature importance SW LightGBM — toutes iterations</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.26.0/plotly.min.js"></script>
<style>
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ background:{BG}; color:{TEXT}; font-family:'Courier New',monospace; padding:16px; }}
h1 {{ font-size:18px; letter-spacing:2px; margin-bottom:4px; }}
.subtitle {{ color:{SUBTEXT}; font-size:12px; margin-bottom:20px; }}
.layout {{ display:grid; grid-template-columns:1fr 300px; gap:16px; }}
.left  {{ display:flex; flex-direction:column; gap:12px; }}
.right {{ display:flex; flex-direction:column; gap:12px; }}
.card {{ background:{BG_PAPER}; border:1px solid #21262D;
         border-radius:8px; padding:14px; }}
.card-title {{ font-size:11px; color:{SUBTEXT}; letter-spacing:1px;
               margin-bottom:10px; text-transform:uppercase; }}
.tabs {{ display:flex; gap:8px; margin-bottom:16px; flex-wrap:wrap; }}
.tab {{ background:#21262D; border:1px solid #30363D; color:{SUBTEXT};
        padding:5px 14px; border-radius:4px; cursor:pointer; font-size:11px; }}
.tab.active {{ background:#2E75B6; border-color:#2E75B6; color:{TEXT}; }}
.section {{ display:none; }}
.section.active {{ display:block; }}
.iter-btns {{ display:flex; gap:6px; flex-wrap:wrap; margin-bottom:12px; }}
.iter-btn {{ background:#21262D; border:1px solid #30363D; color:{SUBTEXT};
             padding:4px 10px; border-radius:4px; cursor:pointer; font-size:11px; }}
.iter-btn.active {{ background:#2E75B6; border-color:#2E75B6; color:{TEXT}; }}
.feat-table {{ width:100%; border-collapse:collapse; font-size:11px; }}
.feat-table th {{ color:{SUBTEXT}; padding:6px 8px; text-align:left;
                  border-bottom:1px solid #21262D; white-space:nowrap; }}
.feat-table td {{ padding:5px 8px; border-bottom:1px solid #0D1117; }}
.feat-table tr:hover td {{ background:#21262D; }}
.bar-bg {{ background:#0D1117; border-radius:2px; height:10px; width:120px; }}
.bar-fill {{ height:10px; border-radius:2px; }}
.cat-badge {{ display:inline-block; padding:1px 5px; border-radius:3px;
              font-size:9px; }}
.filters {{ display:flex; gap:6px; flex-wrap:wrap; margin-bottom:10px;
            align-items:center; }}
.filter-btn {{ background:#21262D; border:1px solid #30363D; color:{SUBTEXT};
               padding:3px 8px; border-radius:4px; cursor:pointer; font-size:10px; }}
.filter-btn.active {{ color:{TEXT}; }}
.search-input {{ background:#21262D; border:1px solid #30363D; color:{TEXT};
                 padding:4px 10px; border-radius:4px; font-size:11px;
                 font-family:inherit; outline:none; width:180px; }}
.search-input::placeholder {{ color:{SUBTEXT}; }}
.info-item {{ background:#21262D; border-radius:6px; padding:8px;
              margin-bottom:6px; }}
.info-val {{ font-size:13px; font-weight:bold; }}
.info-lbl {{ font-size:10px; color:{SUBTEXT}; margin-top:2px; }}
</style>
</head>
<body>

<h1>Feature importance — Sliding window LightGBM (toutes iterations)</h1>
<div class="subtitle">
  10 iterations | 22 mois train + 4 mois val + 1 mois test |
  Hyperparametres Optuna | features v6
</div>

<div class="tabs">
  <button class="tab active" onclick="showSection('evolution')">Evolution categorielle</button>
  <button class="tab" onclick="showSection('evolution_group')">Evolution groupes</button>
  <button class="tab" onclick="showSection('top_features')">Top features</button>
  <button class="tab" onclick="showSection('detail')">Detail par iteration</button>
</div>

<!-- EVOLUTION CATEGORIELLE -->
<div class="section active" id="section-evolution">
  <div class="card">
    <div class="card-title">Evolution importance par categorie au fil des iterations</div>
    <div id="chart-evolution-cat" style="height:400px;"></div>
  </div>
  <div class="card">
    <div class="card-title">Repartition categorie par iteration (barres empilees)</div>
    <div id="chart-stacked-cat" style="height:350px;"></div>
  </div>
</div>

<!-- EVOLUTION GROUPES -->
<div class="section" id="section-evolution_group">
  <div class="card">
    <div class="card-title">Evolution importance par groupe au fil des iterations</div>
    <div id="chart-evolution-group" style="height:500px;"></div>
  </div>
</div>

<!-- TOP FEATURES -->
<div class="section" id="section-top_features">
  <div class="card">
    <div class="card-title">Evolution top 10 features au fil des iterations</div>
    <div id="chart-top-features" style="height:500px;"></div>
  </div>
</div>

<!-- DETAIL PAR ITERATION -->
<div class="section" id="section-detail">
  <div class="layout">
    <div class="left">

      <!-- Selection iteration -->
      <div class="card">
        <div class="card-title">Selection iteration</div>
        <div class="iter-btns" id="iter-btns"></div>
      </div>

      <!-- Top 20 features iteration selectionnee -->
      <div class="card">
        <div class="card-title">Top 20 features
          <span id="iter-label" style="color:{TEXT}"></span>
        </div>
        <div id="chart-iter-top20" style="height:500px;"></div>
      </div>

      <!-- Table features -->
      <div class="card">
        <div class="card-title">Toutes les features</div>
        <div class="filters">
          <input class="search-input" type="text" id="search-input"
                 placeholder="Rechercher..." oninput="renderDetailTable()">
          <button class="filter-btn active" onclick="filterCat('all',this)">Toutes</button>
          <button class="filter-btn" onclick="filterCat('calendaire',this)"
                  style="border-color:#2E75B6">Cal</button>
          <button class="filter-btn" onclick="filterCat('lag_load',this)"
                  style="border-color:#27AE60">Load</button>
          <button class="filter-btn" onclick="filterCat('lag_pv',this)"
                  style="border-color:#E74C3C">PV</button>
          <button class="filter-btn" onclick="filterCat('lag_meteo',this)"
                  style="border-color:#FFC000">Meteo</button>
          <button class="filter-btn" onclick="filterCat('prev_meteo',this)"
                  style="border-color:#9B59B6">Prev</button>
        </div>
        <div style="overflow-x:auto;max-height:500px;overflow-y:auto;">
          <table class="feat-table">
            <thead><tr>
              <th>#</th><th>Feature</th><th>Cat</th>
              <th>Groupe</th><th>Imp.</th><th>Bar</th>
            </tr></thead>
            <tbody id="feat-tbody"></tbody>
          </table>
        </div>
        <div style="margin-top:6px;font-size:10px;color:{SUBTEXT}"
             id="row-count"></div>
      </div>

    </div>

    <div class="right">
      <!-- Info iteration -->
      <div class="card">
        <div class="card-title">Info iteration</div>
        <div id="iter-info"></div>
      </div>

      <!-- Camembert categorie -->
      <div class="card">
        <div class="card-title">Repartition categorie</div>
        <div id="chart-iter-pie" style="height:280px;"></div>
      </div>
    </div>
  </div>
</div>

<script>
const ALL_ITERS    = {all_iter_json};
const CAT_COLORS   = {cat_colors_json};
const N_ITERS      = ALL_ITERS.length;

const CAT_NAMES = {{
  "calendaire":"Calendaire","lag_load":"Lag load",
  "lag_pv":"Lag PV","lag_meteo":"Lag meteo hist",
  "prev_meteo":"Prev meteo"
}};

let selectedIter     = N_ITERS - 1;  // derniere iteration par defaut
let currentCatFilter = "all";

const LAYOUT_BASE = {{
  paper_bgcolor:"{BG_PAPER}", plot_bgcolor:"{BG}",
  font:{{color:"{TEXT}", family:"Courier New", size:11}},
  margin:{{l:50, r:30, t:30, b:50}},
}};

function showSection(name) {{
  document.querySelectorAll(".section").forEach(s=>s.classList.remove("active"));
  document.querySelectorAll(".tab").forEach(t=>t.classList.remove("active"));
  document.getElementById("section-"+name).classList.add("active");
  event.target.classList.add("active");
  if (name==="evolution")       renderEvolutionCat();
  if (name==="evolution_group") renderEvolutionGroup();
  if (name==="top_features")    renderTopFeatures();
  if (name==="detail")          renderDetail();
}}

// ── Evolution categorielle ────────────────────────────────────────────────
function renderEvolutionCat() {{
  let months = ALL_ITERS.map(d=>d.test_month);
  let cats   = Object.keys(CAT_COLORS);

  // Lignes par categorie
  let traces = cats.map(cat => ({{
    x: months,
    y: ALL_ITERS.map(d => d.cat_totals[cat]),
    name: CAT_NAMES[cat] || cat,
    line: {{color: CAT_COLORS[cat], width:2}},
    type:"scatter", mode:"lines+markers",
    marker:{{size:7}},
  }}));

  Plotly.react("chart-evolution-cat", traces, {{
    ...LAYOUT_BASE, height:400,
    xaxis:{{gridcolor:"rgba(200,200,200,0.1)",tickangle:-30}},
    yaxis:{{gridcolor:"rgba(200,200,200,0.1)",title:"Importance totale",
            tickformat:".1%"}},
    legend:{{bgcolor:"rgba(0,0,0,0.4)",bordercolor:"#30363D",borderwidth:1}},
    hovermode:"x unified",
  }}, {{responsive:true}});

  // Barres empilees
  let tracesStacked = cats.map(cat => ({{
    x: months,
    y: ALL_ITERS.map(d => d.cat_totals[cat]),
    name: CAT_NAMES[cat] || cat,
    type:"bar",
    marker:{{color: CAT_COLORS[cat], opacity:0.85}},
  }}));

  Plotly.react("chart-stacked-cat", tracesStacked, {{
    ...LAYOUT_BASE, barmode:"stack", height:350,
    xaxis:{{gridcolor:"rgba(200,200,200,0.1)",tickangle:-30}},
    yaxis:{{gridcolor:"rgba(200,200,200,0.1)",title:"Importance",
            tickformat:".0%"}},
    legend:{{bgcolor:"rgba(0,0,0,0.4)",bordercolor:"#30363D",borderwidth:1}},
  }}, {{responsive:true}});
}}

// ── Evolution groupes ─────────────────────────────────────────────────────
function renderEvolutionGroup() {{
  let months = ALL_ITERS.map(d=>d.test_month);

  // Collecter tous les groupes uniques
  let allGroups = {{}};
  ALL_ITERS.forEach(d => {{
    d.group_totals.forEach(g => {{
      if (!allGroups[g.group]) allGroups[g.group] = {{color: g.color}};
    }});
  }});

  // Garder top 10 groupes par importance moyenne
  let groupMeans = Object.keys(allGroups).map(gname => ({{
    name: gname,
    mean: ALL_ITERS.reduce((sum,d) => {{
      let g = d.group_totals.find(x=>x.group===gname);
      return sum + (g ? g.importance : 0);
    }}, 0) / N_ITERS,
    color: allGroups[gname].color,
  }}));
  groupMeans.sort((a,b)=>b.mean-a.mean);
  let top10Groups = groupMeans.slice(0,10);

  let traces = top10Groups.map(g => ({{
    x: months,
    y: ALL_ITERS.map(d => {{
      let grp = d.group_totals.find(x=>x.group===g.name);
      return grp ? grp.importance : 0;
    }}),
    name: g.name,
    line: {{color: g.color, width:2}},
    type:"scatter", mode:"lines+markers",
    marker:{{size:6}},
  }}));

  Plotly.react("chart-evolution-group", traces, {{
    ...LAYOUT_BASE, height:500,
    xaxis:{{gridcolor:"rgba(200,200,200,0.1)",tickangle:-30}},
    yaxis:{{gridcolor:"rgba(200,200,200,0.1)",title:"Importance totale",
            tickformat:".2%"}},
    legend:{{bgcolor:"rgba(0,0,0,0.4)",bordercolor:"#30363D",borderwidth:1}},
    hovermode:"x unified",
  }}, {{responsive:true}});
}}

// ── Top features evolution ────────────────────────────────────────────────
function renderTopFeatures() {{
  // Top 10 features de la derniere iteration
  let lastIter  = ALL_ITERS[N_ITERS-1];
  let top10feat = lastIter.features.slice(0,10).map(r=>r.feature);
  let months    = ALL_ITERS.map(d=>d.test_month);

  let traces = top10feat.map(feat => {{
    let color = ALL_ITERS[N_ITERS-1].features.find(r=>r.feature===feat)?.color || "#888";
    return {{
      x: months,
      y: ALL_ITERS.map(d => {{
        let f = d.features.find(r=>r.feature===feat);
        return f ? f.importance : 0;
      }}),
      name: feat,
      line: {{color, width:2}},
      type:"scatter", mode:"lines+markers",
      marker:{{size:6}},
    }};
  }});

  Plotly.react("chart-top-features", traces, {{
    ...LAYOUT_BASE, height:500,
    xaxis:{{gridcolor:"rgba(200,200,200,0.1)",tickangle:-30}},
    yaxis:{{gridcolor:"rgba(200,200,200,0.1)",title:"Importance (gain)",
            tickformat:".2%"}},
    legend:{{bgcolor:"rgba(0,0,0,0.4)",bordercolor:"#30363D",borderwidth:1,
             font:{{size:9}}}},
    hovermode:"x unified",
  }}, {{responsive:true}});
}}

// ── Detail par iteration ──────────────────────────────────────────────────
function renderDetail() {{
  // Boutons iteration
  let div = document.getElementById("iter-btns");
  if (div.children.length === 0) {{
    ALL_ITERS.forEach((d,i) => {{
      let btn = document.createElement("button");
      btn.className = "iter-btn" + (i===selectedIter ? " active" : "");
      btn.textContent = `Iter ${{d.iter}} (${{d.test_month}})`;
      btn.onclick = () => selectIter(i);
      div.appendChild(btn);
    }});
  }}
  renderIterDetail(selectedIter);
}}

function selectIter(idx) {{
  selectedIter = idx;
  document.querySelectorAll(".iter-btn").forEach((btn,i) => {{
    btn.classList.toggle("active", i===idx);
  }});
  renderIterDetail(idx);
}}

function renderIterDetail(idx) {{
  let d = ALL_ITERS[idx];
  document.getElementById("iter-label").textContent =
    `— iter ${{d.iter}} | test ${{d.test_month}}`;

  // Top 20 barres
  let top20 = d.features.slice(0,20);
  Plotly.react("chart-iter-top20", [{{
    x: top20.map(r=>r.importance),
    y: top20.map(r=>r.feature),
    type:"bar", orientation:"h",
    marker:{{color:top20.map(r=>r.color), opacity:0.85}},
    text: top20.map(r=>(r.importance*100).toFixed(2)+"%"),
    textposition:"outside", textfont:{{size:10}},
  }}], {{
    ...LAYOUT_BASE, height:500,
    margin:{{l:230,r:80,t:20,b:30}},
    xaxis:{{gridcolor:"rgba(200,200,200,0.1)",title:"Importance"}},
    yaxis:{{autorange:"reversed"}},
  }}, {{responsive:true}});

  // Camembert
  let cats = Object.keys(CAT_COLORS);
  Plotly.react("chart-iter-pie", [{{
    values: cats.map(c=>d.cat_totals[c]),
    labels: cats.map(c=>CAT_NAMES[c]||c),
    type:"pie", hole:0.35,
    marker:{{colors:cats.map(c=>CAT_COLORS[c])}},
    textinfo:"label+percent", textfont:{{size:10}},
  }}], {{
    paper_bgcolor:"{BG_PAPER}",
    font:{{color:"{TEXT}",family:"Courier New",size:10}},
    margin:{{l:10,r:10,t:10,b:10}},
    showlegend:false,
  }}, {{responsive:true}});

  // Info iteration
  document.getElementById("iter-info").innerHTML = `
    ${{[
      ["Mois test",    d.test_month],
      ["Train debut",  d.train_start],
      ["Val debut",    d.val_start],
      ["MAE val",      d.mae_val.toFixed(5)],
      ["N train+val",  d.n_trainval.toLocaleString()],
      ["n_estimators", d.params.n_estimators],
      ["learning_rate",d.params.learning_rate.toFixed(4)],
      ["max_depth",    d.params.max_depth],
      ["num_leaves",   d.params.num_leaves],
    ].map(([lbl,val]) => `
      <div class="info-item">
        <div class="info-val">${{val}}</div>
        <div class="info-lbl">${{lbl}}</div>
      </div>`).join("")}}`;

  renderDetailTable();
}}

function renderDetailTable() {{
  let d        = ALL_ITERS[selectedIter];
  let search   = (document.getElementById("search-input")?.value||"").toLowerCase();
  let filtered = d.features.filter(r => {{
    let catOk  = currentCatFilter==="all" || r.category===currentCatFilter;
    let srchOk = r.feature.toLowerCase().includes(search);
    return catOk && srchOk;
  }});

  let maxVal = d.features[0]?.importance || 1;
  let html   = "";
  filtered.forEach((r,i) => {{
    let rank = d.features.indexOf(r)+1;
    let pct  = maxVal>0 ? (r.importance/maxVal*100) : 0;
    html += `<tr>
      <td style="color:{SUBTEXT}">${{rank}}</td>
      <td style="font-size:10px;max-width:200px;word-break:break-all">
        ${{r.feature}}</td>
      <td><span class="cat-badge"
          style="background:${{r.color}}22;color:${{r.color}};
                 border:1px solid ${{r.color}}44">
        ${{CAT_NAMES[r.category]||r.category}}</span></td>
      <td style="color:{SUBTEXT};font-size:10px">${{r.group}}</td>
      <td style="text-align:right">${{(r.importance*100).toFixed(3)}}%</td>
      <td><div class="bar-bg">
        <div class="bar-fill"
             style="width:${{pct}}%;background:${{r.color}}"></div>
      </div></td>
    </tr>`;
  }});

  document.getElementById("feat-tbody").innerHTML = html;
  document.getElementById("row-count").textContent =
    `${{filtered.length}} features sur ${{d.features.length}} total`;
}}

function filterCat(cat, btn) {{
  currentCatFilter = cat;
  document.querySelectorAll(".filter-btn").forEach(b=>b.classList.remove("active"));
  btn.classList.add("active");
  renderDetailTable();
}}

renderEvolutionCat();
</script>
</body>
</html>"""

    FILE_OUT.write_text(html, encoding="utf-8")
    size_mb = FILE_OUT.stat().st_size / 1024 / 1024
    print(f"\n-> {FILE_OUT}")
    print(f"   {size_mb:.1f} MB")