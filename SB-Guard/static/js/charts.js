/* SB-Guard — Plotly chart helpers + page controllers.
   All charts use the PDO dark theme and share a common layout factory. */

const T = window.SBG_THEME || {
  bg_navy: "#0a1628", panel: "#10233d", accent: "#ff6b35",
  green: "#2ecc71", yellow: "#f1c40f", red: "#e74c3c", text: "#e6edf5",
};
const MUTED = "#8aa0bd";

function baseLayout(extra = {}) {
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: T.text, family: "Segoe UI, Inter, sans-serif", size: 12 },
    margin: { l: 54, r: 18, t: 28, b: 44 },
    xaxis: { gridcolor: "#1d3454", zerolinecolor: "#1d3454" },
    yaxis: { gridcolor: "#1d3454", zerolinecolor: "#1d3454" },
    legend: { orientation: "h", y: 1.15, font: { size: 11 } },
    hovermode: "x unified",
  }, extra);
}
const CONF = { responsive: true, displayModeBar: false };

function color(band) {
  return { CRITICAL: T.red, WARNING: T.yellow, NORMAL: T.green }[band] || T.green;
}
async function getJSON(url) { const r = await fetch(url); return r.json(); }

/* =================================================================== */
/* DASHBOARD                                                            */
/* =================================================================== */
async function initDashboard() {
  await refreshDashboard();
  setInterval(refreshDashboard, 15000);
}

async function refreshDashboard() {
  const [kpi, fleet, health] = await Promise.all([
    getJSON("/api/kpis"), getJSON("/api/fleet"), getJSON("/api/fleet_health?hours=24"),
  ]);

  setText("kpiAlerts", kpi.active_alerts);
  setText("kpiAlertsSub", `${kpi.critical} critical · ${kpi.warning} warning`);
  setText("kpiWells", kpi.wells_monitored);
  setText("kpiPrevented", kpi.failures_prevented);
  setText("kpiSaved", kpi.estimated_cost_saved_fmt);
  setText("kpiHealth", kpi.fleet_health + "%");
  setText("simTime", "Sim time: " + new Date(kpi.current_time).toLocaleString());

  // Well grid
  const grid = document.getElementById("wellGrid");
  if (grid) {
    grid.innerHTML = fleet.wells.map(w => `
      <div class="well-card ${w.color}" onclick="location.href='/inspector?well=${w.well_id}'">
        <div class="wid">${w.well_id}</div>
        <div class="score">${w.score}%</div>
        <div class="meta">MSI ${w.msi}% · SPM ${w.spm}</div>
        <div class="bar"><i style="width:${w.score}%;background:${color(w.band)}"></i></div>
      </div>`).join("");
  }

  // Fleet health line
  Plotly.react("fleetHealthChart", [{
    x: health.t, y: health.health, mode: "lines", fill: "tozeroy",
    line: { color: T.accent, width: 2 }, fillcolor: "rgba(255,107,53,0.12)",
    name: "Fleet Health", hovertemplate: "%{y:.1f}%<extra></extra>",
  }], baseLayout({
    yaxis: { title: "Health Score (%)", range: [0, 100], gridcolor: "#1d3454" },
    xaxis: { title: "Time (last 24h)", gridcolor: "#1d3454" },
  }), CONF);
}

/* =================================================================== */
/* WELL INSPECTOR                                                       */
/* =================================================================== */
let _wellTimer = null;
async function initInspector() {
  const sel = document.getElementById("wellSelect");
  const url = new URLSearchParams(location.search);
  if (url.get("well")) sel.value = url.get("well");
  sel.addEventListener("change", () => loadWell(sel.value));
  await loadWell(sel.value);
  if (_wellTimer) clearInterval(_wellTimer);
  _wellTimer = setInterval(() => loadWell(sel.value), 20000);
}

async function loadWell(wid) {
  const d = await getJSON("/api/well/" + wid);
  if (!d.well_id) return;

  setText("wiScore", d.score + "%");
  setText("wiBand", d.band);
  document.getElementById("wiBand").className = "badge " + d.band;
  setText("wiMsiVal", d.msi + "%");

  // Horizon bars
  ["24h", "48h", "72h"].forEach(h => {
    const v = d.horizons[h];
    const fill = document.getElementById("hz" + h);
    if (fill) { fill.style.width = v + "%"; }
    setText("hzv" + h, v + "%");
  });

  setText("wiReco", d.recommendation);
  document.getElementById("wiReco").style.borderLeftColor = color(d.band);

  // Stats
  const s = d.stats;
  setText("stPprl", Number(s.pprl).toLocaleString() + " lb");
  setText("stMprl", Number(s.mprl).toLocaleString() + " lb");
  setText("stPrlr", Number(s.prlr).toLocaleString() + " lb");
  setText("stSca", s.sca);
  setText("stFill", s.fillage + "%");
  setText("stTemp", s.gearbox_temp + " °C");
  setText("stAnom", s.if_anomaly + "%");

  drawDynacard(d.dynacard);
  drawGauge(d.msi);
  drawWellSeries(d.timeseries);
}

function drawDynacard(dc) {
  const traces = [
    { x: dc.position, y: dc.baseline_load, mode: "lines", name: "Baseline Card",
      line: { color: MUTED, width: 2, dash: "dot" } },
    { x: dc.position, y: dc.current_load, mode: "lines", name: "Current Card",
      line: { color: T.accent, width: 3 }, fill: "tonexty",
      fillcolor: "rgba(255,107,53,0.08)" },
  ];
  Plotly.react("dynacard", traces, baseLayout({
    xaxis: { title: "Polished Rod Position (%)", gridcolor: "#1d3454" },
    yaxis: { title: "Load (lb)", gridcolor: "#1d3454" },
    hovermode: "closest",
  }), CONF);
}

function drawGauge(msi) {
  Plotly.react("msiGauge", [{
    type: "indicator", mode: "gauge+number", value: msi,
    number: { suffix: "%", font: { size: 30, color: T.text } },
    gauge: {
      axis: { range: [0, 100], tickcolor: MUTED },
      bar: { color: msi >= 70 ? T.red : msi >= 45 ? T.yellow : T.green },
      bgcolor: "rgba(0,0,0,0.2)", borderwidth: 0,
      steps: [
        { range: [0, 45], color: "rgba(46,204,113,0.15)" },
        { range: [45, 70], color: "rgba(241,196,15,0.18)" },
        { range: [70, 100], color: "rgba(231,76,60,0.2)" },
      ],
      threshold: { line: { color: T.accent, width: 3 }, value: 70 },
    },
  }], baseLayout({ margin: { l: 24, r: 24, t: 10, b: 10 } }), CONF);
}

function drawWellSeries(ts) {
  const L = baseLayout({ margin: { l: 50, r: 14, t: 24, b: 36 }, showlegend: false });
  Plotly.react("chSpm", [{ x: ts.t, y: ts.spm, mode: "lines",
    line: { color: "#4fa3ff", width: 2 } }],
    Object.assign({}, L, { yaxis: { title: "SPM", gridcolor: "#1d3454" } }), CONF);
  Plotly.react("chVib", [{ x: ts.t, y: ts.lvi, mode: "lines",
    line: { color: T.yellow, width: 2 } }],
    Object.assign({}, L, { yaxis: { title: "Vibration (LVI)", gridcolor: "#1d3454" } }), CONF);
  Plotly.react("chFric", [{ x: ts.t, y: ts.friction, mode: "lines",
    line: { color: "#b07bff", width: 2 } }],
    Object.assign({}, L, { yaxis: { title: "SB Friction", gridcolor: "#1d3454" } }), CONF);
  Plotly.react("chRisk", [{ x: ts.t, y: ts.risk, mode: "lines", fill: "tozeroy",
    line: { color: T.accent, width: 2 }, fillcolor: "rgba(255,107,53,0.12)" }],
    Object.assign({}, L, { yaxis: { title: "Risk (%)", range: [0, 100], gridcolor: "#1d3454" } }), CONF);
}

/* =================================================================== */
/* ALERTS                                                               */
/* =================================================================== */
async function initAlerts() {
  await refreshAlerts();
  setInterval(refreshAlerts, 15000);
}
async function refreshAlerts() {
  const d = await getJSON("/api/alerts");
  setText("alCount", d.counts.active_total);
  const at = document.getElementById("activeTable");
  if (at) {
    at.innerHTML = d.active.length ? d.active.map(a => `
      <tr>
        <td><b>${a.well_id}</b></td>
        <td>${a.alert_type}</td>
        <td><span class="badge ${a.severity}">${a.severity}</span></td>
        <td>${(a.score*100).toFixed(1)}%</td>
        <td>${a.lead_time_h != null ? a.lead_time_h + " h" : "—"}</td>
        <td>${fmtTime(a.detected_at)}</td>
        <td class="muted" style="max-width:280px">${shorten(a.recommended_action)}</td>
      </tr>`).join("")
      : `<tr><td colspan="7" class="loading">No active alerts — fleet nominal.</td></tr>`;
  }
  const ht = document.getElementById("historyTable");
  if (ht) {
    ht.innerHTML = d.history.length ? d.history.map(a => `
      <tr>
        <td><b>${a.well_id}</b></td>
        <td><span class="badge ${a.severity}">${a.severity}</span></td>
        <td>${fmtTime(a.detected_at)}</td>
        <td>${a.lead_time_h != null ? a.lead_time_h + " h" : "—"}</td>
        <td>${outcomeBadge(a.outcome)}</td>
      </tr>`).join("")
      : `<tr><td colspan="5" class="loading">No historical alerts yet.</td></tr>`;
  }
}
function outcomeBadge(o) {
  if (!o) return `<span class="muted">pending</span>`;
  const label = { confirmed: "Confirmed Failure", averted: "Averted",
                  false_alarm: "False Alarm" }[o] || o;
  return `<span class="badge ${o}">${label}</span>`;
}

/* =================================================================== */
/* ANALYTICS                                                            */
/* =================================================================== */
async function initAnalytics() {
  const [m, trends] = await Promise.all([getJSON("/api/metrics"), getJSON("/api/trends")]);
  const cls = (m.metrics.classification) || {};
  const tim = (m.metrics.timing) || {};
  setText("mPrec", fmt(cls.precision));
  setText("mRec", fmt(cls.recall));
  setText("mF1", fmt(cls.f1));
  setText("mAuc", fmt(cls.roc_auc));
  setText("mFar", fmt(cls.false_alarm_rate));
  setText("mLead", (tim.mean_lead_time_h || 0).toFixed(1) + " h");
  setText("mMttd", (tim.mttd_h || 0).toFixed(1) + " h");
  setText("mDet", ((tim.detection_rate || 0) * 100).toFixed(0) + "%");

  // ROC curve
  const roc = m.metrics.roc || { fpr: [0, 1], tpr: [0, 1], auc: 0 };
  Plotly.react("rocChart", [
    { x: roc.fpr, y: roc.tpr, mode: "lines", name: `ROC (AUC=${fmt(roc.auc)})`,
      line: { color: T.accent, width: 3 }, fill: "tozeroy",
      fillcolor: "rgba(255,107,53,0.1)" },
    { x: [0, 1], y: [0, 1], mode: "lines", name: "Random",
      line: { color: MUTED, width: 1, dash: "dash" } },
  ], baseLayout({
    xaxis: { title: "False Positive Rate", range: [0, 1], gridcolor: "#1d3454" },
    yaxis: { title: "True Positive Rate", range: [0, 1], gridcolor: "#1d3454" },
    hovermode: "closest",
  }), CONF);

  // Feature importance
  const fi = m.metrics.feature_importance || { features: [], importance: [] };
  const topF = fi.features.slice(0, 12).reverse();
  const topI = fi.importance.slice(0, 12).reverse();
  Plotly.react("featChart", [{
    type: "bar", orientation: "h", x: topI, y: topF,
    marker: { color: T.accent }, hovertemplate: "%{x:.3f}<extra>%{y}</extra>",
  }], baseLayout({
    margin: { l: 130, r: 18, t: 20, b: 40 },
    xaxis: { title: "Relative Importance", gridcolor: "#1d3454" },
  }), CONF);

  // Monthly failure trend
  const mf = trends.monthly_failures || { months: [], failures: [] };
  Plotly.react("trendChart", [{
    type: "bar", x: mf.months, y: mf.failures, marker: { color: T.red },
    name: "SB Failures", hovertemplate: "%{y} failures<extra>%{x}</extra>",
  }], baseLayout({
    yaxis: { title: "Failures / month", gridcolor: "#1d3454" },
    xaxis: { title: "Month", gridcolor: "#1d3454" },
  }), CONF);

  // Confusion matrix heatmap
  const cm = cls.confusion_matrix || { tn: 0, fp: 0, fn: 0, tp: 0 };
  Plotly.react("cmChart", [{
    type: "heatmap",
    z: [[cm.tn, cm.fp], [cm.fn, cm.tp]],
    x: ["Pred Normal", "Pred Leak"], y: ["Actual Normal", "Actual Leak"],
    colorscale: [[0, T.panel], [1, T.accent]], showscale: false,
    text: [[`TN\n${cm.tn}`, `FP\n${cm.fp}`], [`FN\n${cm.fn}`, `TP\n${cm.tp}`]],
    texttemplate: "%{text}", textfont: { size: 16, color: T.text },
    hovertemplate: "%{y} / %{x}: %{z}<extra></extra>",
  }], baseLayout({ margin: { l: 100, r: 18, t: 20, b: 50 } }), CONF);

  // At-risk ranking
  const rank = await getJSON("/api/at_risk?top=10");
  const rt = document.getElementById("riskRank");
  if (rt) rt.innerHTML = rank.ranking.map((w, i) => `
    <tr><td>${i + 1}</td><td><b>${w.well_id}</b></td>
    <td><span class="badge ${w.band}">${w.band}</span></td>
    <td>${w.score}%</td><td>${w.msi}%</td></tr>`).join("");

  setText("savTotal", "$" + Number(trends.savings.total_savings_usd).toLocaleString());
  setText("savEvents", trends.savings.failures_prevented);
  setText("savBbl", Number(trends.savings.deferred_oil_avoided_bbl).toLocaleString() + " bbl");
}

/* =================================================================== */
/* MODEL MANAGEMENT                                                     */
/* =================================================================== */
async function initModel() {
  const [m, reg, dq] = await Promise.all([
    getJSON("/api/metrics"), getJSON("/api/registry"), getJSON("/api/data_quality"),
  ]);
  // threshold slider
  const sl = document.getElementById("thr");
  sl.value = Math.round(m.threshold * 100);
  setText("thrVal", sl.value + "%");
  sl.addEventListener("input", () => setText("thrVal", sl.value + "%"));
  sl.addEventListener("change", async () => {
    await fetch("/api/threshold", { method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ threshold: sl.value / 100 }) });
  });

  // version table
  const vt = document.getElementById("verTable");
  vt.innerHTML = (reg.versions || []).slice().reverse().map(v => `
    <tr><td><b>${v.version}</b></td><td>${fmtTime(v.trained_at)}</td>
    <td>${v.backend}</td><td>${fmt(v.roc_auc)}</td><td>${fmt(v.f1)}</td>
    <td>${fmt(v.precision)}</td><td>${fmt(v.recall)}</td></tr>`).join("")
    || `<tr><td colspan="7" class="loading">No versions recorded.</td></tr>`;

  // data quality
  setText("dqRows", Number(dq.raw_rows || dq.scored_rows || 0).toLocaleString());
  setText("dqComplete", (dq.completeness_pct != null ? dq.completeness_pct : 100) + "%");
  setText("dqMissing", (dq.missing_pct != null ? dq.missing_pct : 0) + "%");
  setText("dqWells", dq.wells);

  document.getElementById("retrainBtn").addEventListener("click", retrain);
}

async function retrain() {
  const btn = document.getElementById("retrainBtn");
  btn.disabled = true; btn.textContent = "Retraining…";
  await fetch("/api/retrain", { method: "POST" });
  const bar = document.getElementById("retrainBar");
  const poll = setInterval(async () => {
    const s = await getJSON("/api/retrain/status");
    bar.style.width = s.progress + "%";
    setText("retrainMsg", s.message);
    if (!s.running && s.progress >= 100) {
      clearInterval(poll); btn.disabled = false; btn.textContent = "Retrain Model";
      setTimeout(() => location.reload(), 1200);
    } else if (!s.running && s.progress === 0) {
      clearInterval(poll); btn.disabled = false; btn.textContent = "Retrain Model";
    }
  }, 1500);
}

/* =================================================================== */
/* utils                                                               */
/* =================================================================== */
function setText(id, v) { const e = document.getElementById(id); if (e) e.textContent = v; }
function fmt(x) { return (x == null || isNaN(x)) ? "—" : Number(x).toFixed(3); }
function fmtTime(s) { try { return new Date(s).toLocaleString(); } catch { return s; } }
function shorten(s, n = 90) { return s && s.length > n ? s.slice(0, n) + "…" : (s || ""); }
