/**
 * Sigma Risk Analytics — shared JS utilities
 * Phase 10 · Chart.js helpers, API wrappers, UI utilities
 */

"use strict";

/* ─── Palette ──────────────────────────────────────────────────────────────── */
const PALETTE = {
  scarlet:   "#AE0001",
  gold:      "#D3A625",
  goldPale:  "#F0C040",
  goldFaint: "#3A2E08",
  navy:      "#080B14",
  bg800:     "#0D1220",
  bg700:     "#111827",
  bg600:     "#1A2235",
  bg500:     "#1F2A3C",
  text100:   "#F0F2F8",
  text200:   "#C8CEDE",
  text300:   "#8892A4",
  text400:   "#5A6478",

  assets: {
    spx:  "#E8455A",   /* vivid red   */
    ndx:  "#4FA8DE",   /* sky blue    */
    gold: "#D3A625",   /* brand gold  */
    btc:  "#F97316",   /* orange      */
  },

  scenarios: {
    positive: "#2EC4A0",
    negative: "#E8455A",
    neutral:  "#8892A4",
  },
};

/* ─── Chart.js global defaults ─────────────────────────────────────────────── */
if (typeof Chart !== "undefined") {
  Chart.defaults.color          = PALETTE.text300;
  Chart.defaults.font.family    = "'Inter', 'Space Grotesk', sans-serif";
  Chart.defaults.font.size      = 11;
  Chart.defaults.borderColor    = PALETTE.bg500;
  Chart.defaults.resizeDelay    = 350;   /* debounce resize — wait for sidebar/layout transitions to finish */
  Chart.defaults.plugins.legend.labels.boxWidth  = 10;
  Chart.defaults.plugins.legend.labels.padding   = 14;
  Chart.defaults.plugins.legend.labels.color     = PALETTE.text200;
  Chart.defaults.plugins.tooltip.backgroundColor = PALETTE.bg600;
  Chart.defaults.plugins.tooltip.borderColor     = PALETTE.bg500;
  Chart.defaults.plugins.tooltip.borderWidth     = 1;
  Chart.defaults.plugins.tooltip.padding         = 10;
  Chart.defaults.plugins.tooltip.titleColor      = PALETTE.text100;
  Chart.defaults.plugins.tooltip.bodyColor       = PALETTE.text200;
  Chart.defaults.plugins.tooltip.cornerRadius    = 6;
}

/* ─── Chart factory: donut ──────────────────────────────────────────────────── */
function buildAllocationDonut(canvasId, weights) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;

  const labels = Object.keys(weights).map(a => ASSET_LABELS[a] || a.toUpperCase());
  const values = Object.values(weights);
  const colors = Object.keys(weights).map(a => PALETTE.assets[a] || "#8892A4");

  if (ctx._chartInstance) ctx._chartInstance.destroy();

  ctx._chartInstance = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors,
        borderColor: PALETTE.bg800,
        borderWidth: 3,
        hoverOffset: 6,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "68%",
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            color:     PALETTE.text200,
            boxWidth:  10,
            padding:   14,
            font:      { size: 11, family: "'Inter', 'Space Grotesk', sans-serif" },
            generateLabels(chart) {
              const ds  = chart.data.datasets[0];
              return chart.data.labels.map((label, i) => ({
                text:       `${label}  ${(ds.data[i] * 100).toFixed(1)}%`,
                fillStyle:  ds.backgroundColor[i],
                strokeStyle:ds.backgroundColor[i],
                fontColor:  PALETTE.text200,
                color:      PALETTE.text200,
                lineWidth:  0,
                hidden:     false,
                index:      i,
              }));
            },
          },
        },
        tooltip: {
          callbacks: {
            label: ctx => ` ${(ctx.raw * 100).toFixed(1)}%`,
          },
        },
      },
    },
  });
  return ctx._chartInstance;
}

/* ─── Chart factory: drawdown line ─────────────────────────────────────────── */
function buildDrawdownChart(canvasId, dates, drawdownSeries) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  if (ctx._chartInstance) ctx._chartInstance.destroy();

  const datasets = Object.entries(drawdownSeries).map(([asset, values]) => ({
    label:           ASSET_LABELS[asset] || asset.toUpperCase(),
    data:            values,
    borderColor:     PALETTE.assets[asset] || "#8892A4",
    backgroundColor: "transparent",
    borderWidth:     1.6,
    pointRadius:     0,
    tension:         0.1,
  }));

  ctx._chartInstance = new Chart(ctx, {
    type: "line",
    data: { labels: dates, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { color: PALETTE.bg600 },
          ticks: {
            maxTicksLimit: 8,
            color: PALETTE.text400,
          },
        },
        y: {
          grid: { color: PALETTE.bg600 },
          ticks: {
            color: PALETTE.text400,
            callback: v => `${(v * 100).toFixed(0)}%`,
          },
        },
      },
      plugins: {
        legend: { display: true },
        tooltip: {
          callbacks: {
            label: ctx => ` ${(ctx.raw * 100).toFixed(2)}%`,
          },
        },
      },
      interaction: { mode: "index", intersect: false },
    },
  });
  return ctx._chartInstance;
}

/* ─── Chart factory: waterfall bar ─────────────────────────────────────────── */
function buildWaterfallChart(canvasId, scenarios, values) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  if (ctx._chartInstance) ctx._chartInstance.destroy();

  const colors = values.map(v => v >= 0 ? PALETTE.scenarios.positive : PALETTE.scenarios.negative);

  ctx._chartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: scenarios,
      datasets: [{
        label: "Stressed Return",
        data:  values,
        backgroundColor: colors,
        borderColor:     colors,
        borderWidth:     0,
        borderRadius:    3,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: "y",
      scales: {
        x: {
          grid: { color: PALETTE.bg600 },
          ticks: {
            color: PALETTE.text400,
            callback: v => `${(v * 100).toFixed(0)}%`,
          },
        },
        y: {
          grid: { display: false },
          ticks: { color: PALETTE.text300, font: { size: 10 } },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` ${(ctx.raw * 100).toFixed(2)}%`,
          },
        },
      },
    },
  });
  return ctx._chartInstance;
}

/* ─── Chart factory: asset contribution bar ────────────────────────────────── */
function buildContributionChart(canvasId, assets, contributions) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  if (ctx._chartInstance) ctx._chartInstance.destroy();

  const labels = assets.map(a => ASSET_LABELS[a] || a.toUpperCase());
  const colors  = assets.map(a => PALETTE.assets[a] || "#8892A4");

  ctx._chartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Contribution",
        data:  contributions,
        backgroundColor: colors,
        borderColor:     colors,
        borderRadius:    4,
        borderWidth:     0,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { color: PALETTE.bg600 },
          ticks: { color: PALETTE.text400, callback: v => `${(v * 100).toFixed(2)}%` },
        },
        y: {
          grid: { display: false },
          ticks: { color: PALETTE.text300 },
        },
      },
      indexAxis: "y",
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => ` ${(ctx.raw * 100).toFixed(3)}%` } },
      },
    },
  });
  return ctx._chartInstance;
}

/* ─── Chart factory: factor exposure horizontal bar ────────────────────────── */
function buildFactorChart(canvasId, factors, exposures) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  if (ctx._chartInstance) ctx._chartInstance.destroy();

  const sorted = factors
    .map((f, i) => ({ factor: f, value: exposures[i] }))
    .sort((a, b) => b.value - a.value);

  const colors = sorted.map((d, i) => {
    const t = i / Math.max(sorted.length - 1, 1);
    return lerpColor(PALETTE.gold, PALETTE.scarlet, t);
  });

  ctx._chartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: sorted.map(d => d.factor),
      datasets: [{
        data:            sorted.map(d => d.value),
        backgroundColor: colors,
        borderWidth:     0,
        borderRadius:    3,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: "y",
      scales: {
        x: { grid: { color: PALETTE.bg600 }, ticks: { color: PALETTE.text400 } },
        y: { grid: { display: false }, ticks: { color: PALETTE.text300, font: { size: 10 } } },
      },
      plugins: { legend: { display: false } },
    },
  });
  return ctx._chartInstance;
}

/* ─── Helpers ───────────────────────────────────────────────────────────────── */

const ASSET_LABELS = {
  spx:  "S&P 500",
  ndx:  "Nasdaq 100",
  gold: "Gold",
  btc:  "Bitcoin",
};

function lerpColor(hex1, hex2, t) {
  const r1 = parseInt(hex1.slice(1, 3), 16);
  const g1 = parseInt(hex1.slice(3, 5), 16);
  const b1 = parseInt(hex1.slice(5, 7), 16);
  const r2 = parseInt(hex2.slice(1, 3), 16);
  const g2 = parseInt(hex2.slice(3, 5), 16);
  const b2 = parseInt(hex2.slice(5, 7), 16);
  const r = Math.round(r1 + (r2 - r1) * t);
  const g = Math.round(g1 + (g2 - g1) * t);
  const b = Math.round(b1 + (b2 - b1) * t);
  return `#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`;
}

function fmtPct(v, decimals = 1, sign = true) {
  if (v === null || v === undefined || isNaN(v)) return "—";
  const s = (v * 100).toFixed(decimals);
  return sign && v > 0 ? `+${s}%` : `${s}%`;
}

function fmtNum(v, decimals = 2) {
  if (v === null || v === undefined || isNaN(v)) return "—";
  return Number(v).toFixed(decimals);
}

function setClass(el, positive, negative, neutralClass = "") {
  if (!el) return;
  el.classList.remove("text-positive", "text-negative", "text-gold");
  if (positive)       el.classList.add("text-positive");
  else if (negative)  el.classList.add("text-negative");
  else if (neutralClass) el.classList.add(neutralClass);
}

/* ─── API wrappers ──────────────────────────────────────────────────────────── */

async function apiFetch(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

async function apiGet(path)         { return apiFetch(path); }
async function apiPost(path, body)  { return apiFetch(path, { method: "POST", body: JSON.stringify(body) }); }

/* ─── Toast notifications ───────────────────────────────────────────────────── */

function showToast(message, type = "info") {
  let container = document.getElementById("toast-container");
  if (!container) {
    container = document.createElement("div");
    container.id = "toast-container";
    container.style.cssText = `
      position: fixed; bottom: 24px; right: 24px;
      display: flex; flex-direction: column; gap: 8px; z-index: 9999;
    `;
    document.body.appendChild(container);
  }

  const toast = document.createElement("div");
  toast.style.cssText = `
    background: ${PALETTE.bg600}; border: 1px solid ${PALETTE.bg500};
    color: ${PALETTE.text100}; padding: 10px 16px; border-radius: 6px;
    font-size: 13px; max-width: 320px; opacity: 0;
    transition: opacity 0.2s ease; box-shadow: 0 4px 16px rgba(0,0,0,0.5);
  `;
  if (type === "error") toast.style.borderColor = PALETTE.scarlet;
  if (type === "success") toast.style.borderColor = PALETTE.gold;

  toast.textContent = message;
  container.appendChild(toast);
  requestAnimationFrame(() => { toast.style.opacity = "1"; });

  setTimeout(() => {
    toast.style.opacity = "0";
    setTimeout(() => toast.remove(), 200);
  }, 3500);
}

/* ─── Loading state helpers ─────────────────────────────────────────────────── */

function showLoading(buttonEl, text = "Loading…") {
  if (!buttonEl) return;
  buttonEl.disabled = true;
  buttonEl._origText = buttonEl.textContent;
  buttonEl.innerHTML = `<span class="spinner" style="width:14px;height:14px;margin-right:8px;"></span>${text}`;
}

function hideLoading(buttonEl) {
  if (!buttonEl) return;
  buttonEl.disabled = false;
  buttonEl.textContent = buttonEl._origText || "Run";
}

/* ─── Regime colour ─────────────────────────────────────────────────────────── */

const REGIME_COLORS = {
  bull_trend:    "#2EC4A0",
  low_vol_bull:  "#48BB78",
  credit_stress: PALETTE.scarlet,
  high_vol:      "#ED8936",
  recovery:      PALETTE.gold,
  bear_market:   "#E8455A",
};

function regimeColor(regime) {
  return REGIME_COLORS[regime] || PALETTE.text300;
}
