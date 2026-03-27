"""
Phase 9 — FastAPI REST API

Endpoints:
  GET  /                               HTML dashboard (redirect to docs)
  GET  /api/health                     health check
  GET  /api/assets                     available assets + metadata
  POST /api/portfolio/analyze          custom weights → portfolio stats on the fly
  POST /api/scenario/run              custom weights + scenario → stressed results
  GET  /api/scenario/list              all built-in scenarios
  GET  /api/explain/{scenario_id}      plain-English explanation for a scenario
  GET  /api/regime/current             latest regime classification
  GET  /api/portfolio                  full portfolio snapshot (persisted)
  GET  /api/portfolio/weights          base vs regime-adjusted weights
  GET  /api/portfolio/metrics          risk/return metrics
  GET  /api/portfolio/covariance       covariance matrix
  GET  /api/portfolio/expected-returns time-series of ensemble predictions
  GET  /api/stress-tests               all stress scenarios
  GET  /api/regime                     current regime + history
  GET  /api/explainability/{asset}     SHAP + EN coefficients per asset
  GET  /api/explainability             portfolio-level factor exposures
  GET  /api/explanation                full plain-English + quant narrative payload
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Path setup ────────────────────────────────────────────────────────────────
BACKEND_ROOT  = Path(__file__).resolve().parents[1]
FRONTEND_ROOT = BACKEND_ROOT.parent / "frontend"
sys.path.insert(0, str(BACKEND_ROOT))

from services.narrative_engine import NarrativeEngine
from api.pages import router as pages_router

logger = logging.getLogger(__name__)


# ── Scheduler startup ─────────────────────────────────────────────────────────

def _run_scheduler_thread():
    """Runs the live data scheduler in a background thread."""
    import time
    import schedule as sched
    from services.live_data_fetcher import run_hourly, run_daily, run_monthly
    from datetime import datetime

    def _job(name, fn):
        logger.info("SCHEDULER | %s job starting", name)
        try:
            fn()
        except Exception as e:
            logger.error("SCHEDULER | %s job error: %s", name, e)
        logger.info("SCHEDULER | %s job done", name)

    def job_monthly_conditional():
        if datetime.utcnow().day == 1:
            _job("MONTHLY", run_monthly)

    # Run all jobs immediately on startup
    _job("HOURLY",  run_hourly)
    _job("DAILY",   run_daily)
    _job("MONTHLY", run_monthly)

    # Schedule recurring runs
    sched.every().hour.at(":00").do(lambda: _job("HOURLY", run_hourly))
    sched.every().day.at("22:00").do(lambda: _job("DAILY", run_daily))
    sched.every().day.at("08:00").do(job_monthly_conditional)

    logger.info("SCHEDULER | Live data scheduler running (hourly / daily / monthly)")
    while True:
        sched.run_pending()
        time.sleep(30)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start scheduler in a daemon thread so it dies cleanly when the API stops
    t = threading.Thread(target=_run_scheduler_thread, daemon=True, name="LiveDataScheduler")
    t.start()
    logger.info("SCHEDULER | Background thread started (PID thread=%s)", t.ident)
    yield
    # Shutdown: daemon thread stops automatically when main process exits


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Portfolio Stress Testing API",
    description="Macroeconomic & Market Risk Modelling — Phase 9/10 API & Dashboard",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files (served from frontend/) ──────────────────────────────────────
_static_dir = FRONTEND_ROOT / "static"
_static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# ── HTML page routes (Phase 10) ───────────────────────────────────────────────
app.include_router(pages_router)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_python(obj: Any) -> Any:
    """Recursively convert numpy types to native Python for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return [_to_python(v) for v in obj.tolist()]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


# ── Request models ────────────────────────────────────────────────────────────

class PortfolioAnalyzeRequest(BaseModel):
    weights: Dict[str, float]

class PortfolioSaveRequest(BaseModel):
    weights: Dict[str, float]
    metrics: Dict[str, Any]

class ScenarioRunRequest(BaseModel):
    weights: Dict[str, float]
    scenario: str

class ReverseStressRequest(BaseModel):
    weights: Dict[str, float]
    target_loss: float               # e.g. -0.20 for a 20% loss
    top_n: int = 6                   # number of macro drivers to return

class DynamicNarrativeRequest(BaseModel):
    weights: Dict[str, float]
    scenario_result: Optional[Dict[str, Any]] = None   # output of /api/scenario/run


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_csv(relative: str) -> pd.DataFrame:
    path = BACKEND_ROOT / "data" / relative
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"Data not found: {relative}. Run the pipeline first.")
    return pd.read_csv(path)


def _load_json(relative: str) -> dict:
    path = BACKEND_ROOT / "data" / relative
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"Data not found: {relative}. Run the pipeline first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _narrative_engine() -> NarrativeEngine:
    try:
        return NarrativeEngine(backend_root=BACKEND_ROOT)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Narrative engine failed to load: {e}")


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/api/health")
def health():
    """Simple health check."""
    return {"status": "ok", "backend_root": str(BACKEND_ROOT)}


@app.get("/api/assets")
def list_assets():
    """Available assets with metadata (label, available models, current expected return)."""
    er_df   = _load_csv("portfolio/expected_returns.csv")
    metrics = _load_json("portfolio/portfolio_metrics.json")

    latest_er = er_df.iloc[-1]
    annual_er = metrics.get("asset_expected_returns_annual", {})
    annual_vol = metrics.get("asset_volatilities_annual", {})

    ASSET_META = {
        "spx":  {"label": "S&P 500",       "models": ["elastic_net", "xgboost"], "in_portfolio": True},
        "ndx":  {"label": "Nasdaq 100",     "models": ["elastic_net", "xgboost"], "in_portfolio": True},
        "gold": {"label": "Gold",           "models": ["elastic_net", "xgboost"], "in_portfolio": True},
        "btc":  {"label": "Bitcoin",        "models": ["elastic_net", "xgboost"], "in_portfolio": False},
    }

    result = []
    for asset, meta in ASSET_META.items():
        col = f"{asset}_expected_return"
        result.append({
            "asset":                  asset,
            "label":                  meta["label"],
            "models":                 meta["models"],
            "in_portfolio":           meta["in_portfolio"],
            "expected_return_monthly": round(float(latest_er[col]), 6) if col in latest_er.index else None,
            "expected_return_annual":  round(float(annual_er[asset]), 4) if asset in annual_er else None,
            "volatility_annual":       round(float(annual_vol[asset]), 4) if asset in annual_vol else None,
        })
    return _to_python(result)


@app.post("/api/portfolio/analyze")
def portfolio_analyze(req: PortfolioAnalyzeRequest):
    """
    Compute portfolio statistics for any custom set of weights.

    Input:  { "weights": { "spx": 0.3, "ndx": 0.4, "gold": 0.3 } }
    Output: expected return, volatility, Sharpe, VaR, diversification ratio, asset contributions.
    """
    w = req.weights
    total = sum(w.values())
    if abs(total - 1.0) > 0.01:
        raise HTTPException(status_code=422, detail=f"Weights must sum to 1.0 (got {total:.4f})")

    assets = list(w.keys())
    valid  = {"spx", "ndx", "gold", "btc"}
    bad    = [a for a in assets if a not in valid]
    if bad:
        raise HTTPException(status_code=422, detail=f"Unknown assets: {bad}. Valid: {sorted(valid)}")

    # Expected returns (most recent monthly ensemble prediction)
    er_df = _load_csv("portfolio/expected_returns.csv")
    latest = er_df.iloc[-1]
    er_monthly = {}
    for asset in assets:
        col = f"{asset}_expected_return"
        if col not in latest.index:
            raise HTTPException(status_code=503, detail=f"No expected return for {asset}. Run Phase 6 for this asset.")
        er_monthly[asset] = float(latest[col])

    # Covariance matrix (only portfolio assets have it)
    cov_df = _load_csv("portfolio/covariance_matrix.csv").set_index(
        pd.read_csv(BACKEND_ROOT / "data" / "portfolio" / "covariance_matrix.csv").columns[0]
    )
    port_assets = [a for a in assets if a in cov_df.index]
    w_vec  = np.array([w[a] for a in port_assets])
    cov    = cov_df.loc[port_assets, port_assets].values

    port_var_monthly   = float(w_vec @ cov @ w_vec)
    port_vol_monthly   = float(port_var_monthly ** 0.5)
    port_er_monthly    = sum(w[a] * er_monthly[a] for a in assets)
    asset_vols_monthly = {a: float(cov_df.loc[a, a] ** 0.5) if a in cov_df.index else None for a in port_assets}

    # Risk-free rate
    rf_monthly = float(_load_json("portfolio/portfolio_metrics.json").get("risk_free_rate_monthly", 0.003))

    sharpe_annual      = (port_er_monthly - rf_monthly) / port_vol_monthly * (12 ** 0.5) if port_vol_monthly > 0 else 0
    var_95_monthly     = -1.645 * port_vol_monthly
    cvar_95_monthly    = -port_vol_monthly * 0.3989 / 0.05   # normal: φ(z)/α

    # Diversification ratio = weighted sum of individual vols / portfolio vol
    div_ratio = sum(w[a] * asset_vols_monthly[a] for a in port_assets) / port_vol_monthly if port_vol_monthly > 0 else 1.0

    # Per-asset return contributions
    contributions = {a: round(w[a] * er_monthly[a] * 12, 6) for a in assets}

    return _to_python({
        "weights":                   w,
        "expected_return_monthly":   round(port_er_monthly, 6),
        "expected_return_annual":    round(port_er_monthly * 12, 6),
        "volatility_monthly":        round(port_vol_monthly, 6),
        "volatility_annual":         round(port_vol_monthly * (12 ** 0.5), 6),
        "sharpe_ratio_annual":       round(sharpe_annual, 4),
        "var_95_monthly":            round(var_95_monthly, 6),
        "cvar_95_monthly":           round(cvar_95_monthly, 6),
        "diversification_ratio":     round(div_ratio, 4),
        "asset_er_monthly":          {a: round(er_monthly[a], 6) for a in assets},
        "asset_contributions_annual": contributions,
        "as_of_date":                str(latest["date"]),
    })


@app.post("/api/portfolio/save")
def portfolio_save(req: PortfolioSaveRequest):
    """Persist a user-built portfolio so the Dashboard can display it."""
    import datetime
    payload = {
        "weights": req.weights,
        "metrics": req.metrics,
        "saved_at": datetime.datetime.utcnow().isoformat(),
    }
    save_path = BACKEND_ROOT / "data" / "portfolio" / "user_portfolio.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(_to_python(payload), f, indent=2)
    return {"status": "saved"}


@app.delete("/api/portfolio/save")
def portfolio_clear():
    """Remove the user-saved portfolio from the Dashboard."""
    save_path = BACKEND_ROOT / "data" / "portfolio" / "user_portfolio.json"
    if save_path.exists():
        save_path.unlink()
    return {"status": "cleared"}


@app.post("/api/scenario/run")
def scenario_run(req: ScenarioRunRequest):
    """
    Apply any built-in scenario to custom portfolio weights.

    Input:  { "weights": {"spx": 0.3, "ndx": 0.4, "gold": 0.3}, "scenario": "gfc_peak" }
    Output: portfolio return for that scenario, asset contributions, scenario metadata.
    """
    stress_df = _load_csv("portfolio/stress_test_results.csv")
    row = stress_df[stress_df["scenario"] == req.scenario]
    if row.empty:
        available = stress_df["scenario"].tolist()
        raise HTTPException(status_code=404, detail=f"Scenario '{req.scenario}' not found. Available: {available}")

    row = row.iloc[0]
    w   = req.weights

    asset_returns = {}
    contributions = {}
    for asset in ["spx", "ndx", "gold"]:
        col = f"{asset}_total_return"
        if col in row.index and pd.notna(row[col]):
            asset_returns[asset] = float(row[col])
            contributions[asset] = round(w.get(asset, 0) * float(row[col]), 6)
        else:
            asset_returns[asset] = None
            contributions[asset] = 0.0

    portfolio_return = sum(contributions.values())

    return _to_python({
        "scenario":          req.scenario,
        "stress_type":       row["stress_type"],
        "weights_used":      w,
        "portfolio_return":  round(portfolio_return, 6),
        "asset_returns":     asset_returns,
        "contributions":     contributions,
        "start_date":        row.get("start_date"),
        "end_date":          row.get("end_date"),
        "n_months":          row.get("n_months"),
        "description":       row.get("scenario_description"),
    })


@app.get("/api/scenario/list")
def scenario_list():
    """All built-in scenarios with type, dates, and pre-computed portfolio impact."""
    stress_df = _load_csv("portfolio/stress_test_results.csv")
    stress_df = stress_df.sort_values(["stress_type", "portfolio_total_return"])
    return _to_python(stress_df.to_dict(orient="records"))


@app.get("/api/regime/current")
def regime_current():
    """Latest regime classification (simplified — just current, no history)."""
    df = _load_csv("regimes/regime_dataset.csv")
    current = df.iloc[-1]
    return _to_python({
        "date":       str(pd.to_datetime(current["date"]).date()),
        "regime":     current["regime_label"],
        "confidence": round(float(current["regime_confidence"]), 4),
        "description": {
            "calm":             "Low-volatility growth environment. Risk assets typically perform well.",
            "inflation_stress": "Elevated inflation with central bank tightening. Growth equities under pressure.",
            "credit_stress":    "Widening credit spreads and risk-off sentiment. Quality and defensive assets preferred.",
            "crisis":           "Acute market stress. Correlations spike and diversification breaks down.",
        }.get(current["regime_label"], ""),
    })


@app.get("/api/explain/{scenario_id}")
def explain_scenario(scenario_id: str):
    """
    Plain-English explanation block + SHAP summary for a specific scenario.

    Returns:
      - scenario metadata
      - portfolio impact with the current regime-adjusted weights
      - plain-English narrative explaining the scenario drivers
      - top SHAP features most relevant to this scenario
    """
    stress_df = _load_csv("portfolio/stress_test_results.csv")
    row = stress_df[stress_df["scenario"] == scenario_id]
    if row.empty:
        available = stress_df["scenario"].tolist()
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found. Available: {available}")

    row       = row.iloc[0]
    metrics   = _load_json("portfolio/portfolio_metrics.json")
    weights_w = _load_csv("portfolio/portfolio_weights_regime_adjusted.csv")
    adj_w     = dict(zip(weights_w["asset"], weights_w["weight"]))

    portfolio_return = float(row["portfolio_total_return"])
    stress_type      = row["stress_type"]
    scenario_name    = scenario_id.replace("_", " ").title()

    # Build per-asset contribution
    contributions = {}
    for asset in ["spx", "ndx", "gold"]:
        col = f"{asset}_total_return"
        if col in row.index and pd.notna(row[col]):
            contributions[asset] = {
                "weight":      round(float(adj_w.get(asset, 0)), 4),
                "asset_return": round(float(row[col]), 4),
                "contribution": round(float(adj_w.get(asset, 0)) * float(row[col]), 4),
            }

    # Plain-English narrative for this scenario
    narrative = _scenario_narrative(scenario_id, portfolio_return, contributions, adj_w, metrics)

    # SHAP features most associated with this scenario's macro drivers
    shap_context = _scenario_shap_context(scenario_id)

    return _to_python({
        "scenario":          scenario_id,
        "scenario_name":     scenario_name,
        "stress_type":       stress_type,
        "start_date":        row.get("start_date"),
        "end_date":          row.get("end_date"),
        "portfolio_return":  round(portfolio_return, 4),
        "asset_contributions": contributions,
        "explanation": {
            "text":          narrative,
            "shap_context":  shap_context,
        },
    })


def _scenario_narrative(
    scenario_id: str,
    portfolio_return: float,
    contributions: dict,
    weights: dict,
    metrics: dict,
) -> str:
    """Generate plain-English explanation text for a specific scenario."""
    name     = scenario_id.replace("_", " ").title()
    port_pct = portfolio_return * 100

    # Biggest loss contributor
    losers  = [(a, d) for a, d in contributions.items() if d["contribution"] < 0]
    gainers = [(a, d) for a, d in contributions.items() if d["contribution"] > 0]
    losers.sort(key=lambda x: x[1]["contribution"])
    gainers.sort(key=lambda x: -x[1]["contribution"])

    ASSET_LABELS = {"spx": "S&P 500", "ndx": "Nasdaq 100", "gold": "Gold"}

    text = f"In the '{name}' scenario, the portfolio would return {port_pct:+.1f}% cumulatively. "

    if losers:
        worst_asset, worst_data = losers[0]
        text += (
            f"{ASSET_LABELS.get(worst_asset, worst_asset)} ({worst_data['weight']:.0%} weight) "
            f"is the primary source of loss, contributing {worst_data['contribution'] * 100:+.1f}% "
            f"(asset return: {worst_data['asset_return'] * 100:+.1f}%). "
        )

    if gainers:
        best_asset, best_data = gainers[0]
        text += (
            f"{ASSET_LABELS.get(best_asset, best_asset)} ({best_data['weight']:.0%} weight) "
            f"partially offsets the loss, contributing {best_data['contribution'] * 100:+.1f}% "
            f"(asset return: {best_data['asset_return'] * 100:+.1f}%). "
        )

    # Scenario-specific context
    SCENARIO_CONTEXT = {
        "gfc_peak": (
            "The 2008 Global Financial Crisis peak (Sep–Dec 2008) was driven by a collapse in credit markets, "
            "bank solvency fears, and forced deleveraging. Equities fell sharply while Gold initially sold off "
            "on liquidity demands before recovering as a safe haven."
        ),
        "gfc_recovery": (
            "The GFC recovery (Mar 2009–Dec 2010) saw a powerful equity rebound as central banks flooded markets "
            "with liquidity. Growth assets significantly outperformed defensive positions."
        ),
        "euro_crisis": (
            "The Eurozone sovereign debt crisis (2011–2012) was characterised by contagion from peripheral "
            "European sovereigns. Gold rallied as a safe haven while equities showed high volatility."
        ),
        "covid_crash": (
            "The COVID-19 crash (Feb–Mar 2020) was one of the fastest equity declines in history. "
            "Even Gold initially fell on forced selling, but recovered rapidly once central banks intervened."
        ),
        "covid_recovery": (
            "The COVID recovery (Apr 2020–Dec 2021) was driven by unprecedented fiscal and monetary stimulus, "
            "producing exceptional equity returns — particularly in technology (Nasdaq)."
        ),
        "inflation_spike_22": (
            "The 2022 inflation spike saw the Fed raise rates by 425bps. Both equities and bonds fell sharply "
            "— a rare simultaneous drawdown — as rising rates compressed valuations and hurt duration assets."
        ),
    }

    if scenario_id in SCENARIO_CONTEXT:
        text += SCENARIO_CONTEXT[scenario_id]
    elif "regime_calm" in scenario_id:
        text += "In calm regimes, low volatility and steady growth support risk assets across the board."
    elif "regime_inflation" in scenario_id:
        text += "Inflation stress periods reward commodity exposure while pressuring growth equities."
    elif "regime_credit" in scenario_id:
        text += "Credit stress regimes favour quality and defensive assets; high-yield spreads widen."
    elif "regime_crisis" in scenario_id:
        text += "Crisis regimes see correlation spikes: assets that normally offset each other can fall together."
    elif "rates_up" in scenario_id:
        text += "A 100bps rate shock compresses equity valuations and increases discount rates on future cash flows."
    elif "systemic" in scenario_id:
        text += "Systemic crises trigger forced selling, liquidity hoarding, and breakdown of diversification."

    return text


def _scenario_shap_context(scenario_id: str) -> List[dict]:
    """
    Return SHAP features most relevant to this scenario's macro drivers.
    Maps scenario types to the feature(s) most likely driving the stress.
    """
    SCENARIO_FEATURE_MAP = {
        "gfc_peak":            ["high_yield_spread", "vix_level", "spx_vol_3m"],
        "gfc_recovery":        ["spx_return", "ndx_return", "regime_confidence"],
        "euro_crisis":         ["high_yield_spread", "eurusd_return", "ecb_level"],
        "covid_crash":         ["vix_level", "spx_vol_3m", "high_yield_spread"],
        "covid_recovery":      ["spx_return", "ndx_return", "us_cpi_yoy"],
        "inflation_spike_22":  ["us_cpi_yoy", "us10y_yield", "us2y_yield"],
        "rates_up_100bps":     ["us10y_yield", "us2y_yield", "yield_spread"],
        "hawkish_policy_shock":["us2y_yield", "fed_funds_level", "us_cpi_yoy"],
        "stagflation_regime":  ["us_cpi_yoy", "high_yield_spread", "vix_level"],
        "systemic_crisis":     ["vix_level", "high_yield_spread", "spx_vol_3m"],
        "credit_spread_widening_200bps": ["high_yield_spread", "yield_spread"],
    }

    FACTOR_LABELS = {
        "us_cpi_yoy":        "US CPI inflation",
        "vix_level":         "market volatility (VIX)",
        "us10y_yield":       "10-year Treasury yield",
        "us2y_yield":        "2-year Treasury yield",
        "yield_spread":      "yield curve (2s10s spread)",
        "high_yield_spread": "high-yield credit spreads",
        "spx_vol_3m":        "S&P 500 realized volatility",
        "eurusd_return":     "EUR/USD exchange rate",
        "spx_return":        "S&P 500 momentum",
        "ndx_return":        "Nasdaq momentum",
        "ecb_level":         "ECB policy rate",
        "regime_confidence": "regime certainty",
        "fed_funds_level":   "Fed Funds rate",
    }

    relevant_features = SCENARIO_FEATURE_MAP.get(scenario_id, ["us_cpi_yoy", "vix_level", "high_yield_spread"])

    # Pull actual SHAP values for these features across assets
    result = []
    for feature in relevant_features:
        entry = {"feature": feature, "label": FACTOR_LABELS.get(feature, feature), "per_asset": {}}
        for asset in ["spx", "ndx", "gold"]:
            path = BACKEND_ROOT / "data" / "explainability" / f"shap_global_xgb_{asset}.csv"
            if path.exists():
                df  = pd.read_csv(path)
                row = df[df["feature"] == feature]
                if not row.empty:
                    entry["per_asset"][asset] = round(float(row["mean_abs_shap"].iloc[0]), 6)
        result.append(entry)

    return result


# ── Portfolio ─────────────────────────────────────────────────────────────────

@app.get("/api/portfolio")
def portfolio_snapshot():
    """Full portfolio snapshot: weights, expected returns, and risk metrics."""
    metrics    = _load_json("portfolio/portfolio_metrics.json")
    weights_df = _load_csv("portfolio/portfolio_weights_regime_adjusted.csv")
    base_df    = _load_csv("portfolio/portfolio_weights.csv")

    weights = weights_df[["asset", "weight", "regime", "regime_confidence", "as_of_date"]].to_dict(orient="records")
    base    = base_df[["asset", "weight"]].to_dict(orient="records")

    return _to_python({
        "as_of_date":       metrics.get("as_of_date"),
        "regime":           weights_df["regime"].iloc[0],
        "regime_confidence": float(weights_df["regime_confidence"].iloc[0]),
        "base_weights":     base,
        "adjusted_weights": weights,
        "expected_returns": {
            a: round(r, 6)
            for a, r in metrics.get("asset_expected_returns_annual", {}).items()
        },
        "metrics": {
            "expected_return_annual":  metrics.get("expected_return_annual"),
            "volatility_annual":       metrics.get("volatility_annual"),
            "sharpe_ratio":            metrics.get("sharpe_ratio"),
            "var_95_monthly":          metrics.get("var_95_monthly"),
            "cvar_95_monthly":         metrics.get("cvar_95_monthly"),
            "max_drawdown":            metrics.get("max_drawdown"),
            "diversification_ratio":   metrics.get("diversification_ratio"),
            "risk_free_rate_annual":   metrics.get("risk_free_rate_annual"),
        },
    })


@app.get("/api/portfolio/weights")
def portfolio_weights():
    """Base (MVO) vs regime-adjusted weights side by side."""
    adj_df  = _load_csv("portfolio/portfolio_weights_regime_adjusted.csv")
    base_df = _load_csv("portfolio/portfolio_weights.csv")

    base_map = dict(zip(base_df["asset"], base_df["weight"]))
    result = []
    for _, row in adj_df.iterrows():
        asset = row["asset"]
        result.append({
            "asset":          asset,
            "base_weight":    round(float(base_map.get(asset, 0)), 6),
            "adjusted_weight": round(float(row["weight"]), 6),
            "regime_tilt":    round(float(row["weight"]) - float(base_map.get(asset, 0)), 6),
            "regime":         row["regime"],
            "regime_confidence": float(row["regime_confidence"]),
        })
    return _to_python(result)


@app.get("/api/portfolio/metrics")
def portfolio_metrics():
    """Risk/return metrics for the regime-adjusted portfolio."""
    return _to_python(_load_json("portfolio/portfolio_metrics.json"))


@app.get("/api/portfolio/covariance")
def portfolio_covariance():
    """Ledoit-Wolf shrinkage covariance matrix (36-month rolling)."""
    df = _load_csv("portfolio/covariance_matrix.csv")
    df = df.set_index(df.columns[0])
    return _to_python({
        "assets":  list(df.columns),
        "matrix":  df.values.tolist(),
        "correlations": {
            f"{r}_{c}": round(float(df.loc[r, c]) / (df.loc[r, r] ** 0.5 * df.loc[c, c] ** 0.5), 4)
            for r in df.index
            for c in df.columns
            if r != c
        },
    })


@app.get("/api/portfolio/expected-returns")
def portfolio_expected_returns():
    """Time-series of ensemble expected return predictions (test set)."""
    df = _load_csv("portfolio/expected_returns.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return _to_python(df.to_dict(orient="records"))


# ── Stress Tests ──────────────────────────────────────────────────────────────

@app.get("/api/stress-tests")
def stress_tests(type: Optional[str] = None):
    """
    All stress scenarios.  Optionally filter by type:
      historical | regime_shock | macro_scenario
    """
    df = _load_csv("portfolio/stress_test_results.csv")
    if type:
        df = df[df["stress_type"] == type]
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No scenarios of type '{type}'")
    df = df.sort_values(["stress_type", "portfolio_total_return"])
    records = df.to_dict(orient="records")
    return _to_python(records)


@app.get("/api/stress-tests/{scenario_name}")
def stress_test_by_scenario(scenario_name: str):
    """Single scenario by name."""
    df = _load_csv("portfolio/stress_test_results.csv")
    row = df[df["scenario"] == scenario_name]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_name}' not found.")
    return _to_python(row.iloc[0].to_dict())


# ── Regime ────────────────────────────────────────────────────────────────────

@app.get("/api/regime")
def regime(history_months: int = 24):
    """Current regime + recent history."""
    df = _load_csv("regimes/regime_dataset.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    current = df.iloc[-1]
    recent  = df.tail(history_months)[["date", "regime_label", "regime_confidence"]].to_dict(orient="records")

    # Regime distribution over history window
    dist = df.tail(history_months)["regime_label"].value_counts(normalize=True).round(4).to_dict()

    return _to_python({
        "current": {
            "date":       current["date"],
            "regime":     current["regime_label"],
            "confidence": round(float(current["regime_confidence"]), 4),
        },
        "history":      recent,
        "distribution": dist,
    })


# ── Explainability ────────────────────────────────────────────────────────────

@app.get("/api/explainability")
def portfolio_explainability():
    """Portfolio-level factor exposure decomposition (weighted SHAP across assets)."""
    engine = _narrative_engine()
    return _to_python({
        "factor_exposures":      engine._portfolio_factor_exposures(),
        "per_asset_top_factors": engine._per_asset_top_factors(),
    })


@app.get("/api/explainability/{asset}")
def asset_explainability(asset: str):
    """
    SHAP global importance + ElasticNet coefficients for a single asset.
    Assets: spx | ndx | gold | btc
    """
    asset = asset.lower()
    valid = ["spx", "ndx", "gold", "btc"]
    if asset not in valid:
        raise HTTPException(status_code=422, detail=f"Asset must be one of {valid}")

    xgb_path = BACKEND_ROOT / "data" / "explainability" / f"shap_global_xgb_{asset}.csv"
    en_path  = BACKEND_ROOT / "data" / "explainability" / f"elastic_net_coefficients_{asset}.csv"
    cmp_path = BACKEND_ROOT / "data" / "explainability" / "feature_importance_comparison.csv"

    result: dict = {"asset": asset}

    if xgb_path.exists():
        df = pd.read_csv(xgb_path)
        result["xgb_shap_global"] = df.to_dict(orient="records")

    if en_path.exists():
        df = pd.read_csv(en_path)
        active = df[df["feature"] != "__intercept__"]
        intercept_row = df[df["feature"] == "__intercept__"]
        result["en_intercept"] = float(intercept_row["coefficient"].iloc[0]) if not intercept_row.empty else None
        result["en_active_coefficients"] = active[active["coefficient"] != 0].to_dict(orient="records")
        result["en_n_active"] = int((active["coefficient"] != 0).sum())

    if cmp_path.exists():
        df = pd.read_csv(cmp_path)
        asset_df = df[df["asset"] == asset].sort_values("xgb_mean_abs_shap", ascending=False)
        result["feature_comparison"] = asset_df.to_dict(orient="records")

    # Local explanation (most recent prediction)
    try:
        report = _load_json("explainability/explainability_report.json")
        if asset in report.get("assets", {}):
            result["latest_explanation"] = report["assets"][asset].get("latest_explanation")
    except Exception:
        pass

    return _to_python(result)


# ── Plain-English Explanation ─────────────────────────────────────────────────

@app.get("/api/explanation")
def explanation():
    """Full pre-computed explanation payload from the pipeline outputs."""
    engine = _narrative_engine()
    return _to_python(engine.generate())


# ── Dynamic Narrative ──────────────────────────────────────────────────────────

@app.post("/api/narrative/dynamic")
def dynamic_narrative(req: DynamicNarrativeRequest):
    """
    Generate a live narrative for any user-supplied weights.

    Input:
      {
        "weights": {"spx": 0.5, "ndx": 0.3, "gold": 0.2},
        "scenario_result": { ...output of /api/scenario/run... }   // optional
      }
    Output: same structure as /api/explanation but computed for the supplied weights.
    """
    w = req.weights
    total = sum(w.values())
    if total <= 0:
        raise HTTPException(status_code=422, detail="Weights must sum to a positive value.")
    # Normalise
    w = {k: v / total for k, v in w.items()}

    engine = _narrative_engine()
    return _to_python(engine.generate_for_weights(w, req.scenario_result))


# ── Reverse Stress Test ────────────────────────────────────────────────────────

_MACRO_FEATURES = [
    "us_cpi_yoy", "yield_spread", "vix_level", "high_yield_spread",
    "us10y_yield", "us2y_yield", "spx_vol_3m", "eurusd_return", "gbpusd_return",
]

_MACRO_LABELS = {
    "us_cpi_yoy":        "US CPI inflation (YoY %)",
    "yield_spread":      "Yield curve (10Y–2Y spread)",
    "vix_level":         "VIX (market volatility)",
    "high_yield_spread": "High-yield credit spread",
    "us10y_yield":       "US 10-year Treasury yield",
    "us2y_yield":        "US 2-year Treasury yield",
    "spx_vol_3m":        "S&P 500 realised volatility (3M)",
    "eurusd_return":     "EUR/USD return",
    "gbpusd_return":     "GBP/USD return",
}

_MACRO_DIRECTION = {
    "us_cpi_yoy":        ("rising inflation", "falling inflation"),
    "yield_spread":      ("curve steepening", "curve inversion / flattening"),
    "vix_level":         ("volatility spike", "volatility collapse"),
    "high_yield_spread": ("credit spread widening (stress)", "credit spread tightening"),
    "us10y_yield":       ("rising long rates", "falling long rates"),
    "us2y_yield":        ("rising short rates (Fed tightening)", "falling short rates (Fed easing)"),
    "spx_vol_3m":        ("realised vol surge", "realised vol compression"),
    "eurusd_return":     ("EUR strengthening / USD weakening", "EUR weakening / USD strengthening"),
    "gbpusd_return":     ("GBP strengthening / USD weakening", "GBP weakening / USD strengthening"),
}


@app.post("/api/stress/reverse")
def reverse_stress_test(req: ReverseStressRequest):
    """
    Reverse stress test: given a target portfolio loss, find the minimum-norm
    combination of macro shocks that would cause it.

    Input:
      { "weights": {"spx": 0.4, "ndx": 0.4, "gold": 0.2}, "target_loss": -0.20 }

    Output:
      - required_shocks: list of macro features with required delta, z-score, and plain-English label
      - minimum_norm_loss: what the minimum-norm shock produces (should ≈ target_loss)
      - closest_historical: the existing scenario whose portfolio return is nearest the target
      - interpretation: plain-English summary
    """
    w = req.weights
    total = sum(w.values())
    if total <= 0 or abs(total - 1.0) > 0.02:
        raise HTTPException(status_code=422, detail=f"Weights must sum to 1.0 (got {total:.3f}).")
    if req.target_loss >= 0:
        raise HTTPException(status_code=422, detail="target_loss must be negative (e.g. -0.20 for a 20% loss).")

    # ── 1. Load historical monthly features ──────────────────────────────────
    feat_path = BACKEND_ROOT / "data" / "features" / "features_monthly_full_history.csv"
    if not feat_path.exists():
        raise HTTPException(status_code=503, detail="features_monthly_full_history.csv not found. Run Phase 3.")
    feat_df = pd.read_csv(feat_path)

    # ── 2. Compute historical portfolio returns for the given weights ──────────
    ASSET_RET_COLS = {"spx": "spx_return", "ndx": "ndx_return", "gold": "gold_return", "btc": "btc_return"}
    available_assets = [a for a in w if ASSET_RET_COLS.get(a) in feat_df.columns]
    if not available_assets:
        raise HTTPException(status_code=422, detail="No matching asset return columns found in features data.")

    port_returns = pd.Series(0.0, index=feat_df.index)
    for asset in available_assets:
        col = ASSET_RET_COLS[asset]
        port_returns += w[asset] * feat_df[col].fillna(0)

    # ── 3. Build OLS sensitivity: port_return ~ macro_features ────────────────
    avail_features = [f for f in _MACRO_FEATURES if f in feat_df.columns]
    X_raw = feat_df[avail_features].copy()
    y_raw = port_returns.copy()

    # Align and drop NaN rows
    combined = pd.concat([X_raw, y_raw.rename("__target__")], axis=1).dropna()
    X = combined[avail_features].values
    y = combined["__target__"].values

    if len(y) < 12:
        raise HTTPException(status_code=503, detail="Insufficient clean data rows for sensitivity estimation.")

    # OLS with intercept
    X_const = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(X_const, y, rcond=None)
    intercept  = coef[0]
    sensitivity = coef[1:]   # one coefficient per macro feature

    # ── 4. Minimum-norm solution: find Δf minimising ||Δf||² s.t. β·Δf = target ──
    target_adj  = req.target_loss - intercept
    norm_sq     = float(sensitivity @ sensitivity)
    if norm_sq < 1e-12:
        raise HTTPException(status_code=500, detail="Sensitivity vector is near-zero; model cannot back-solve.")

    delta_X = sensitivity * (target_adj / norm_sq)   # pseudo-inverse solution

    # ── 5. Express shocks as z-scores relative to historical feature std ───────
    feat_means = combined[avail_features].mean()
    feat_stds  = combined[avail_features].std().replace(0, 1e-9)

    # ── 6. Build result table, sort by |z-score| ──────────────────────────────
    shocks = []
    for i, feat in enumerate(avail_features):
        delta  = float(delta_X[i])
        z      = delta / float(feat_stds[feat])
        sens   = float(sensitivity[i])
        mean_v = float(feat_means[feat])
        std_v  = float(feat_stds[feat])
        pos_label, neg_label = _MACRO_DIRECTION.get(feat, ("increase", "decrease"))
        shocks.append({
            "feature":       feat,
            "label":         _MACRO_LABELS.get(feat, feat),
            "required_delta": round(delta, 6),
            "z_score":       round(z, 3),
            "abs_z_score":   round(abs(z), 3),
            "sensitivity":   round(sens, 6),
            "current_mean":  round(mean_v, 4),
            "historical_std": round(std_v, 4),
            "direction":     pos_label if delta > 0 else neg_label,
            "direction_sign": "+" if delta > 0 else "−",
        })

    shocks.sort(key=lambda s: -s["abs_z_score"])
    top_shocks = shocks[:req.top_n]

    # ── 7. Find closest historical scenario ───────────────────────────────────
    stress_path = BACKEND_ROOT / "data" / "portfolio" / "stress_test_results.csv"
    closest_scenario = None
    if stress_path.exists():
        stress_df = pd.read_csv(stress_path)
        # Recompute portfolio return for custom weights for each scenario
        stress_df["custom_port_return"] = sum(
            w.get(a, 0) * stress_df[f"{a}_total_return"].fillna(0)
            for a in ["spx", "ndx", "gold"]
        )
        stress_df["distance"] = (stress_df["custom_port_return"] - req.target_loss).abs()
        best_row = stress_df.nsmallest(1, "distance").iloc[0]
        closest_scenario = {
            "scenario":        str(best_row["scenario"]),
            "stress_type":     str(best_row.get("stress_type", "—")),
            "portfolio_return": round(float(best_row["custom_port_return"]), 4),
            "description":     str(best_row.get("scenario_description", "")) if pd.notna(best_row.get("scenario_description")) else None,
        }

    # ── 8. Plain-English interpretation ───────────────────────────────────────
    target_pct = f"{abs(req.target_loss):.0%}"
    top1 = top_shocks[0] if top_shocks else None
    top2 = top_shocks[1] if len(top_shocks) > 1 else None

    interp_parts = [
        f"To produce a portfolio loss of {target_pct}, the minimum macro shock "
        f"requires primarily: "
    ]
    for s in top_shocks[:3]:
        interp_parts.append(
            f"{s['label']} shifting by {s['required_delta']:+.3f} "
            f"({s['abs_z_score']:.1f} standard deviations — {s['direction']})"
        )

    severity = (
        "extreme (beyond any historical precedent)" if any(s["abs_z_score"] > 4 for s in top_shocks) else
        "severe (comparable to GFC / COVID-level stress)" if any(s["abs_z_score"] > 2.5 for s in top_shocks) else
        "elevated (comparable to a significant market correction)" if any(s["abs_z_score"] > 1.5 for s in top_shocks) else
        "moderate (within the range of recent stress events)"
    )

    interpretation = (
        f"A {target_pct} portfolio loss requires {severity} macro conditions. "
        + ". ".join(interp_parts[1:]) + ". "
    )
    if closest_scenario:
        interpretation += (
            f"The closest historical analogue is '{closest_scenario['scenario'].replace('_', ' ').title()}' "
            f"which produced a {closest_scenario['portfolio_return']:+.1%} return under your current weights."
        )

    return _to_python({
        "target_loss":        req.target_loss,
        "weights_used":       w,
        "required_shocks":    top_shocks,
        "severity":           severity,
        "closest_historical": closest_scenario,
        "interpretation":     interpretation,
        "model_note":         "Minimum-norm OLS inversion: finds the smallest macro shock vector (in Euclidean norm) that achieves the target return.",
    })


# ── Manual Live-Data Refresh ───────────────────────────────────────────────────

_refresh_locks: Dict[str, threading.Lock] = {
    "hourly":  threading.Lock(),
    "daily":   threading.Lock(),
    "monthly": threading.Lock(),
}

@app.post("/api/live-data/refresh/{feed_type}")
async def refresh_live_data(feed_type: str):
    """
    Manually trigger a live-data pull for one feed frequency.
    Returns the updated freshness metadata and (for hourly) latest prices.
    Runs synchronously in a thread-pool executor so it does not block the event loop.
    A per-frequency lock prevents duplicate concurrent refreshes.
    """
    import asyncio
    from services.live_data_fetcher import run_hourly, run_daily, run_monthly

    if feed_type not in ("hourly", "daily", "monthly"):
        raise HTTPException(status_code=422, detail="feed_type must be 'hourly', 'daily', or 'monthly'.")

    fn_map = {"hourly": run_hourly, "daily": run_daily, "monthly": run_monthly}
    lock   = _refresh_locks[feed_type]

    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail=f"{feed_type} refresh already in progress.")

    def _run():
        try:
            return fn_map[feed_type]()
        finally:
            lock.release()

    try:
        ok = await asyncio.get_event_loop().run_in_executor(None, _run)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Re-read freshness file and (for hourly) prices so the response is up-to-date
    DATA_ROOT = BACKEND_ROOT / "data"
    freshness_path = DATA_ROOT / "live" / "data_freshness.json"
    prices_path    = DATA_ROOT / "live" / "latest_prices.json"

    freshness = {}
    if freshness_path.exists():
        with open(freshness_path) as f:
            freshness = json.load(f)

    prices = {}
    fetched_at = None
    if prices_path.exists():
        with open(prices_path) as f:
            raw = json.load(f)
            prices    = raw.get("prices", {})
            fetched_at = raw.get("fetched_at")

    info = freshness.get(feed_type, {})
    return {
        "ok":           ok,
        "feed_type":    feed_type,
        "last_updated": info.get("last_updated"),
        "status":       info.get("status", "error"),
        "detail":       info.get("detail", ""),
        "fetched_at":   fetched_at,
        "prices":       prices if feed_type == "hourly" else {},
    }
