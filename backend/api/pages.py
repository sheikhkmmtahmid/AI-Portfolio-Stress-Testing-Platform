"""
Phase 10 — HTML Template Routes

Serves all five dashboard pages via Jinja2 server-rendered templates.
Each route loads the required data from the pipeline output files and
passes it directly to the template context.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import asyncio
import threading

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates

# ── Paths ─────────────────────────────────────────────────────────────────────
BACKEND_ROOT  = Path(__file__).resolve().parents[1]
FRONTEND_ROOT = BACKEND_ROOT.parent / "frontend"
sys.path.insert(0, str(BACKEND_ROOT))

router    = APIRouter()
templates = Jinja2Templates(directory=str(FRONTEND_ROOT / "templates"))

# ── Live-data refresh locks ────────────────────────────────────────────────────
_refresh_locks = {
    "hourly":  threading.Lock(),
    "daily":   threading.Lock(),
    "monthly": threading.Lock(),
}

NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate",
    "Pragma": "no-cache",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_python(obj: Any) -> Any:
    """Recursively convert numpy types to native Python."""
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


def _load_json(relative: str) -> Optional[dict]:
    path = BACKEND_ROOT / "data" / relative
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_csv(relative: str) -> Optional[pd.DataFrame]:
    path = BACKEND_ROOT / "data" / relative
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _portfolio_context() -> dict:
    """Build the standard portfolio_data dict passed to most pages."""
    metrics  = _load_json("portfolio/portfolio_metrics.json")
    if metrics is None:
        return {}

    # Weights
    base_df  = _load_csv("portfolio/portfolio_weights.csv")
    reg_df   = _load_csv("portfolio/portfolio_weights_regime_adjusted.csv")

    base_w = {}
    reg_w  = {}
    if base_df is not None and "asset" in base_df.columns and "weight" in base_df.columns:
        base_w = dict(zip(base_df["asset"], base_df["weight"].astype(float)))
    if reg_df is not None and "asset" in reg_df.columns and "weight" in reg_df.columns:
        reg_w  = dict(zip(reg_df["asset"],  reg_df["weight"].astype(float)))

    # As-of date from expected returns
    er_df   = _load_csv("portfolio/expected_returns.csv")
    as_of   = str(er_df.iloc[-1]["date"])[:10] if (er_df is not None and "date" in er_df.columns) else "—"

    return _to_python({
        "metrics":    metrics,
        "weights":    {"base": base_w, "regime_adjusted": reg_w},
        "as_of_date": as_of,
    })


def _regime_context() -> dict:
    """Build the regime_data dict: current + history + counts."""
    df = _load_csv("regimes/regime_dataset.csv")
    if df is None:
        return {}

    current = df.iloc[-1]
    history = []
    for _, row in df.tail(36).iterrows():
        history.append({
            "date":    str(row.get("date", ""))[:10],
            "regime":  row.get("regime_label", ""),
        })

    counts: Dict[str, int] = {}
    for r in df["regime_label"].dropna():
        counts[r] = counts.get(r, 0) + 1

    return _to_python({
        "date":          str(pd.to_datetime(current["date"]).date()),
        "regime":        current.get("regime_label", "unknown"),
        "confidence":    float(current.get("regime_confidence", 1.0)),
        "history":       history,
        "regime_counts": counts,
    })


def _stress_context() -> dict:
    """Load stress test results as {scenario_id: {portfolio_return, asset_returns, …}}."""
    df = _load_csv("portfolio/stress_test_results.csv")
    if df is None:
        return {}

    result = {}
    for _, row in df.iterrows():
        sc_id = row.get("scenario", "")
        asset_contribs = {}
        asset_returns  = {}
        for asset in ["spx", "ndx", "gold", "btc"]:
            ret_col = f"{asset}_total_return"
            if ret_col in row.index and pd.notna(row[ret_col]):
                ar = float(row[ret_col])
                asset_returns[asset]  = ar
                # use portfolio weight if available, else 0
                asset_contribs[asset] = ar  # raw; weighting done in template/JS

        result[sc_id] = {
            "portfolio_return":   float(row.get("portfolio_total_return", 0)),
            "stress_type":        row.get("stress_type", ""),
            "asset_returns":      asset_returns,
            "asset_contributions": asset_contribs,
            "description":        row.get("scenario_description", ""),
        }
    return _to_python(result)


def _drawdown_context() -> dict:
    """Build drawdown series for the chart: {dates: [...], spx: [...], ndx: [...], …}."""
    prices = _load_csv("features/merged_macro_market.csv")
    if prices is None:
        # try the raw features file
        prices = _load_csv("features/feature_engineered_dataset.csv")
    if prices is None:
        return {}

    date_col = "date" if "date" in prices.columns else prices.columns[0]
    prices[date_col] = pd.to_datetime(prices[date_col])
    prices = prices.sort_values(date_col).reset_index(drop=True)

    result: dict = {"dates": [str(d)[:10] for d in prices[date_col]]}

    col_map = {
        "spx":  ["spx_return",  "SPX_return",  "spx_price"],
        "ndx":  ["ndx_return",  "NDX_return",  "ndx_price"],
        "gold": ["gold_return", "GOLD_return", "gold_price"],
        "btc":  ["btc_return",  "BTC_return",  "btc_price"],
    }

    for asset, candidates in col_map.items():
        col = next((c for c in candidates if c in prices.columns), None)
        if col is None:
            continue
        series = prices[col].fillna(0).astype(float)

        # If return series, build cumulative
        if "return" in col.lower():
            cum = (1 + series).cumprod()
        else:
            cum = series / series.iloc[0]

        # Drawdown = cum / running_max − 1
        dd = (cum / cum.cummax() - 1).tolist()
        result[asset] = [round(v, 6) for v in dd]

    return _to_python(result)


def _factor_exposures_context() -> dict:
    """Load portfolio-level factor exposures from explanation JSON."""
    expl = _load_json("explainability/portfolio_factor_exposures.json")
    if expl is None:
        # try narrative engine output
        narr = _load_json("explainability/narrative_payload.json")
        if narr and "factor_exposures" in narr:
            return narr["factor_exposures"]
        return {}
    return _to_python(expl)


def _shap_context() -> dict:
    """Load per-asset SHAP importance (top features)."""
    result = {}
    for asset in ["spx", "ndx", "gold", "btc"]:
        data = _load_json(f"explainability/{asset}_shap_importance.json")
        if data:
            result[asset] = data
    return _to_python(result)


def _narrative_context() -> dict:
    """Load NarrativeEngine plain-English output."""
    payload = _load_json("explainability/narrative_payload.json")
    if payload and "narratives" in payload:
        return payload["narratives"]
    return {}


def _worst_scenario(stress_data: dict, portfolio_data: dict) -> Optional[dict]:
    """Find the worst scenario and build its detailed breakdown."""
    if not stress_data or not portfolio_data:
        return None

    weights = portfolio_data.get("weights", {}).get("regime_adjusted", {})
    worst_id = min(stress_data, key=lambda k: stress_data[k]["portfolio_return"])
    sc = stress_data[worst_id]

    asset_contribs = {}
    for asset, ar in sc["asset_returns"].items():
        asset_contribs[asset] = weights.get(asset, 0) * ar

    return _to_python({
        "id":                worst_id,
        "name":              worst_id.replace("_", " ").title(),
        "portfolio_return":  sc["portfolio_return"],
        "asset_returns":     sc["asset_returns"],
        "asset_contributions": asset_contribs,
        "weights":           weights,
    })


# ── Scenario list helper ───────────────────────────────────────────────────────

def _scenario_list(stress_data: dict) -> list:
    """Convert stress_data dict to sorted list for template rendering."""
    result = []
    for sc_id, sc in stress_data.items():
        result.append({
            "scenario_id":    sc_id,
            "name":           sc_id.replace("_", " ").title(),
            "portfolio_return": sc["portfolio_return"],
            "stress_type":    sc["stress_type"],
            "description":    sc["description"],
        })
    result.sort(key=lambda x: x["portfolio_return"])
    return result


def _user_portfolio_context() -> Optional[dict]:
    """Load the user-saved portfolio (from Portfolio Builder) if it exists."""
    data = _load_json("portfolio/user_portfolio.json")
    if not data:
        return None
    return _to_python(data)


def _live_data_context() -> dict:
    """Load live data freshness metadata and latest prices for the dashboard."""
    freshness = _load_json("live/data_freshness.json") or {}
    prices    = _load_json("live/latest_prices.json")  or {}

    # Build per-frequency display info
    FREQ_META = {
        "hourly": {
            "label":   "Hourly",
            "source":  "Yahoo Finance",
            "url":     "https://finance.yahoo.com",
            "assets":  "SPX, NDX, Gold, BTC, VIX",
            "icon":    "⚡",
        },
        "daily": {
            "label":   "Daily",
            "source":  "Yahoo Finance",
            "url":     "https://finance.yahoo.com",
            "assets":  "SPX, NDX, Gold, BTC, VIX, US Yields, DXY, EUR/USD, GBP/USD, QQQ",
            "icon":    "📅",
        },
        "monthly": {
            "label":   "Monthly",
            "source":  "FRED (Federal Reserve)",
            "url":     "https://fred.stlouisfed.org",
            "assets":  "CPI, Fed Funds, HY Spread, TIPS 10Y, Breakeven 10Y, ECB Rate, US 2Y/10Y Yield",
            "icon":    "📊",
        },
    }

    feeds = []
    for freq, meta in FREQ_META.items():
        info = freshness.get(freq, {})
        feeds.append({
            **meta,
            "status":       info.get("status", "never"),
            "last_updated": info.get("last_updated", "Never"),
            "detail":       info.get("detail", "Not yet fetched"),
        })

    return _to_python({
        "feeds":      feeds,
        "prices":     prices.get("prices", {}),
        "fetched_at": prices.get("fetched_at", None),
    })


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/api/live-data/refresh/{feed_type}")
async def refresh_live_data(feed_type: str):
    """Manually trigger a live-data pull for one feed frequency."""
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
        loop = asyncio.get_running_loop()
        ok = await loop.run_in_executor(None, _run)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    freshness_path = BACKEND_ROOT / "data" / "live" / "data_freshness.json"
    prices_path    = BACKEND_ROOT / "data" / "live" / "latest_prices.json"

    freshness = json.loads(freshness_path.read_text()) if freshness_path.exists() else {}
    prices_raw = json.loads(prices_path.read_text()) if prices_path.exists() else {}

    info = freshness.get(feed_type, {})
    return JSONResponse({
        "ok":           ok,
        "feed_type":    feed_type,
        "last_updated": info.get("last_updated"),
        "status":       info.get("status", "error"),
        "detail":       info.get("detail", ""),
        "fetched_at":   prices_raw.get("fetched_at"),
        "prices":       prices_raw.get("prices", {}) if feed_type == "hourly" else {},
    })


@router.get("/", response_class=HTMLResponse)
@router.get("/dashboard", response_class=HTMLResponse)
async def page_dashboard(request: Request):
    portfolio_data   = _portfolio_context()
    regime_data      = _regime_context()
    stress_data      = _stress_context()
    drawdown_data    = _drawdown_context()
    factor_exposures = _factor_exposures_context()
    user_portfolio   = _user_portfolio_context()
    live_data        = _live_data_context()

    resp = templates.TemplateResponse(request, "dashboard.html", {
        "active_page":      "dashboard",
        "portfolio_data":   portfolio_data or None,
        "regime_data":      regime_data    or None,
        "stress_data":      stress_data    or None,
        "drawdown_data":    drawdown_data  or None,
        "factor_exposures": factor_exposures or None,
        "user_portfolio":   user_portfolio or None,
        "live_data":        live_data,
    })
    resp.headers.update(NO_CACHE_HEADERS)
    return resp


@router.get("/portfolio", response_class=HTMLResponse)
async def page_portfolio(request: Request):
    portfolio_data = _portfolio_context()
    regime_data    = _regime_context()

    return templates.TemplateResponse(request, "portfolio.html", {
        "active_page":    "portfolio",
        "portfolio_data": portfolio_data or None,
        "regime_data":    regime_data    or None,
    })


@router.get("/scenarios", response_class=HTMLResponse)
async def page_scenarios(request: Request):
    portfolio_data = _portfolio_context()
    regime_data    = _regime_context()
    stress_data    = _stress_context()
    scenarios      = _scenario_list(stress_data) if stress_data else []

    return templates.TemplateResponse(request, "scenario.html", {
        "active_page":    "scenarios",
        "portfolio_data": portfolio_data or None,
        "regime_data":    regime_data    or None,
        "scenarios":      scenarios,
        "stress_data":    stress_data    or None,
    })


@router.get("/results", response_class=HTMLResponse)
async def page_results(request: Request):
    portfolio_data   = _portfolio_context()
    regime_data      = _regime_context()
    stress_data      = _stress_context()
    drawdown_data    = _drawdown_context()
    factor_exposures = _factor_exposures_context()
    shap_data        = _shap_context()
    narrative        = _narrative_context()
    worst            = _worst_scenario(stress_data, portfolio_data) if stress_data else None

    # Drawdown stats per asset
    dd_stats = {}
    if drawdown_data:
        for asset in ["spx", "ndx", "gold", "btc"]:
            if asset in drawdown_data:
                series = drawdown_data[asset]
                valid  = [v for v in series if v is not None]
                if valid:
                    dd_stats[asset] = {
                        "max_drawdown": round(min(valid), 4),
                        "avg_drawdown": round(sum(valid) / len(valid), 4),
                    }

    return templates.TemplateResponse(request, "results.html", {
        "active_page":     "results",
        "portfolio_data":  portfolio_data  or None,
        "regime_data":     regime_data     or None,
        "stress_data":     stress_data     or None,
        "drawdown_data":   drawdown_data   or None,
        "factor_exposures": factor_exposures or None,
        "shap_data":       shap_data       or None,
        "narrative":       narrative       or None,
        "worst_scenario":  worst,
        "drawdown_stats":  dd_stats        or None,
    })


@router.get("/methodology", response_class=HTMLResponse)
async def page_methodology(request: Request):
    portfolio_data = _portfolio_context()
    regime_data    = _regime_context()

    return templates.TemplateResponse(request, "methodology.html", {
        "active_page":    "methodology",
        "portfolio_data": portfolio_data or None,
        "regime_data":    regime_data    or None,
    })
