"""
Live Data Fetcher
=================
Pulls market and macro data from:
  - Yahoo Finance (yfinance) — asset prices, VIX, yields, FX   [Hourly + Daily]
  - FRED API (Federal Reserve)  — macro indicators              [Monthly]

Update schedule:
  Hourly  : SPX, NDX, Gold, BTC prices + VIX
  Daily   : All hourly + US yields, DXY, EUR/USD, GBP/USD, QQQ
  Monthly : CPI, Fed Funds, HY Spread, TIPS, Breakeven, ECB rate (via FRED)

After each fetch the relevant pipeline phases are re-run automatically:
  Hourly  → save live_prices.json  (no pipeline re-run — models are monthly)
  Daily   → append features → re-run Phase 7
  Monthly → append features → re-run Phase 5 + Phase 7
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BACKEND_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = BACKEND_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
LIVE_DIR     = DATA_DIR / "live"
LIVE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BACKEND_ROOT))

FEATURES_FILE     = FEATURES_DIR / "features_monthly_full_history.csv"
FEATURES_BTC_FILE = FEATURES_DIR / "features_monthly_btc.csv"
FRESHNESS_FILE    = LIVE_DIR / "data_freshness.json"
LIVE_PRICES_FILE  = LIVE_DIR / "latest_prices.json"

# ── yfinance ticker map ────────────────────────────────────────────────────────
HOURLY_TICKERS = {
    # Assets
    "spx":    "^GSPC",
    "ndx":    "^NDX",
    "gold":   "GC=F",
    "btc":    "BTC-USD",
    "vix":    "^VIX",
    # Major FX
    "eurusd": "EURUSD=X",
    "gbpusd": "GBPUSD=X",
    "dxy":    "DX-Y.NYB",
    # USD Crosses
    "usdjpy": "JPY=X",
    "usdchf": "CHF=X",
    "audusd": "AUDUSD=X",
    "usdcad": "CAD=X",
    "nzdusd": "NZDUSD=X",
    "usdcny": "CNY=X",
    "usdsek": "SEK=X",
    "usdnok": "NOK=X",
    # BDT pairs
    "usdbdt": "BDT=X",
    "gbpbdt": "GBPBDT=X",
    "eurbdt": "EURBDT=X",
    # Note: goldbdt is derived (gold × usdbdt), not fetched directly
}

DAILY_EXTRA_TICKERS = {
    "us10y":  "^TNX",       # US 10-year yield (%)
    "us2y":   "^FVX",       # US 5-year as 2Y proxy (closest on yfinance)
    "dxy":    "DX-Y.NYB",   # US Dollar Index
    "eurusd": "EURUSD=X",
    "gbpusd": "GBPUSD=X",
    "qqq":    "QQQ",
}

# FRED series IDs (requires FRED_API_KEY env variable)
FRED_SERIES = {
    "us_cpi_yoy":              "CPIAUCSL",
    "fed_funds_level":         "FEDFUNDS",
    "high_yield_spread":       "BAMLH0A0HYM2",
    "tips_10y_level":          "DFII10",
    "breakeven_10y_level":     "T10YIE",
    "us2y_yield":              "DGS2",
    "us10y_yield":             "DGS10",
    "ecb_level":               "ECBDFR",
    "us_cpi_yoy_raw":          "CPIAUCSL",   # raw level for YoY calc
}

# ── Freshness helpers ──────────────────────────────────────────────────────────

def _load_freshness() -> dict:
    if FRESHNESS_FILE.exists():
        with open(FRESHNESS_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_freshness(freq: str, status: str, detail: str = "") -> None:
    data = _load_freshness()
    data[freq] = {
        "last_updated": datetime.utcnow().isoformat(),
        "status":       status,
        "detail":       detail,
    }
    with open(FRESHNESS_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ── Hourly update ─────────────────────────────────────────────────────────────

def run_hourly() -> bool:
    """
    Fetch latest prices for SPX, NDX, Gold, BTC, VIX from Yahoo Finance.
    Saves to data/live/latest_prices.json.
    No pipeline re-run (models are monthly).
    """
    logger.info("[HOURLY] Fetching live prices from Yahoo Finance...")
    try:
        prices = {}
        for name, ticker in HOURLY_TICKERS.items():
            try:
                tk   = yf.Ticker(ticker)
                hist = tk.history(period="2d", interval="1h")
                if hist.empty:
                    hist = tk.history(period="5d")
                if hist.empty:
                    logger.warning(f"  No data for {ticker}")
                    continue

                latest_close = float(hist["Close"].iloc[-1])
                prev_close   = float(hist["Close"].iloc[-2]) if len(hist) > 1 else latest_close
                day_return   = (latest_close - prev_close) / prev_close if prev_close else 0.0

                prices[name] = {
                    "ticker":      ticker,
                    "price":       round(latest_close, 4),
                    "prev_close":  round(prev_close, 4),
                    "day_return":  round(day_return, 6),
                    "day_pct":     round(day_return * 100, 3),
                    "as_of":       hist.index[-1].isoformat(),
                }
                logger.info(f"  {name.upper():5s}  {latest_close:>12.2f}  ({day_return:+.2%})")
            except Exception as e:
                logger.warning(f"  Failed {ticker}: {e}")

        if not prices:
            _save_freshness("hourly", "error", "No price data returned")
            return False

        # Derive Gold/BDT = Gold (USD) × USD/BDT
        if "gold" in prices and "usdbdt" in prices:
            gold_usd      = prices["gold"]["price"]
            gold_prev_usd = prices["gold"]["prev_close"]
            bdt_rate      = prices["usdbdt"]["price"]
            bdt_prev      = prices["usdbdt"]["prev_close"]
            gold_bdt      = gold_usd  * bdt_rate
            gold_bdt_prev = gold_prev_usd * bdt_prev
            bdt_day_ret   = (gold_bdt - gold_bdt_prev) / gold_bdt_prev if gold_bdt_prev else 0.0
            prices["goldbdt"] = {
                "ticker":     "GC=F \u00d7 BDT=X",
                "price":      round(gold_bdt, 2),
                "prev_close": round(gold_bdt_prev, 2),
                "day_return": round(bdt_day_ret, 6),
                "day_pct":    round(bdt_day_ret * 100, 3),
                "as_of":      prices["gold"]["as_of"],
            }
            logger.info(f"  GOLDBDT  {gold_bdt:>12,.2f}  ({bdt_day_ret:+.2%})  [derived]")

        payload = {
            "prices":      prices,
            "fetched_at":  datetime.utcnow().isoformat(),
            "source":      "Yahoo Finance (yfinance)",
            "frequency":   "hourly",
        }
        with open(LIVE_PRICES_FILE, "w") as f:
            json.dump(payload, f, indent=2)

        _save_freshness("hourly", "ok",
                        f"Fetched {len(prices)} assets — {datetime.utcnow().strftime('%H:%M UTC')}")
        logger.info(f"[HOURLY] Done — saved {len(prices)} prices.")
        return True

    except Exception as e:
        logger.error(f"[HOURLY] Failed: {e}")
        _save_freshness("hourly", "error", str(e))
        return False


# ── Daily update ──────────────────────────────────────────────────────────────

def run_daily() -> bool:
    """
    Fetch end-of-day prices + yields/FX from Yahoo Finance.
    Appends a new row to features_monthly_full_history.csv (month-end only).
    Always re-runs Phase 7 for fresh portfolio metrics.
    """
    logger.info("[DAILY] Fetching daily market data from Yahoo Finance...")
    try:
        all_tickers = {**HOURLY_TICKERS, **DAILY_EXTRA_TICKERS}
        raw = {}

        for name, ticker in all_tickers.items():
            try:
                hist = yf.Ticker(ticker).history(period="5d")
                if hist.empty:
                    continue
                raw[name] = hist["Close"]
                logger.info(f"  {name:10s} latest={float(hist['Close'].iloc[-1]):.4f}")
            except Exception as e:
                logger.warning(f"  Failed {ticker}: {e}")

        if "spx" not in raw:
            _save_freshness("daily", "error", "SPX data unavailable")
            return False

        # ── Build latest daily feature row ────────────────────────────────────
        today = datetime.utcnow().date()
        # Load existing features to compute rolling stats
        features_df = pd.read_csv(FEATURES_FILE, parse_dates=["date"])
        features_df = features_df.sort_values("date").reset_index(drop=True)

        # Only append at month end (or if today is beyond last row's month)
        last_date = pd.to_datetime(features_df["date"].iloc[-1]).date()
        if today.year == last_date.year and today.month == last_date.month:
            logger.info("[DAILY] Still same month as last feature row — skipping append.")
            _save_freshness("daily", "ok",
                            f"Prices updated, no new monthly row yet — {today}")
        else:
            # Compute returns for the new month
            new_row = _build_daily_feature_row(raw, features_df, today)
            if new_row is not None:
                new_df = pd.concat([features_df, pd.DataFrame([new_row])], ignore_index=True)
                new_df.to_csv(FEATURES_FILE, index=False)
                logger.info(f"[DAILY] Appended new feature row for {today}")

            _save_freshness("daily", "ok",
                            f"Appended new monthly row for {today.strftime('%Y-%m')}")

        # Always run Phase 7 to refresh portfolio metrics with updated data
        _run_phase7()
        return True

    except Exception as e:
        logger.error(f"[DAILY] Failed: {e}")
        _save_freshness("daily", "error", str(e))
        return False


def _build_daily_feature_row(raw: dict, hist_df: pd.DataFrame, today) -> Optional[dict]:
    """Build a feature row from latest daily prices, using rolling windows from history."""
    try:
        month_str = today.strftime("%Y-%m-01")

        def _ret(series, n=1):
            if series is None or len(series) < n + 1:
                return 0.0
            return float((series.iloc[-1] - series.iloc[-(n+1)]) / series.iloc[-(n+1)])

        def _vol(series, n):
            if series is None or len(series) < n:
                return float(hist_df[f"spx_vol_{n}m"].iloc[-1]) if f"spx_vol_{n}m" in hist_df.columns else 0.02
            rets = series.pct_change().dropna().tail(n)
            return float(rets.std() * (252 ** 0.5 / 12 ** 0.5))

        # Returns (1-month approximation from latest close vs ~22 trading days ago)
        spx_ret  = _ret(raw.get("spx"),  22)
        ndx_ret  = _ret(raw.get("ndx"),  22)
        gold_ret = _ret(raw.get("gold"), 22)
        btc_ret  = _ret(raw.get("btc"),  22)

        # Vols (annualised from daily std)
        spx_vol3  = _vol(raw.get("spx"),  66)
        spx_vol6  = _vol(raw.get("spx"),  132)
        ndx_vol3  = _vol(raw.get("ndx"),  66)
        ndx_vol6  = _vol(raw.get("ndx"),  132)
        gold_vol3 = _vol(raw.get("gold"), 66)
        gold_vol6 = _vol(raw.get("gold"), 132)

        # Levels
        vix_level   = float(raw["vix"].iloc[-1])   if "vix"   in raw else float(hist_df["vix_level"].iloc[-1])
        us10y_yield = float(raw["us10y"].iloc[-1]) / 100 if "us10y" in raw else float(hist_df["us10y_yield"].iloc[-1])
        us2y_yield  = float(raw["us2y"].iloc[-1])  / 100 if "us2y"  in raw else float(hist_df["us2y_yield"].iloc[-1])
        yield_spread = us10y_yield - us2y_yield
        dxy_level   = float(raw["dxy"].iloc[-1])   if "dxy"   in raw else float(hist_df["dxy_level"].iloc[-1])

        eurusd_ret = _ret(raw.get("eurusd"), 22)
        gbpusd_ret = _ret(raw.get("gbpusd"), 22)
        qqq_ret    = _ret(raw.get("qqq"),    22)

        # Carry forward monthly values we can't recompute from daily prices
        last = hist_df.iloc[-1]

        row = {
            "date":                    month_str,
            "spx_return":              spx_ret,
            "ndx_return":              ndx_ret,
            "gold_return":             gold_ret,
            "gold_return_3m":          gold_ret * 3,
            "gold_vol_3m":             gold_vol3,
            "gold_vol_6m":             gold_vol6,
            "gold_drawdown":           0.0,
            "gold_max_dd_6m":          0.0,
            "eurusd_return":           eurusd_ret,
            "gbpusd_return":           gbpusd_ret,
            "spx_vol_3m":              spx_vol3,
            "spx_vol_6m":              spx_vol6,
            "ndx_vol_3m":              ndx_vol3,
            "ndx_vol_6m":              ndx_vol6,
            "vix_level":               vix_level,
            "us2y_yield":              us2y_yield,
            "us10y_yield":             us10y_yield,
            "yield_spread":            yield_spread,
            "us_cpi_yoy":              float(last["us_cpi_yoy"]),        # carried forward
            "high_yield_spread":       float(last["high_yield_spread"]), # carried forward
            "vix_spike":               1 if vix_level > 30 else 0,
            "spx_drawdown":            0.0,
            "spx_max_dd_6m":           0.0,
            "ndx_drawdown":            0.0,
            "ndx_max_dd_6m":           0.0,
            "ecb_level":               float(last["ecb_level"]),         # carried forward
            "ecb_yoy":                 float(last["ecb_yoy"]),
            "fed_funds_level":         float(last["fed_funds_level"]),   # carried forward
            "fed_funds_change_1m":     0.0,
            "tips_10y_level":          float(last["tips_10y_level"]),    # carried forward
            "tips_10y_change_1m":      0.0,
            "breakeven_10y_level":     float(last["breakeven_10y_level"]),
            "breakeven_10y_change_1m": 0.0,
            "real_yield_tips":         float(last["real_yield_tips"]),
            "real_yield_tips_change_1m": 0.0,
            "dxy_level":               dxy_level,
            "dxy_return":              _ret(raw.get("dxy"), 22),
            "dxy_return_3m":           _ret(raw.get("dxy"), 66),
            "qqq_return":              qqq_ret,
        }
        return row

    except Exception as e:
        logger.error(f"[DAILY] Feature row build failed: {e}")
        return None


# ── Monthly update (FRED) ─────────────────────────────────────────────────────

def run_monthly(fred_api_key: Optional[str] = None) -> bool:
    """
    Fetch monthly macro indicators from FRED.
    Appends new data to features CSV and re-runs Phase 5 + Phase 7.

    Requires FRED_API_KEY environment variable or fred_api_key argument.
    Register free at: https://fred.stlouisfed.org/docs/api/api_key.html
    """
    key = fred_api_key or os.environ.get("FRED_API_KEY", "")
    if not key:
        msg = ("FRED_API_KEY not set — skipping monthly macro update. "
               "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html "
               "and set it as environment variable FRED_API_KEY.")
        logger.warning(f"[MONTHLY] {msg}")
        _save_freshness("monthly", "no_api_key", msg)
        return False

    logger.info("[MONTHLY] Fetching macro data from FRED...")
    try:
        from fredapi import Fred
        fred = Fred(api_key=key)

        features_df = pd.read_csv(FEATURES_FILE, parse_dates=["date"])
        features_df = features_df.sort_values("date").reset_index(drop=True)
        last_date   = pd.to_datetime(features_df["date"].iloc[-1])

        # Fetch each FRED series
        # Fetch from 3 months before last feature date to handle publication lags
        fetch_start = (last_date - pd.DateOffset(months=3)).date()
        fred_data: dict[str, pd.Series] = {}
        for col, series_id in FRED_SERIES.items():
            try:
                s = fred.get_series(series_id, observation_start=str(fetch_start))
                if len(s) == 0:
                    logger.warning(f"  {series_id}: no data returned from FRED")
                    continue
                fred_data[col] = s
                logger.info(f"  {series_id:20s} → {len(s)} obs, latest={float(s.iloc[-1]):.4f}")
            except Exception as e:
                logger.warning(f"  Failed to fetch {series_id}: {e}")

        if not fred_data:
            _save_freshness("monthly", "error", "All FRED fetches failed")
            return False

        # Build new monthly rows from FRED data
        new_rows_added = 0
        existing_dates = set(features_df["date"].dt.to_period("M").astype(str))

        # Get dates available across all series
        all_dates = sorted(set().union(*[set(s.index) for s in fred_data.values()]))
        for dt in all_dates:
            period = pd.Period(dt, "M").strftime("%Y-%m")
            if period in existing_dates:
                continue
            if dt > datetime.utcnow():
                continue

            # Get last known row for carry-forward
            last = features_df.iloc[-1]

            def _get(col_name: str, default=None):
                s = fred_data.get(col_name)
                if s is None:
                    return default if default is not None else float(last.get(col_name, 0))
                val = s.get(dt, None)
                return float(val) if val is not None and not pd.isna(val) else (
                    default if default is not None else float(last.get(col_name, 0)))

            us10y = _get("us10y_yield") / 100
            us2y  = _get("us2y_yield")  / 100
            cpi_level = _get("us_cpi_yoy_raw")
            # YoY CPI: need level from 12 months ago
            cpi_12m_ago = fred_data.get("us_cpi_yoy_raw")
            if cpi_12m_ago is not None:
                dt_12m = dt - pd.DateOffset(months=12)
                prev_level = float(cpi_12m_ago.asof(dt_12m)) if hasattr(cpi_12m_ago, "asof") else cpi_level
                cpi_yoy = (cpi_level - prev_level) / prev_level if prev_level else 0.0
            else:
                cpi_yoy = float(last.get("us_cpi_yoy", 0))

            row = dict(last)  # start with carry-forward
            row.update({
                "date":                    dt.strftime("%Y-%m-01"),
                "us2y_yield":              us2y,
                "us10y_yield":             us10y,
                "yield_spread":            us10y - us2y,
                "us_cpi_yoy":              cpi_yoy,
                "fed_funds_level":         _get("fed_funds_level") / 100,
                "fed_funds_change_1m":     _get("fed_funds_level") / 100 - float(last.get("fed_funds_level", 0)),
                "high_yield_spread":       _get("high_yield_spread") / 100,
                "tips_10y_level":          _get("tips_10y_level") / 100,
                "tips_10y_change_1m":      _get("tips_10y_level") / 100 - float(last.get("tips_10y_level", 0)),
                "breakeven_10y_level":     _get("breakeven_10y_level") / 100,
                "breakeven_10y_change_1m": _get("breakeven_10y_level") / 100 - float(last.get("breakeven_10y_level", 0)),
                "ecb_level":               _get("ecb_level") / 100,
                "ecb_yoy":                 _get("ecb_level") / 100 - float(last.get("ecb_level", 0)),
                "real_yield_tips":         _get("tips_10y_level") / 100,
                "real_yield_tips_change_1m": _get("tips_10y_level") / 100 - float(last.get("tips_10y_level", 0)),
            })

            features_df = pd.concat([features_df, pd.DataFrame([row])], ignore_index=True)
            existing_dates.add(period)
            new_rows_added += 1
            logger.info(f"  Added row for {period}")

        if new_rows_added > 0:
            features_df = features_df.sort_values("date").reset_index(drop=True)
            features_df.to_csv(FEATURES_FILE, index=False)
            logger.info(f"[MONTHLY] Appended {new_rows_added} new rows to features CSV")
            # Re-run Phase 5 (regime) then Phase 7
            _run_phase5()
        else:
            logger.info("[MONTHLY] No new rows to append — features up to date")

        _run_phase7()
        _save_freshness("monthly", "ok",
                        f"FRED sync complete — {new_rows_added} new rows — {datetime.utcnow().strftime('%Y-%m-%d')}")
        return True

    except Exception as e:
        logger.error(f"[MONTHLY] Failed: {e}")
        _save_freshness("monthly", "error", str(e))
        return False


# ── Pipeline runners ───────────────────────────────────────────────────────────

def _run_phase5() -> None:
    """Re-run regime detection (Phase 5) with latest features."""
    try:
        logger.info("[PIPELINE] Running Phase 5 (regime detection)...")
        import subprocess
        result = subprocess.run(
            [sys.executable, str(BACKEND_ROOT / "run_phase3.py")],
            capture_output=True, text=True, cwd=str(BACKEND_ROOT)
        )
        # Phase 5 is actually run_phase5.py
        result = subprocess.run(
            [sys.executable, str(BACKEND_ROOT / "run_phase5.py")],
            capture_output=True, text=True, cwd=str(BACKEND_ROOT)
        )
        if result.returncode == 0:
            logger.info("[PIPELINE] Phase 5 completed successfully")
        else:
            logger.warning(f"[PIPELINE] Phase 5 warning: {result.stderr[-500:]}")
    except Exception as e:
        logger.error(f"[PIPELINE] Phase 5 failed: {e}")


def _run_phase7() -> None:
    """Re-run portfolio construction (Phase 7) with latest data."""
    try:
        logger.info("[PIPELINE] Running Phase 7 (portfolio construction)...")
        from services.portfolio_engine import PortfolioEngine
        engine = PortfolioEngine(BACKEND_ROOT)
        engine.run()
        logger.info("[PIPELINE] Phase 7 completed successfully")
    except Exception as e:
        logger.error(f"[PIPELINE] Phase 7 failed: {e}")
