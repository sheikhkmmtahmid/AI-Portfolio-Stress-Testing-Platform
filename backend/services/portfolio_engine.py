"""
Phase 7 – Portfolio Construction & Stress Testing Engine

Consumes outputs from:
  - Phase 6   : asset-level predictions + metrics
  - Phase 5/5.5: regime labels + transition matrix
  - Phase 3   : historical return features
  - Phase 4   : macro scenario dataset

Produces:
  - expected_returns.csv
  - covariance_matrix.csv
  - portfolio_weights.csv
  - portfolio_weights_regime_adjusted.csv
  - stress_test_results.csv
  - portfolio_metrics.json

NO model retraining. NO feature engineering. Fully reproducible.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ASSETS = ["spx", "ndx", "gold", "btc"]

RETURN_COLS = {
    "spx": "spx_return",
    "ndx": "ndx_return",
    "gold": "gold_return",
    "btc": "btc_return",
}

# Annualisation factor (monthly data)
MONTHS_PER_YEAR = 12

# Regime-based weight tilts (additive, scaled by regime_confidence)
REGIME_TILTS: Dict[str, Dict[str, float]] = {
    "calm":             {"spx":  0.05, "ndx":  0.05, "gold": -0.10},
    "inflation_stress": {"spx": -0.10, "ndx": -0.10, "gold":  0.20},
    "credit_stress":    {"spx": -0.15, "ndx": -0.15, "gold":  0.30},
    "crisis":           {"spx": -0.20, "ndx": -0.20, "gold":  0.40},
}

# Historical stress periods: (label, start, end)
HISTORICAL_STRESS_PERIODS: List[Tuple[str, str, str]] = [
    ("dot_com_crash",      "2001-03-01", "2002-10-01"),
    ("gfc_peak",           "2008-09-01", "2009-03-01"),
    ("gfc_recovery",       "2009-03-01", "2009-12-01"),
    ("euro_crisis",        "2011-07-01", "2012-07-01"),
    ("covid_crash",        "2020-02-01", "2020-04-01"),
    ("covid_recovery",     "2020-04-01", "2020-12-01"),
    ("inflation_spike_22", "2022-01-01", "2022-12-01"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Timer helper
# ─────────────────────────────────────────────────────────────────────────────

class _Timer:
    def __init__(self) -> None:
        self._t0 = time.time()

    def log(self, tag: str, msg: str) -> None:
        elapsed = time.time() - self._t0
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        logger.info(f"[{h:02d}:{m:02d}:{s:02d}] [{tag}] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Paths dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Phase7Paths:
    backend_root: Path

    # Inputs
    features_file: Path = field(init=False)
    features_btc_file: Path = field(init=False)
    regime_file: Path = field(init=False)
    regime_summary_file: Path = field(init=False)
    scenarios_file: Path = field(init=False)
    phase6_metrics_file: Path = field(init=False)
    phase61_summary_file: Path = field(init=False)
    predictions_dir: Path = field(init=False)

    # Outputs
    portfolio_dir: Path = field(init=False)
    expected_returns_file: Path = field(init=False)
    covariance_file: Path = field(init=False)
    gold_fx_corr_file: Path = field(init=False)
    weights_file: Path = field(init=False)
    weights_regime_file: Path = field(init=False)
    stress_results_file: Path = field(init=False)
    metrics_file: Path = field(init=False)

    def __post_init__(self) -> None:
        r = self.backend_root

        self.features_file         = r / "data" / "features" / "features_monthly_full_history.csv"
        self.features_btc_file     = r / "data" / "features" / "features_monthly_btc.csv"
        self.regime_file           = r / "data" / "regimes"  / "regime_dataset.csv"
        self.regime_summary_file   = r / "data" / "regimes"  / "regime_summary.csv"
        self.scenarios_file        = r / "data" / "scenarios" / "scenario_dataset.csv"
        self.phase6_metrics_file   = r / "models" / "phase6"   / "phase6_metrics.json"
        self.phase61_summary_file  = r / "models" / "phase6_1" / "phase6_1_summary.json"
        self.predictions_dir       = r / "models" / "phase6"   / "predictions"

        self.portfolio_dir         = r / "data" / "portfolio"
        self.expected_returns_file = self.portfolio_dir / "expected_returns.csv"
        self.covariance_file       = self.portfolio_dir / "covariance_matrix.csv"
        self.gold_fx_corr_file     = self.portfolio_dir / "gold_fx_correlations.json"
        self.weights_file          = self.portfolio_dir / "portfolio_weights.csv"
        self.weights_regime_file   = self.portfolio_dir / "portfolio_weights_regime_adjusted.csv"
        self.stress_results_file   = self.portfolio_dir / "stress_test_results.csv"
        self.metrics_file          = self.portfolio_dir / "portfolio_metrics.json"

        self.portfolio_dir.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main engine
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioEngine:
    """
    Phase 7: Portfolio Construction & Stress Testing.

    Consumes Phase 3/5/6 outputs. No model retraining.
    """

    def __init__(self, backend_root: str | Path) -> None:
        self.paths = Phase7Paths(backend_root=Path(backend_root))
        self.timer = _Timer()

        # Cached data (populated during run)
        self._features_df: Optional[pd.DataFrame] = None
        self._regime_df: Optional[pd.DataFrame] = None
        self._phase6_metrics: Optional[dict] = None
        self._phase61_summary: Optional[dict] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> dict:
        self.timer.log("START", "Phase 7 started")

        self._load_inputs()

        rf_monthly = self._get_risk_free_rate_monthly()
        self.timer.log("INFO", f"Risk-free rate (monthly): {rf_monthly:.6f}  ({rf_monthly * MONTHS_PER_YEAR * 100:.2f}% p.a.)")

        # 7.1 Expected returns
        expected_returns, model_weights, er_timeseries = self.compute_expected_returns()
        self.timer.log("7.1", f"Expected returns: spx={expected_returns['spx']:.4f}  ndx={expected_returns['ndx']:.4f}  gold={expected_returns['gold']:.4f}")

        # 7.2 Covariance matrix
        cov_matrix = self.compute_covariance_matrix()
        self.timer.log("7.2", "Covariance matrix computed (Ledoit-Wolf shrinkage, 36-month window)")

        # 7.2b Gold-FX correlations
        gold_fx_corrs = self.compute_gold_fx_correlations()
        self.timer.log("7.2b", f"Gold-FX correlations: gold_dxy={gold_fx_corrs.get('gold_dxy', 'n/a')}  gold_eurusd={gold_fx_corrs.get('gold_eurusd', 'n/a')}")

        # 7.3 Optimise portfolio
        base_weights = self.optimize_portfolio(expected_returns, cov_matrix, risk_free_rate=rf_monthly)
        self.timer.log("7.3", f"Optimised weights: " + "  ".join(f"{a}={base_weights[a]:.4f}" for a in ASSETS))

        # 7.4 Regime adjustment
        adj_weights, current_regime, regime_conf = self.apply_regime_adjustment(base_weights)
        self.timer.log("7.4", f"Regime={current_regime} (conf={regime_conf:.2f})  Adjusted: " + "  ".join(f"{a}={adj_weights[a]:.4f}" for a in ASSETS))

        # 7.5 Stress tests (on regime-adjusted weights)
        stress_df = self.run_stress_tests(adj_weights, cov_matrix)
        self.timer.log("7.5", f"Stress tests complete: {len(stress_df)} scenarios")

        # 7.6 Portfolio metrics
        metrics = self.compute_portfolio_metrics(adj_weights, expected_returns, cov_matrix, rf_monthly)
        self.timer.log("7.6", f"Sharpe={metrics['sharpe_ratio']:.3f}  VaR95={metrics['var_95_monthly']:.4f}  MaxDD={metrics['max_drawdown']:.4f}")

        self.timer.log("END", "Phase 7 completed successfully")
        return {
            "expected_returns": expected_returns,
            "model_weights":    model_weights,
            "base_weights":     base_weights.to_dict(),
            "adjusted_weights": adj_weights.to_dict(),
            "current_regime":   current_regime,
            "regime_confidence": regime_conf,
            "portfolio_metrics": metrics,
        }

    # ── 7.1 Expected Return Engine ────────────────────────────────────────────

    def compute_expected_returns(self) -> Tuple[Dict[str, float], Dict[str, dict], pd.DataFrame]:
        """
        Ensemble ElasticNet + XGBoost predictions using inverse-RMSE weighting.
        Uses out-of-sample (test) predictions only — no data leakage.
        Returns:
          - expected_returns : dict {asset: next-period expected return}
          - model_weights    : dict {asset: {model: weight}}
          - timeseries       : DataFrame of all test-set ensemble predictions
        """
        metrics = self._phase6_metrics

        expected_returns: Dict[str, float] = {}
        model_weights:    Dict[str, dict]  = {}
        asset_series:     List[pd.DataFrame] = []

        for asset in ASSETS:
            en_rmse  = metrics[asset]["elastic_net"]["test_rmse"]
            xgb_rmse = metrics[asset]["xgboost"]["test_rmse"]

            # Inverse-RMSE ensemble weights
            en_w  = 1.0 / en_rmse
            xgb_w = 1.0 / xgb_rmse
            total = en_w + xgb_w
            en_w  /= total
            xgb_w /= total

            en_preds  = self._load_predictions(asset, "elastic_net")
            xgb_preds = self._load_predictions(asset, "xgboost")

            # Test-set only
            en_test  = en_preds[en_preds["dataset"]  == "test"][["date", "predicted"]].rename(columns={"predicted": "en_pred"})
            xgb_test = xgb_preds[xgb_preds["dataset"] == "test"][["date", "predicted"]].rename(columns={"predicted": "xgb_pred"})

            merged = pd.merge(en_test, xgb_test, on="date")
            merged["ensemble_pred"] = en_w * merged["en_pred"] + xgb_w * merged["xgb_pred"]
            merged = merged.sort_values("date").reset_index(drop=True)

            # Most recent prediction → expected return for next period
            most_recent_pred = merged["ensemble_pred"].iloc[-1]
            expected_returns[asset] = float(most_recent_pred)
            model_weights[asset]    = {"elastic_net": round(en_w, 4), "xgboost": round(xgb_w, 4),
                                       "en_test_rmse": round(en_rmse, 6), "xgb_test_rmse": round(xgb_rmse, 6)}

            asset_series.append(
                merged[["date", "ensemble_pred"]].rename(columns={"ensemble_pred": f"{asset}_expected_return"})
            )

        # Merge all asset time-series on date
        ts = asset_series[0]
        for df in asset_series[1:]:
            ts = pd.merge(ts, df, on="date", how="outer")
        ts = ts.sort_values("date").reset_index(drop=True)

        # Metadata row
        ts.attrs["model_weights"] = model_weights

        ts.to_csv(self.paths.expected_returns_file, index=False)
        logger.info(f"  Saved expected_returns.csv  ({len(ts)} rows, most recent: {ts['date'].iloc[-1].date() if hasattr(ts['date'].iloc[-1], 'date') else ts['date'].iloc[-1]})")

        return expected_returns, model_weights, ts

    # ── 7.2 Risk Model ────────────────────────────────────────────────────────

    def compute_covariance_matrix(self, window: int = 36) -> pd.DataFrame:
        """
        Ledoit-Wolf shrinkage covariance on the most recent `window` months of returns.
        """
        df = self._features_df.copy()
        return_cols = [RETURN_COLS[a] for a in ASSETS]
        returns = df[return_cols].dropna().tail(window)

        lw = LedoitWolf()
        lw.fit(returns.values)

        cov = pd.DataFrame(lw.covariance_, index=ASSETS, columns=ASSETS)
        cov.to_csv(self.paths.covariance_file)
        logger.info(f"  Saved covariance_matrix.csv  (window={window} months, shrinkage={lw.shrinkage_:.4f})")

        return cov

    def compute_gold_fx_correlations(self, window: int = 36) -> dict:
        """
        Pearson correlations between gold_return and FX factors
        (eurusd_return, gbpusd_return, dxy_return) over the most recent `window` months.
        Saves result to gold_fx_correlations.json.
        """
        df = self._features_df.copy()
        fx_cols = ["gold_return", "eurusd_return", "gbpusd_return", "dxy_return"]
        available = [c for c in fx_cols if c in df.columns]

        if "gold_return" not in available or len(available) < 2:
            logger.warning("  Gold or FX return columns missing — skipping Gold-FX correlation")
            return {}

        subset = df[available].dropna().tail(window)
        corr = subset.corr()

        result = {
            "window_months":   window,
            "n_observations":  len(subset),
            "gold_eurusd": round(float(corr.loc["gold_return", "eurusd_return"]), 4) if "eurusd_return" in corr.columns else None,
            "gold_gbpusd": round(float(corr.loc["gold_return", "gbpusd_return"]), 4) if "gbpusd_return" in corr.columns else None,
            "gold_dxy":    round(float(corr.loc["gold_return", "dxy_return"]),    4) if "dxy_return"    in corr.columns else None,
        }

        with open(self.paths.gold_fx_corr_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info(f"  Saved gold_fx_correlations.json  (window={window} months, n={len(subset)})")

        return result

    # ── 7.3 Portfolio Optimisation ────────────────────────────────────────────

    def optimize_portfolio(
        self,
        expected_returns: Dict[str, float],
        cov_matrix: pd.DataFrame,
        risk_free_rate: float = 0.0,
        max_weight: float = 0.60,
    ) -> pd.Series:
        """
        Maximize Sharpe ratio via SLSQP.
        Constraints: sum(w)=1, 0 <= w <= max_weight.
        """
        n   = len(ASSETS)
        er  = np.array([expected_returns[a] for a in ASSETS])
        cov = cov_matrix.values

        def neg_sharpe(w: np.ndarray) -> float:
            ret = float(np.dot(w, er))
            vol = float(np.sqrt(w @ cov @ w))
            if vol < 1e-12:
                return 0.0
            return -(ret - risk_free_rate) / vol

        result = minimize(
            neg_sharpe,
            x0     = np.array([1.0 / n] * n),
            method = "SLSQP",
            bounds = [(0.0, max_weight)] * n,
            constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
            options={"ftol": 1e-12, "maxiter": 1000},
        )

        weights = pd.Series(np.maximum(result.x, 0.0), index=ASSETS)
        weights /= weights.sum()  # re-normalise after clipping

        out = weights.reset_index()
        out.columns = ["asset", "weight"]
        out["optimiser_status"] = result.message
        out.to_csv(self.paths.weights_file, index=False)
        logger.info(f"  Saved portfolio_weights.csv  (optimiser: {result.message})")

        return weights

    # ── 7.4 Regime-Aware Adjustment ───────────────────────────────────────────

    def apply_regime_adjustment(
        self,
        base_weights: pd.Series,
        max_weight: float = 0.70,
    ) -> Tuple[pd.Series, str, float]:
        """
        Apply regime-based tilts scaled by regime_confidence.
        Returns adjusted weights, current regime label, and confidence.
        """
        regime_df = self._regime_df.sort_values("date")
        latest    = regime_df.iloc[-1]

        current_regime = str(latest["regime_label"])
        regime_conf    = float(latest["regime_confidence"])

        tilts = REGIME_TILTS.get(current_regime, {a: 0.0 for a in ASSETS})

        adjusted = base_weights.copy()
        for asset in ASSETS:
            adjusted[asset] = base_weights[asset] + regime_conf * tilts.get(asset, 0.0)

        adjusted = adjusted.clip(lower=0.0, upper=max_weight)
        adjusted /= adjusted.sum()

        out = adjusted.reset_index()
        out.columns = ["asset", "weight"]
        out["regime"]           = current_regime
        out["regime_confidence"] = regime_conf
        out["as_of_date"]       = str(latest["date"])

        out.to_csv(self.paths.weights_regime_file, index=False)
        logger.info(f"  Saved portfolio_weights_regime_adjusted.csv")

        return adjusted, current_regime, regime_conf

    # ── 7.5 Stress Testing ────────────────────────────────────────────────────

    def run_stress_tests(
        self,
        weights: pd.Series,
        cov_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Three stress test types:
          A. Historical episode replay
          B. Regime shock (force worst regime)
          C. Macro scenario-based (Phase 4 scenarios)
        """
        records: List[dict] = []

        records.extend(self._stress_historical(weights))
        records.extend(self._stress_regime_shock(weights, cov_matrix))
        records.extend(self._stress_scenarios(weights))

        stress_df = pd.DataFrame(records)
        stress_df.to_csv(self.paths.stress_results_file, index=False)
        logger.info(f"  Saved stress_test_results.csv  ({len(stress_df)} scenarios)")

        return stress_df

    def _stress_historical(self, weights: pd.Series) -> List[dict]:
        """Replay portfolio through known historical crises."""
        df = self._regime_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        records: List[dict] = []
        for label, start_str, end_str in HISTORICAL_STRESS_PERIODS:
            start = pd.Timestamp(start_str)
            end   = pd.Timestamp(end_str)
            window = df[(df["date"] >= start) & (df["date"] <= end)].copy()

            if window.empty:
                continue

            ret_cols = [RETURN_COLS[a] for a in ASSETS]
            monthly_returns = window[ret_cols].values  # shape (T, 3)

            port_monthly = monthly_returns @ weights.values
            cumulative_return    = float(np.prod(1 + port_monthly) - 1)
            mean_monthly_return  = float(port_monthly.mean())
            annualised_vol       = float(port_monthly.std() * np.sqrt(MONTHS_PER_YEAR))
            max_drawdown         = float(self._compute_max_drawdown(port_monthly))

            records.append({
                "stress_type":        "historical",
                "scenario":           label,
                "start_date":         start_str,
                "end_date":           end_str,
                "n_months":           len(window),
                "spx_total_return":   float((window[RETURN_COLS["spx"]] + 1).prod() - 1),
                "ndx_total_return":   float((window[RETURN_COLS["ndx"]] + 1).prod() - 1),
                "gold_total_return":  float((window[RETURN_COLS["gold"]] + 1).prod() - 1),
                "portfolio_total_return":    round(cumulative_return, 6),
                "portfolio_mean_monthly_ret": round(mean_monthly_return, 6),
                "portfolio_annualised_vol":  round(annualised_vol, 6),
                "portfolio_max_drawdown":    round(max_drawdown, 6),
            })

        return records

    def _stress_regime_shock(self, weights: pd.Series, cov_matrix: pd.DataFrame) -> List[dict]:
        """
        Evaluate portfolio if market transitions to each regime.
        Uses historical regime-specific returns from regime_dataset.
        """
        df = self._regime_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        records: List[dict] = []
        for regime in ["calm", "inflation_stress", "credit_stress", "crisis"]:
            regime_rows = df[df["regime_label"] == regime].copy()
            if regime_rows.empty:
                continue

            ret_cols     = [RETURN_COLS[a] for a in ASSETS]
            monthly_rets = regime_rows[ret_cols].values
            port_monthly = monthly_rets @ weights.values

            mean_ret = float(port_monthly.mean())
            std_ret  = float(port_monthly.std())
            vol_ann  = float(std_ret * np.sqrt(MONTHS_PER_YEAR))
            max_dd   = float(self._compute_max_drawdown(port_monthly))
            var_95   = float(np.percentile(port_monthly, 5))
            cvar_95  = float(port_monthly[port_monthly <= var_95].mean()) if (port_monthly <= var_95).any() else var_95

            records.append({
                "stress_type":               "regime_shock",
                "scenario":                  f"regime_{regime}",
                "start_date":                str(regime_rows["date"].min().date()),
                "end_date":                  str(regime_rows["date"].max().date()),
                "n_months":                  len(regime_rows),
                "spx_total_return":          float(regime_rows[RETURN_COLS["spx"]].mean()),
                "ndx_total_return":          float(regime_rows[RETURN_COLS["ndx"]].mean()),
                "gold_total_return":         float(regime_rows[RETURN_COLS["gold"]].mean()),
                "portfolio_total_return":    round(mean_ret * len(regime_rows), 6),
                "portfolio_mean_monthly_ret": round(mean_ret, 6),
                "portfolio_annualised_vol":  round(vol_ann, 6),
                "portfolio_max_drawdown":    round(max_dd, 6),
                "portfolio_var_95_monthly":  round(var_95, 6),
                "portfolio_cvar_95_monthly": round(cvar_95, 6),
            })

        return records

    def _stress_scenarios(self, weights: pd.Series) -> List[dict]:
        """
        Apply Phase 4 macro scenario deltas to portfolio weights.
        Uses pre-computed return deltas: spx_return_delta, ndx_return_delta, gold_return_delta.
        """
        if not self.paths.scenarios_file.exists():
            logger.warning("  Scenario file not found — skipping scenario stress tests")
            return []

        scenarios_df = pd.read_csv(self.paths.scenarios_file)

        # Identify delta columns (btc_return_delta is optional — defaults to 0.0)
        delta_cols = {
            "spx":  "spx_return_delta",
            "ndx":  "ndx_return_delta",
            "gold": "gold_return_delta",
        }

        missing = [c for c in delta_cols.values() if c not in scenarios_df.columns]
        if missing:
            logger.warning(f"  Missing delta columns in scenario dataset: {missing}")
            return []

        has_btc_delta = "btc_return_delta" in scenarios_df.columns

        records: List[dict] = []
        scenario_names = scenarios_df["scenario_name"].unique() if "scenario_name" in scenarios_df.columns else []

        for scenario in scenario_names:
            srows = scenarios_df[scenarios_df["scenario_name"] == scenario]
            if srows.empty:
                continue

            # Average delta across all months in this scenario
            spx_delta  = float(srows[delta_cols["spx"]].mean())
            ndx_delta  = float(srows[delta_cols["ndx"]].mean())
            gold_delta = float(srows[delta_cols["gold"]].mean())
            btc_delta  = float(srows["btc_return_delta"].mean()) if has_btc_delta else 0.0

            delta_arr   = np.array([spx_delta, ndx_delta, gold_delta, btc_delta])
            port_impact = float(weights.values @ delta_arr)

            # Description from first row
            desc = srows["scenario_description"].iloc[0] if "scenario_description" in srows.columns else ""

            records.append({
                "stress_type":               "macro_scenario",
                "scenario":                  scenario,
                "start_date":                "",
                "end_date":                  "",
                "n_months":                  len(srows),
                "spx_total_return":          round(spx_delta, 6),
                "ndx_total_return":          round(ndx_delta, 6),
                "gold_total_return":         round(gold_delta, 6),
                "portfolio_total_return":    round(port_impact, 6),
                "portfolio_mean_monthly_ret": round(port_impact / max(len(srows), 1), 6),
                "portfolio_annualised_vol":  None,
                "portfolio_max_drawdown":    None,
                "scenario_description":      desc,
            })

        return records

    # ── 7.6 Portfolio Metrics ─────────────────────────────────────────────────

    def compute_portfolio_metrics(
        self,
        weights: pd.Series,
        expected_returns: Dict[str, float],
        cov_matrix: pd.DataFrame,
        risk_free_rate_monthly: float,
    ) -> dict:
        """
        Compute full suite of portfolio risk/return metrics.
        All metrics reported both monthly and annualised.
        """
        w   = weights.values
        er  = np.array([expected_returns[a] for a in ASSETS])
        cov = cov_matrix.values

        # ── Return & volatility
        expected_return_monthly = float(np.dot(w, er))
        variance_monthly        = float(w @ cov @ w)
        vol_monthly             = float(np.sqrt(variance_monthly))
        expected_return_annual  = float(expected_return_monthly * MONTHS_PER_YEAR)
        vol_annual              = float(vol_monthly * np.sqrt(MONTHS_PER_YEAR))

        # ── Sharpe ratio (monthly, then annualised)
        sharpe_monthly  = (expected_return_monthly - risk_free_rate_monthly) / vol_monthly if vol_monthly > 0 else 0.0
        sharpe_annual   = sharpe_monthly * np.sqrt(MONTHS_PER_YEAR)

        # ── Historical portfolio returns for VaR / CVaR / MaxDD
        hist_ret_cols = [RETURN_COLS[a] for a in ASSETS]
        hist_returns  = self._regime_df[hist_ret_cols].dropna().values
        port_hist     = hist_returns @ w  # shape (T,)

        var_95_monthly  = float(np.percentile(port_hist, 5))
        cvar_95_monthly = float(port_hist[port_hist <= var_95_monthly].mean()) if (port_hist <= var_95_monthly).any() else var_95_monthly
        max_drawdown    = float(self._compute_max_drawdown(port_hist))

        # ── Diversification ratio
        asset_vols = np.sqrt(np.diag(cov))
        weighted_vol_sum = float(np.dot(w, asset_vols))
        div_ratio = weighted_vol_sum / vol_monthly if vol_monthly > 0 else 1.0

        metrics = {
            "as_of_date":               str(pd.Timestamp.now().date()),
            "assets":                   ASSETS,
            "weights":                  {a: round(float(weights[a]), 6) for a in ASSETS},

            # Monthly metrics
            "expected_return_monthly":   round(expected_return_monthly, 6),
            "volatility_monthly":        round(vol_monthly, 6),
            "sharpe_ratio_monthly":      round(sharpe_monthly, 6),
            "var_95_monthly":            round(var_95_monthly, 6),
            "cvar_95_monthly":           round(cvar_95_monthly, 6),

            # Annualised metrics
            "expected_return_annual":    round(expected_return_annual, 6),
            "volatility_annual":         round(vol_annual, 6),
            "sharpe_ratio":              round(sharpe_annual, 6),
            "var_95_annual":             round(var_95_monthly * np.sqrt(MONTHS_PER_YEAR), 6),

            # Risk
            "max_drawdown":              round(max_drawdown, 6),
            "diversification_ratio":     round(div_ratio, 6),
            "risk_free_rate_monthly":    round(risk_free_rate_monthly, 6),
            "risk_free_rate_annual":     round(risk_free_rate_monthly * MONTHS_PER_YEAR, 6),

            # Per-asset annualised stats
            "asset_expected_returns_annual": {
                a: round(float(er[i] * MONTHS_PER_YEAR), 6) for i, a in enumerate(ASSETS)
            },
            "asset_volatilities_annual": {
                a: round(float(asset_vols[i] * np.sqrt(MONTHS_PER_YEAR)), 6) for i, a in enumerate(ASSETS)
            },
        }

        with open(self.paths.metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"  Saved portfolio_metrics.json")

        return metrics

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_inputs(self) -> None:
        self.timer.log("LOAD", "Loading Phase 3/5/6 outputs")

        self._features_df = pd.read_csv(self.paths.features_file, parse_dates=["date"])
        self._features_df = self._features_df.sort_values("date").reset_index(drop=True)

        # Merge btc_return from the BTC-specific features file (shorter history)
        btc_df = pd.read_csv(self.paths.features_btc_file, parse_dates=["date"])
        btc_df = btc_df[["date", "btc_return"]].sort_values("date").reset_index(drop=True)
        self._features_df = pd.merge(self._features_df, btc_df, on="date", how="left")

        self._regime_df = pd.read_csv(self.paths.regime_file, parse_dates=["date"])
        self._regime_df = self._regime_df.sort_values("date").reset_index(drop=True)
        # Merge btc_return; fill NaN with 0.0 for pre-Bitcoin periods
        self._regime_df = pd.merge(self._regime_df, btc_df, on="date", how="left")
        self._regime_df["btc_return"] = self._regime_df["btc_return"].fillna(0.0)

        with open(self.paths.phase6_metrics_file, "r", encoding="utf-8") as f:
            self._phase6_metrics = json.load(f)

        if self.paths.phase61_summary_file.exists():
            with open(self.paths.phase61_summary_file, "r", encoding="utf-8") as f:
                self._phase61_summary = json.load(f)

        self.timer.log("LOAD", (
            f"Features: {self._features_df.shape}  "
            f"Regimes: {self._regime_df.shape}  "
            f"Phase6 metrics: {list(self._phase6_metrics.keys())}"
        ))

    def _load_predictions(self, asset: str, model: str) -> pd.DataFrame:
        path = self.paths.predictions_dir / f"{model}_{asset}_predictions.csv"
        df   = pd.read_csv(path, parse_dates=["date"])
        return df

    def _get_risk_free_rate_monthly(self) -> float:
        """Extract most recent Fed Funds rate (annualised %) and convert to monthly decimal."""
        if "fed_funds_level" in self._regime_df.columns:
            ff_annual_pct = self._regime_df["fed_funds_level"].dropna().iloc[-1]
            return float(ff_annual_pct) / 100.0 / MONTHS_PER_YEAR
        return 0.0

    @staticmethod
    def _compute_max_drawdown(monthly_returns: np.ndarray) -> float:
        """Maximum drawdown from a series of monthly returns."""
        cumulative = np.cumprod(1 + monthly_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns   = (cumulative - running_max) / running_max
        return float(drawdowns.min())
