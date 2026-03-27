"""
Narrative Engine — Phase 9

Reads Phase 7 (portfolio) and Phase 8 (SHAP) outputs and generates:
  - Portfolio-level factor exposure decomposition (weighted SHAP)
  - Plain-English insights modelled on quant portfolio commentary
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


ASSETS = ["spx", "ndx", "gold", "btc"]
PORTFOLIO_ASSETS = ["spx", "ndx", "gold"]   # BTC excluded from MVO weights

ASSET_LABELS = {
    "spx": "S&P 500 (SPX)",
    "ndx": "Nasdaq 100 (NDX)",
    "gold": "Gold",
    "btc": "Bitcoin (BTC)",
}

FACTOR_LABELS = {
    "us_cpi_yoy":         "US CPI inflation",
    "vix_level":          "market volatility (VIX)",
    "us10y_yield":        "10-year Treasury yield",
    "us2y_yield":         "2-year Treasury yield",
    "yield_spread":       "yield curve (2s10s spread)",
    "high_yield_spread":  "high-yield credit spreads",
    "spx_vol_3m":         "S&P 500 realized volatility",
    "eurusd_return":      "EUR/USD exchange rate",
    "gbpusd_return":      "GBP/USD exchange rate",
    "spx_return":         "S&P 500 momentum",
    "ndx_return":         "Nasdaq momentum",
    "gold_return":        "Gold momentum",
    "ecb_level":          "ECB policy rate",
    "ecb_yoy":            "ECB rate change (YoY)",
    "regime_confidence":  "regime certainty",
    "dxy_return":         "US Dollar index",
    "qqq_return":         "Nasdaq QQQ momentum",
    "btc_return":         "Bitcoin momentum",
}

REGIME_CONTEXT = {
    "calm": {
        "label": "calm growth",
        "description": "low volatility, steady growth — risk assets typically perform well",
        "equity_bias": "positive",
        "gold_bias": "neutral",
        "fx_gold_bias": "the typical inverse gold-dollar relationship holds; dollar strength modestly pressures gold",
    },
    "inflation_stress": {
        "label": "inflation stress",
        "description": "elevated inflation with central bank tightening — growth equities under pressure, commodities supported",
        "equity_bias": "negative",
        "gold_bias": "positive",
        "fx_gold_bias": "gold rallies despite potential dollar strength as inflation-hedge demand overrides the usual FX drag",
    },
    "credit_stress": {
        "label": "credit stress",
        "description": "widening credit spreads and risk-off sentiment — quality and defensive assets preferred",
        "equity_bias": "cautious",
        "gold_bias": "positive",
        "fx_gold_bias": "safe-haven flows lift both gold and the dollar simultaneously, temporarily breaking their typical inverse link",
    },
    "crisis": {
        "label": "acute crisis",
        "description": "systemic market stress — correlations spike and diversification breaks down",
        "equity_bias": "very negative",
        "gold_bias": "mixed (initial sell-off then recovery)",
        "fx_gold_bias": "an initial dollar surge pressures gold, followed by a safe-haven recovery once acute liquidity stress eases",
    },
}


class NarrativeEngine:
    """
    Generates quant + plain-English explanations from Phase 7 & 8 outputs.
    Call .generate() to get the full explanation payload.
    """

    def __init__(self, backend_root: str | Path) -> None:
        self.root = Path(backend_root)
        self._load()

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        p = self.root / "data"
        m = self.root / "models"

        self.weights_adj  = pd.read_csv(p / "portfolio" / "portfolio_weights_regime_adjusted.csv")
        self.weights_base = pd.read_csv(p / "portfolio" / "portfolio_weights.csv")
        self.stress       = pd.read_csv(p / "portfolio" / "stress_test_results.csv")
        self.cov_matrix   = pd.read_csv(p / "portfolio" / "covariance_matrix.csv", index_col=0)
        self.er_series    = pd.read_csv(p / "portfolio" / "expected_returns.csv")

        with open(p / "portfolio" / "portfolio_metrics.json", "r", encoding="utf-8") as f:
            self.metrics = json.load(f)

        with open(p / "explainability" / "explainability_report.json", "r", encoding="utf-8") as f:
            self.shap_report = json.load(f)

        self.shap_global: Dict[str, pd.DataFrame] = {}
        for asset in ASSETS:
            path = p / "explainability" / f"shap_global_xgb_{asset}.csv"
            if path.exists():
                self.shap_global[asset] = pd.read_csv(path)

        # Gold-FX correlations (saved by portfolio_engine step 7.2b)
        gold_fx_path = p / "portfolio" / "gold_fx_correlations.json"
        if gold_fx_path.exists():
            with open(gold_fx_path, "r", encoding="utf-8") as f:
                self.gold_fx_corr: dict = json.load(f)
        else:
            self.gold_fx_corr = self._compute_gold_fx_corr_inline(p)

        # Current regime from weights file (most recent row)
        self.regime        = self.weights_adj["regime"].iloc[0]
        self.regime_conf   = float(self.weights_adj["regime_confidence"].iloc[0])
        self.as_of_date    = self.weights_adj["as_of_date"].iloc[0]

        # Weight dicts
        self.adj_weights  = dict(zip(self.weights_adj["asset"], self.weights_adj["weight"]))
        self.base_weights = dict(zip(self.weights_base["asset"], self.weights_base["weight"]))

    def _compute_gold_fx_corr_inline(self, data_path: Path, window: int = 36) -> dict:
        """Compute gold-FX correlations from the features file if the saved JSON is absent."""
        try:
            feat_path = data_path / "features" / "features_monthly_full_history.csv"
            if not feat_path.exists():
                return {}
            cols = ["gold_return", "eurusd_return", "gbpusd_return", "dxy_return"]
            df = pd.read_csv(feat_path, usecols=lambda c: c in cols)
            df = df.dropna().tail(window)
            if len(df) < 12 or "gold_return" not in df.columns:
                return {}
            corr = df.corr()
            return {
                "window_months":  window,
                "n_observations": len(df),
                "gold_eurusd": round(float(corr.loc["gold_return", "eurusd_return"]), 4) if "eurusd_return" in corr.columns else None,
                "gold_gbpusd": round(float(corr.loc["gold_return", "gbpusd_return"]), 4) if "gbpusd_return" in corr.columns else None,
                "gold_dxy":    round(float(corr.loc["gold_return", "dxy_return"]),    4) if "dxy_return"    in corr.columns else None,
            }
        except Exception:
            return {}

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self) -> dict:
        """Return full explanation payload using pipeline-computed weights."""
        factor_exposures = self._portfolio_factor_exposures()
        return {
            "as_of_date":        self.as_of_date,
            "regime":            self.regime,
            "regime_confidence": self.regime_conf,
            "quant": {
                "factor_exposures":      factor_exposures,
                "per_asset_top_factors": self._per_asset_top_factors(),
                "stress_factor_map":     self._stress_factor_map(),
            },
            "narratives": self._plain_english_narratives(factor_exposures),
        }

    def generate_for_weights(self, custom_weights: Dict[str, float],
                              scenario_result: dict | None = None) -> dict:
        """
        Generate a fully dynamic narrative for any user-supplied weights.

        custom_weights  – e.g. {"spx": 0.5, "ndx": 0.3, "gold": 0.2}
        scenario_result – optional output from /api/scenario/run, used to enrich
                          the stress narrative paragraph.
        """
        # Temporarily override weight state
        orig_adj   = self.adj_weights
        orig_base  = self.base_weights

        # Normalise to sum = 1
        total = sum(custom_weights.values()) or 1.0
        self.adj_weights  = {k: v / total for k, v in custom_weights.items()}
        self.base_weights = self.adj_weights.copy()   # no regime-tilt delta shown

        try:
            factor_exposures = self._portfolio_factor_exposures()
            narratives       = self._plain_english_narratives(factor_exposures)

            # Append a scenario-specific paragraph if the caller supplied one
            if scenario_result:
                narratives.append(self._narrative_scenario_impact(scenario_result))

            return {
                "as_of_date":        self.as_of_date,
                "regime":            self.regime,
                "regime_confidence": self.regime_conf,
                "weights_used":      self.adj_weights,
                "quant": {
                    "factor_exposures":      factor_exposures,
                    "per_asset_top_factors": self._per_asset_top_factors(),
                },
                "narratives": narratives,
            }
        finally:
            # Always restore original state
            self.adj_weights  = orig_adj
            self.base_weights = orig_base

    def _narrative_scenario_impact(self, scenario_result: dict) -> dict:
        """Generate a paragraph interpreting the result of a specific scenario run."""
        sc_id    = scenario_result.get("scenario", "unknown")
        pf_ret   = scenario_result.get("portfolio_return", 0.0)
        contribs = scenario_result.get("contributions", {})
        sc_desc  = scenario_result.get("description") or sc_id.replace("_", " ").title()

        direction = "loss" if pf_ret < 0 else "gain"
        severity  = (
            "severe"     if pf_ret < -0.20 else
            "significant" if pf_ret < -0.10 else
            "moderate"   if pf_ret < -0.05 else
            "mild"
        )

        # Find worst-contributing asset
        if contribs:
            worst_asset = min(contribs, key=lambda a: contribs.get(a, 0))
            worst_val   = contribs.get(worst_asset, 0)
        else:
            worst_asset, worst_val = None, 0

        text = (
            f"Under the '{sc_desc}' scenario, your portfolio would experience "
            f"a {severity} {direction} of {abs(pf_ret):.1%}. "
        )
        if worst_asset and worst_val < -0.01:
            text += (
                f"{ASSET_LABELS.get(worst_asset, worst_asset.upper())} is the largest detractor, "
                f"contributing {worst_val:.2%} to the total portfolio return. "
            )

        regime_rc = REGIME_CONTEXT.get(self.regime, {})
        if regime_rc:
            text += (
                f"Given the current {regime_rc.get('label', self.regime)} regime, "
                f"{regime_rc.get('description', 'macro conditions are shifting')}."
            )

        return {"type": "scenario_impact", "title": f"Scenario Impact: {sc_desc}", "text": text}

    # ── Quant: factor exposure decomposition ──────────────────────────────────

    def _portfolio_factor_exposures(self) -> List[dict]:
        """
        Portfolio-level factor importance = Σ_i (weight_i × mean_abs_shap_i_f)
        across portfolio assets (SPX, NDX, Gold).  Sorted descending.
        """
        factor_totals: Dict[str, float] = {}

        for asset in PORTFOLIO_ASSETS:
            weight = self.adj_weights.get(asset, 0.0)
            if weight == 0 or asset not in self.shap_global:
                continue
            df = self.shap_global[asset]
            for _, row in df.iterrows():
                feature = row["feature"]
                shap    = float(row["mean_abs_shap"])
                factor_totals[feature] = factor_totals.get(feature, 0.0) + weight * shap

        rows = [
            {
                "feature":       f,
                "label":         FACTOR_LABELS.get(f, f),
                "portfolio_shap": round(v, 6),
                "rank":          i + 1,
            }
            for i, (f, v) in enumerate(
                sorted(factor_totals.items(), key=lambda x: -x[1])
            )
        ]
        return rows

    def _per_asset_top_factors(self, top_n: int = 5) -> Dict[str, List[dict]]:
        result = {}
        for asset in ASSETS:
            if asset not in self.shap_global:
                continue
            df = self.shap_global[asset].head(top_n)
            result[asset] = [
                {
                    "rank":          i + 1,
                    "feature":       row["feature"],
                    "label":         FACTOR_LABELS.get(row["feature"], row["feature"]),
                    "mean_abs_shap": round(float(row["mean_abs_shap"]), 6),
                }
                for i, (_, row) in enumerate(df.iterrows())
            ]
        return result

    def _stress_factor_map(self) -> List[dict]:
        """Historical stress scenarios sorted by portfolio impact."""
        hist = self.stress[self.stress["stress_type"] == "historical"].copy()
        hist = hist.sort_values("portfolio_total_return")
        return [
            {
                "scenario":            row["scenario"],
                "portfolio_return_pct": round(float(row["portfolio_total_return"]) * 100, 2),
                "spx_return_pct":       round(float(row["spx_total_return"]) * 100, 2) if pd.notna(row.get("spx_total_return")) else None,
                "ndx_return_pct":       round(float(row["ndx_total_return"]) * 100, 2) if pd.notna(row.get("ndx_total_return")) else None,
                "gold_return_pct":      round(float(row["gold_total_return"]) * 100, 2) if pd.notna(row.get("gold_total_return")) else None,
            }
            for _, row in hist.iterrows()
        ]

    # ── Plain-English narratives ───────────────────────────────────────────────

    def _plain_english_narratives(self, factor_exposures: List[dict]) -> List[dict]:
        narratives = []
        narratives.append(self._narrative_composition())
        narratives.append(self._narrative_dominant_factor(factor_exposures))
        narratives.append(self._narrative_regime_context())
        narratives.append(self._narrative_stress_vulnerability())
        narratives.append(self._narrative_diversification())
        narratives.append(self._narrative_hedge_effectiveness())
        return [n for n in narratives if n is not None]

    def _narrative_composition(self) -> dict:
        active = {a: w for a, w in self.adj_weights.items() if w > 0.01}
        sorted_assets = sorted(active.items(), key=lambda x: -x[1])

        parts = [f"{ASSET_LABELS[a]} ({w:.0%})" for a, w in sorted_assets]
        alloc_str = " and ".join(parts) if len(parts) <= 2 else ", ".join(parts[:-1]) + f", and {parts[-1]}"

        rc = REGIME_CONTEXT.get(self.regime, {})
        regime_label = rc.get("label", self.regime.replace("_", " "))

        text = (
            f"Your portfolio is allocated to {alloc_str}. "
            f"This positioning reflects the current {regime_label} regime "
            f"(confidence: {self.regime_conf:.0%}), "
            f"characterised by {rc.get('description', 'shifting macro conditions')}."
        )

        # Regime tilt commentary
        base_eq  = sum(self.base_weights.get(a, 0) for a in ["spx", "ndx"])
        adj_eq   = sum(self.adj_weights.get(a, 0) for a in ["spx", "ndx"])
        delta_eq = adj_eq - base_eq
        if abs(delta_eq) > 0.03:
            direction = "reduced" if delta_eq < 0 else "increased"
            text += (
                f" Compared to the unconstrained optimal, equity exposure has been "
                f"{direction} by {abs(delta_eq):.0%} to reflect regime risk."
            )

        return {"type": "portfolio_composition", "title": "Portfolio Composition", "text": text}

    def _narrative_dominant_factor(self, factor_exposures: List[dict]) -> dict:
        if not factor_exposures:
            return None

        top = factor_exposures[0]
        second = factor_exposures[1] if len(factor_exposures) > 1 else None

        feature = top["feature"]
        label   = top["label"]
        shap    = top["portfolio_shap"]

        # Per-asset sensitivity to this factor
        asset_sensitivity = []
        for asset in PORTFOLIO_ASSETS:
            w = self.adj_weights.get(asset, 0)
            if w < 0.01 or asset not in self.shap_global:
                continue
            df = self.shap_global[asset]
            row = df[df["feature"] == feature]
            if not row.empty:
                asset_sensitivity.append((asset, w, float(row["mean_abs_shap"].iloc[0])))

        text = (
            f"The dominant macro driver across your portfolio is {label} "
            f"(weighted SHAP importance: {shap:.4f}). "
        )

        if asset_sensitivity:
            sens_parts = [
                f"{ASSET_LABELS[a]} ({w:.0%} weight, SHAP={s:.4f})"
                for a, w, s in sorted(asset_sensitivity, key=lambda x: -x[2])
            ]
            text += f"It significantly influences {', '.join(sens_parts)}. "

        # Contextual interpretation for known factors
        if feature == "us_cpi_yoy":
            text += (
                "Rising inflation historically pressures growth equities by compressing valuation multiples, "
                "while providing partial support for Gold as an inflation hedge."
            )
        elif feature == "us10y_yield":
            text += (
                "Rising long-term yields increase the discount rate on future earnings, "
                "weighing on growth equities (especially Nasdaq) while supporting the US Dollar."
            )
        elif feature == "vix_level":
            text += (
                "Elevated market volatility triggers risk-off rotation, reducing equity returns "
                "and typically supporting defensive assets like Gold."
            )
        elif feature == "high_yield_spread":
            text += (
                "Widening credit spreads signal deteriorating credit conditions, "
                "which historically leads to equity drawdowns and safe-haven flows into Gold."
            )
        elif feature == "dxy_return":
            text += (
                "The US Dollar Index is one of gold's strongest inverse drivers. "
                "A rising DXY makes gold more expensive in local currency terms globally, suppressing demand, "
                "while a falling DXY acts as a direct tailwind for gold prices."
            )
        elif feature == "eurusd_return":
            text += (
                "A rising EUR/USD (weaker US Dollar) historically supports gold, "
                "as dollar-denominated commodities become cheaper for foreign buyers, boosting global demand. "
                "The inverse also holds: dollar strength from a falling EUR/USD typically weighs on gold."
            )
        elif feature == "gbpusd_return":
            text += (
                "GBP/USD movements signal broader USD directional trends. "
                "A strengthening dollar (falling GBP/USD) tends to pressure gold, "
                "since gold is priced in dollars and becomes relatively more expensive globally."
            )

        if second:
            text += (
                f" The second-largest factor is {second['label']} "
                f"(portfolio SHAP: {second['portfolio_shap']:.4f})."
            )

        return {"type": "dominant_risk_factor", "title": "Dominant Risk Factor", "text": text}

    def _narrative_regime_context(self) -> dict:
        rc = REGIME_CONTEXT.get(self.regime, {})
        regime_label = rc.get("label", self.regime.replace("_", " "))

        # Regime shock return from stress tests
        regime_stress = self.stress[
            self.stress["scenario"] == f"regime_{self.regime}"
        ]
        regime_return = None
        if not regime_stress.empty:
            regime_return = float(regime_stress["portfolio_total_return"].iloc[0])

        # Worst historical scenario
        hist = self.stress[self.stress["stress_type"] == "historical"].copy()
        worst = hist.sort_values("portfolio_total_return").iloc[0]
        worst_ret  = float(worst["portfolio_total_return"])
        worst_name = worst["scenario"].replace("_", " ")

        text = (
            f"The model detects a {regime_label} regime with {self.regime_conf:.0%} confidence as of "
            f"{str(self.as_of_date)[:10]}. "
            f"This regime is characterised by {rc.get('description', 'elevated macro uncertainty')}. "
        )

        if regime_return is not None:
            text += (
                f"Historical {regime_label} periods produced an annualised portfolio return of "
                f"{regime_return * 100:+.1f}% for the current allocation. "
            )

        text += (
            f"The most damaging historical scenario for this allocation is the "
            f"'{worst_name}' episode ({worst_ret * 100:+.1f}% cumulative). "
        )

        equity_bias = rc.get("equity_bias", "neutral")
        gold_bias   = rc.get("gold_bias", "neutral")
        text += (
            f"In this regime type, equities typically show {equity_bias} momentum "
            f"and Gold tends to be {gold_bias}."
        )

        return {"type": "regime_context", "title": "Regime Context", "text": text}

    def _narrative_stress_vulnerability(self) -> dict:
        hist = self.stress[self.stress["stress_type"] == "historical"].copy()
        hist = hist.sort_values("portfolio_total_return")

        worst     = hist.iloc[0]
        second    = hist.iloc[1] if len(hist) > 1 else None

        worst_ret  = float(worst["portfolio_total_return"])
        worst_name = worst["scenario"].replace("_", " ")

        # Macro scenarios (worst)
        macro = self.stress[self.stress["stress_type"] == "macro_scenario"].copy()
        macro_worst = macro.sort_values("portfolio_total_return").iloc[0] if not macro.empty else None

        text = (
            f"Your portfolio's greatest historical stress was the '{worst_name}' episode, "
            f"producing a {worst_ret * 100:+.1f}% cumulative loss. "
        )

        # Decompose: which asset contributed most to that loss?
        spx_ret  = float(worst.get("spx_total_return", 0) or 0)
        ndx_ret  = float(worst.get("ndx_total_return", 0) or 0)
        gold_ret = float(worst.get("gold_total_return", 0) or 0)

        w_spx  = self.adj_weights.get("spx", 0)
        w_ndx  = self.adj_weights.get("ndx", 0)
        w_gold = self.adj_weights.get("gold", 0)

        contributions = {
            "spx":  w_spx  * spx_ret,
            "ndx":  w_ndx  * ndx_ret,
            "gold": w_gold * gold_ret,
        }
        biggest_loss    = min(contributions, key=contributions.get)
        biggest_contrib = contributions[biggest_loss]
        biggest_offset  = max(contributions, key=contributions.get)
        offset_contrib  = contributions[biggest_offset]

        if biggest_contrib < -0.005:
            text += (
                f"{ASSET_LABELS[biggest_loss]} (weight: {self.adj_weights[biggest_loss]:.0%}) "
                f"contributed {biggest_contrib * 100:+.1f}% to the loss. "
            )
        if offset_contrib > 0.005:
            text += (
                f"{ASSET_LABELS[biggest_offset]} (weight: {self.adj_weights[biggest_offset]:.0%}) "
                f"partially offset losses, contributing {offset_contrib * 100:+.1f}%. "
            )

        if second is not None:
            second_ret  = float(second["portfolio_total_return"])
            second_name = second["scenario"].replace("_", " ")
            text += (
                f"The second-worst scenario is '{second_name}' ({second_ret * 100:+.1f}%). "
            )

        if macro_worst is not None:
            mw_ret  = float(macro_worst["portfolio_total_return"])
            mw_name = macro_worst["scenario"].replace("_", " ")
            text += (
                f"Among forward-looking macro shocks, a '{mw_name}' scenario "
                f"would cost {mw_ret * 100:+.1f}%."
            )

        return {"type": "stress_vulnerability", "title": "Stress Vulnerability", "text": text}

    def _narrative_diversification(self) -> dict:
        div_ratio = float(self.metrics.get("diversification_ratio", 1.0))
        max_dd    = float(self.metrics.get("max_drawdown", 0.0))
        var_95    = float(self.metrics.get("var_95_monthly", 0.0))
        sharpe    = float(self.metrics.get("sharpe_ratio", 0.0))

        # SPX-NDX correlation
        try:
            spx_ndx_cov  = float(self.cov_matrix.loc["spx", "ndx"])
            spx_vol      = float(self.cov_matrix.loc["spx", "spx"]) ** 0.5
            ndx_vol      = float(self.cov_matrix.loc["ndx", "ndx"]) ** 0.5
            spx_ndx_corr = spx_ndx_cov / (spx_vol * ndx_vol) if spx_vol * ndx_vol > 0 else 0
        except (KeyError, ZeroDivisionError):
            spx_ndx_corr = None

        try:
            ndx_gold_cov  = float(self.cov_matrix.loc["ndx", "gold"])
            gold_vol      = float(self.cov_matrix.loc["gold", "gold"]) ** 0.5
            ndx_gold_corr = ndx_gold_cov / (ndx_vol * gold_vol) if ndx_vol * gold_vol > 0 else 0
        except (KeyError, ZeroDivisionError):
            ndx_gold_corr = None

        text = (
            f"The portfolio achieves a diversification ratio of {div_ratio:.2f} "
            f"(>1 indicates genuine diversification benefit). "
        )

        if ndx_gold_corr is not None:
            corr_label = "near-zero" if abs(ndx_gold_corr) < 0.15 else ("positive" if ndx_gold_corr > 0 else "negative")
            text += (
                f"Gold and Nasdaq carry a {corr_label} 36-month rolling correlation ({ndx_gold_corr:.2f}), "
                f"providing genuine risk offset between the two largest positions. "
            )

        if spx_ndx_corr is not None and spx_ndx_corr > 0.7:
            text += (
                f"However, S&P 500 and Nasdaq are highly correlated ({spx_ndx_corr:.2f}). "
                f"Holding both increases equity concentration without proportional diversification benefit — "
                f"in a broad equity sell-off, they tend to fall together. "
            )

        # Gold-FX correlation narrative
        if self.gold_fx_corr:
            gold_dxy    = self.gold_fx_corr.get("gold_dxy")
            gold_eurusd = self.gold_fx_corr.get("gold_eurusd")
            fx_window   = self.gold_fx_corr.get("window_months", 36)
            rc          = REGIME_CONTEXT.get(self.regime, {})
            fx_gold_bias = rc.get("fx_gold_bias", "")

            if gold_dxy is not None:
                if gold_dxy < -0.4:
                    dxy_label = "strongly negative"
                elif gold_dxy < -0.15:
                    dxy_label = "negative"
                elif abs(gold_dxy) < 0.15:
                    dxy_label = "near-zero"
                else:
                    dxy_label = "positive"
                typical = "typical" if gold_dxy < -0.15 else ("weakened" if abs(gold_dxy) < 0.15 else "atypical")
                text += (
                    f"Gold's {fx_window}-month rolling correlation with the US Dollar Index (DXY) is "
                    f"{dxy_label} ({gold_dxy:+.2f}), reflecting the {typical} inverse gold-dollar relationship. "
                )
                if fx_gold_bias:
                    text += f"In the current {rc.get('label', self.regime)} regime, {fx_gold_bias}. "

            if gold_eurusd is not None:
                if gold_eurusd > 0.15:
                    eurusd_label = "positive"
                elif abs(gold_eurusd) < 0.15:
                    eurusd_label = "near-zero"
                else:
                    eurusd_label = "negative"
                text += (
                    f"The Gold-EUR/USD correlation is {eurusd_label} ({gold_eurusd:+.2f}): "
                    f"since EUR/USD and DXY move inversely, gold and EUR/USD tend to track together "
                    f"through the dollar channel. "
                )

        text += (
            f"The monthly VaR at 95% confidence is {var_95 * 100:+.1f}%, and the historical max drawdown "
            f"for this allocation is {max_dd * 100:+.1f}%. "
            f"The Sharpe ratio stands at {sharpe:.2f} annualised."
        )

        return {"type": "diversification", "title": "Diversification Analysis", "text": text}

    def _narrative_hedge_effectiveness(self) -> dict:
        w_gold = self.adj_weights.get("gold", 0)
        w_ndx  = self.adj_weights.get("ndx", 0)
        w_spx  = self.adj_weights.get("spx", 0)
        w_eq   = w_ndx + w_spx

        # Gold vs inflation scenario
        inflation_scenario = self.stress[
            self.stress["scenario"].str.contains("inflation", case=False, na=False)
        ]

        text = ""

        if w_gold > 0.40:
            text += (
                f"Gold carries a significant {w_gold:.0%} portfolio weight, acting as the primary hedge. "
            )
            if not inflation_scenario.empty:
                inf_row  = inflation_scenario.iloc[0]
                port_ret = float(inf_row["portfolio_total_return"])
                gold_ret = float(inf_row.get("gold_total_return", 0) or 0)
                text += (
                    f"In the historical inflation spike scenario, Gold returned {gold_ret * 100:+.1f}% "
                    f"while the total portfolio delivered {port_ret * 100:+.1f}%. "
                )
        elif w_gold > 0.15:
            text += (
                f"Gold provides a modest hedge at {w_gold:.0%} of the portfolio. "
                f"This partially offsets equity drawdowns but may be insufficient in a severe sell-off. "
            )
        else:
            text += (
                f"Gold allocation is minimal ({w_gold:.0%}). "
                f"The portfolio carries substantial unhedged equity risk. "
            )

        # Equity concentration commentary
        if w_eq > 0.50:
            text += (
                f"With {w_eq:.0%} in equities, the portfolio is highly exposed to growth equities and "
                f"will lose more under rate-hike or risk-off scenarios. "
                f"Consider whether the current regime justifies this concentration."
            )
        elif w_eq > 0.20:
            text += (
                f"The {w_eq:.0%} equity allocation provides growth exposure while the gold position "
                f"acts as a partial counterweight in stress periods."
            )
        else:
            text += (
                f"Equity exposure is low ({w_eq:.0%}), which reduces upside participation "
                f"but limits drawdown risk in the current {self.regime.replace('_', ' ')} environment."
            )

        return {"type": "hedge_effectiveness", "title": "Hedge Effectiveness", "text": text}
