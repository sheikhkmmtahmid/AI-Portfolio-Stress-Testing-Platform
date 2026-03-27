from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScenarioDefinition:
    name: str
    description: str
    additive_shocks: Dict[str, float]
    multiplicative_shocks: Dict[str, float]
    rules: Dict[str, Any]


class ScenarioEngine:
    """
    Phase 4 deterministic scenario engine.

    Input:
        data/features/features_monthly_full_history.csv

    Outputs:
        data/scenarios/scenario_dataset.csv
        data/scenarios/scenario_summary.csv
    """

    def __init__(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        scenario_definitions: Optional[List[ScenarioDefinition]] = None,
    ) -> None:
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.scenario_definitions = (
            scenario_definitions if scenario_definitions is not None else self._default_scenarios()
        )

        self.df: Optional[pd.DataFrame] = None

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Starting Phase 4 scenario engine")
        self.df = self._load_input_data()

        scenario_frames: List[pd.DataFrame] = []

        for scenario in self.scenario_definitions:
            logger.info("Applying scenario: %s", scenario.name)
            shocked = self._apply_scenario(self.df.copy(deep=True), scenario)
            scenario_frames.append(shocked)

        scenario_dataset = pd.concat(scenario_frames, axis=0, ignore_index=True)
        scenario_dataset = self._finalise_output_columns(scenario_dataset)

        scenario_summary = self._build_summary(scenario_dataset)

        scenario_dataset_path = self.output_dir / "scenario_dataset.csv"
        scenario_summary_path = self.output_dir / "scenario_summary.csv"

        scenario_dataset.to_csv(scenario_dataset_path, index=False)
        scenario_summary.to_csv(scenario_summary_path, index=False)

        logger.info("Scenario dataset saved to: %s", scenario_dataset_path)
        logger.info("Scenario summary saved to: %s", scenario_summary_path)

        return scenario_dataset, scenario_summary

    def _load_input_data(self) -> pd.DataFrame:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input dataset not found: {self.input_path}")

        df = pd.read_csv(self.input_path)

        if df.empty:
            raise ValueError("Input dataset is empty")

        if "date" not in df.columns:
            raise ValueError("Input dataset must contain a 'date' column")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().any():
            raise ValueError("Some rows contain invalid dates")

        df = df.sort_values("date").reset_index(drop=True)

        logger.info("Loaded input data with shape: %s", df.shape)
        logger.info("Date range: %s to %s", df["date"].min(), df["date"].max())
        logger.info("Columns: %s", list(df.columns))

        return df

    def _apply_scenario(self, df: pd.DataFrame, scenario: ScenarioDefinition) -> pd.DataFrame:
        df["scenario_name"] = scenario.name
        df["scenario_description"] = scenario.description

        original_df = df.copy(deep=True)

        self._apply_additive_shocks(df, scenario.additive_shocks)
        self._apply_multiplicative_shocks(df, scenario.multiplicative_shocks)
        self._apply_correlated_rules(df, original_df, scenario.rules)
        self._recompute_dependent_fields(df)
        self._clip_ranges(df)
        self._add_delta_columns(df, original_df)

        return df

    def _apply_additive_shocks(self, df: pd.DataFrame, shocks: Dict[str, float]) -> None:
        for col, shock in shocks.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") + shock

    def _apply_multiplicative_shocks(self, df: pd.DataFrame, shocks: Dict[str, float]) -> None:
        for col, shock in shocks.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") * (1.0 + shock)

    def _apply_correlated_rules(
        self,
        df: pd.DataFrame,
        original_df: pd.DataFrame,
        rules: Dict[str, Any],
    ) -> None:
        if rules.get("propagate_rates_to_risk", False):
            self._propagate_rates_to_risk(df, original_df)

        if rules.get("propagate_inflation_to_rates", False):
            self._propagate_inflation_to_rates(df, original_df)

        if rules.get("propagate_vix_to_volatility", False):
            self._propagate_vix_to_volatility(df, original_df)

        if rules.get("propagate_credit_to_risk", False):
            self._propagate_credit_to_risk(df, original_df)

        if rules.get("propagate_usd_squeeze", False):
            self._propagate_usd_squeeze(df)

        if rules.get("propagate_systemic_crisis", False):
            self._propagate_systemic_crisis(df)

        if rules.get("propagate_growth_scare", False):
            self._propagate_growth_scare(df)

    def _propagate_rates_to_risk(self, df: pd.DataFrame, original_df: pd.DataFrame) -> None:
        delta_2y = (
            (pd.to_numeric(df["us2y_yield"], errors="coerce") - pd.to_numeric(original_df["us2y_yield"], errors="coerce"))
            if "us2y_yield" in df.columns else 0.0
        )
        delta_10y = (
            (pd.to_numeric(df["us10y_yield"], errors="coerce") - pd.to_numeric(original_df["us10y_yield"], errors="coerce"))
            if "us10y_yield" in df.columns else 0.0
        )

        combined_rate_move = delta_2y + delta_10y
        scaled = combined_rate_move / 0.02  # 2 x 100bps total reference move

        for col in ["spx_vol_3m", "spx_vol_6m", "ndx_vol_3m", "ndx_vol_6m"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") * (1.0 + 0.08 * scaled)

        for col in ["spx_return", "ndx_return"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") - (0.010 * scaled)

        if "gold_return" in df.columns:
            df["gold_return"] = pd.to_numeric(df["gold_return"], errors="coerce") - (0.003 * scaled)

        for col in ["eurusd_return", "gbpusd_return"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") - (0.005 * scaled)

    def _propagate_inflation_to_rates(self, df: pd.DataFrame, original_df: pd.DataFrame) -> None:
        if "us_cpi_yoy" not in df.columns:
            return

        inflation_delta = (
            pd.to_numeric(df["us_cpi_yoy"], errors="coerce")
            - pd.to_numeric(original_df["us_cpi_yoy"], errors="coerce")
        )

        scaled = inflation_delta / 0.01  # +1% inflation reference

        if "us2y_yield" in df.columns:
            df["us2y_yield"] = pd.to_numeric(df["us2y_yield"], errors="coerce") + (0.0025 * scaled)

        if "us10y_yield" in df.columns:
            df["us10y_yield"] = pd.to_numeric(df["us10y_yield"], errors="coerce") + (0.0015 * scaled)

        for col in ["spx_return", "ndx_return"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") - (0.006 * scaled)

        if "gold_return" in df.columns:
            df["gold_return"] = pd.to_numeric(df["gold_return"], errors="coerce") + (0.005 * scaled)

    def _propagate_vix_to_volatility(self, df: pd.DataFrame, original_df: pd.DataFrame) -> None:
        if "vix_level" not in df.columns:
            return

        vix_delta = (
            pd.to_numeric(df["vix_level"], errors="coerce")
            - pd.to_numeric(original_df["vix_level"], errors="coerce")
        )
        scaled = vix_delta / 10.0  # +10 VIX points reference

        for col in ["spx_vol_3m", "spx_vol_6m", "ndx_vol_3m", "ndx_vol_6m"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") * (1.0 + 0.20 * scaled)

        for col in ["spx_return", "ndx_return"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") - (0.015 * scaled)

        if "gold_return" in df.columns:
            df["gold_return"] = pd.to_numeric(df["gold_return"], errors="coerce") + (0.002 * scaled)

        for col in ["spx_drawdown", "ndx_drawdown", "gold_drawdown", "spx_max_dd_6m", "ndx_max_dd_6m", "gold_max_dd_6m"]:
            if col in df.columns:
                if "gold" in col:
                    df[col] = pd.to_numeric(df[col], errors="coerce") - (0.005 * scaled)
                else:
                    df[col] = pd.to_numeric(df[col], errors="coerce") - (0.030 * scaled)

        if "vix_spike" in df.columns:
            df["vix_spike"] = 1

    def _propagate_credit_to_risk(self, df: pd.DataFrame, original_df: pd.DataFrame) -> None:
        if "high_yield_spread" not in df.columns:
            return

        hy_delta = (
            pd.to_numeric(df["high_yield_spread"], errors="coerce")
            - pd.to_numeric(original_df["high_yield_spread"], errors="coerce")
        )
        scaled = hy_delta / 0.02  # +200bps reference

        for col in ["spx_return", "ndx_return"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") - (0.012 * scaled)

        if "gold_return" in df.columns:
            df["gold_return"] = pd.to_numeric(df["gold_return"], errors="coerce") + (0.003 * scaled)

        for col in ["spx_vol_3m", "spx_vol_6m", "ndx_vol_3m", "ndx_vol_6m"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") * (1.0 + 0.12 * scaled)

        for col in ["spx_drawdown", "ndx_drawdown"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") - (0.020 * scaled)

    def _propagate_usd_squeeze(self, df: pd.DataFrame) -> None:
        for col in ["eurusd_return", "gbpusd_return"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") - 0.020

        for col in ["spx_return", "ndx_return"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") - 0.010

        if "gold_return" in df.columns:
            df["gold_return"] = pd.to_numeric(df["gold_return"], errors="coerce") - 0.005

        if "vix_level" in df.columns:
            df["vix_level"] = pd.to_numeric(df["vix_level"], errors="coerce") + 5.0

    def _propagate_systemic_crisis(self, df: pd.DataFrame) -> None:
        for col in ["spx_return", "ndx_return"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") - 0.030

        if "gold_return" in df.columns:
            df["gold_return"] = pd.to_numeric(df["gold_return"], errors="coerce") + 0.010

        for col in ["spx_vol_3m", "spx_vol_6m", "ndx_vol_3m", "ndx_vol_6m"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") * 1.35

        if "vix_level" in df.columns:
            df["vix_level"] = pd.to_numeric(df["vix_level"], errors="coerce") + 15.0

        if "high_yield_spread" in df.columns:
            df["high_yield_spread"] = pd.to_numeric(df["high_yield_spread"], errors="coerce") + 0.03

        for col in ["spx_drawdown", "ndx_drawdown", "spx_max_dd_6m", "ndx_max_dd_6m"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") - 0.050

        if "gold_drawdown" in df.columns:
            df["gold_drawdown"] = pd.to_numeric(df["gold_drawdown"], errors="coerce") - 0.010

        if "gold_max_dd_6m" in df.columns:
            df["gold_max_dd_6m"] = pd.to_numeric(df["gold_max_dd_6m"], errors="coerce") - 0.010

        if "vix_spike" in df.columns:
            df["vix_spike"] = 1

    def _propagate_growth_scare(self, df: pd.DataFrame) -> None:
        for col in ["spx_return", "ndx_return"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") - 0.012

        if "gold_return" in df.columns:
            df["gold_return"] = pd.to_numeric(df["gold_return"], errors="coerce") + 0.004

        for col in ["us2y_yield", "us10y_yield"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") - 0.0025

    def _recompute_dependent_fields(self, df: pd.DataFrame) -> None:
        if "us10y_yield" in df.columns and "us2y_yield" in df.columns:
            df["yield_spread"] = (
                pd.to_numeric(df["us10y_yield"], errors="coerce")
                - pd.to_numeric(df["us2y_yield"], errors="coerce")
            )

        if "vix_level" in df.columns and "vix_spike" in df.columns:
            df["vix_spike"] = np.where(
                pd.to_numeric(df["vix_level"], errors="coerce") >= 30.0,
                1,
                pd.to_numeric(df["vix_spike"], errors="coerce").fillna(0).astype(int),
            )

    def _clip_ranges(self, df: pd.DataFrame) -> None:
        volatility_cols = ["gold_vol_3m", "gold_vol_6m", "spx_vol_3m", "spx_vol_6m", "ndx_vol_3m", "ndx_vol_6m"]
        for col in volatility_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=0.0)

        drawdown_cols = ["gold_drawdown", "gold_max_dd_6m", "spx_drawdown", "spx_max_dd_6m", "ndx_drawdown", "ndx_max_dd_6m"]
        for col in drawdown_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").clip(upper=0.0)

    def _add_delta_columns(self, df: pd.DataFrame, original_df: pd.DataFrame) -> None:
        tracked_cols = [
            "us2y_yield",
            "us10y_yield",
            "yield_spread",
            "us_cpi_yoy",
            "vix_level",
            "high_yield_spread",
            "spx_return",
            "ndx_return",
            "gold_return",
            "eurusd_return",
            "gbpusd_return",
        ]

        for col in tracked_cols:
            if col in df.columns and col in original_df.columns:
                df[f"{col}_delta"] = (
                    pd.to_numeric(df[col], errors="coerce")
                    - pd.to_numeric(original_df[col], errors="coerce")
                )

    def _finalise_output_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        front = ["date", "scenario_name", "scenario_description"]
        front = [c for c in front if c in df.columns]
        rest = [c for c in df.columns if c not in front]
        return df[front + rest]

    def _build_summary(self, scenario_dataset: pd.DataFrame) -> pd.DataFrame:
        summary_cols = [
            "us2y_yield",
            "us10y_yield",
            "yield_spread",
            "us_cpi_yoy",
            "vix_level",
            "high_yield_spread",
            "spx_return",
            "ndx_return",
            "gold_return",
            "eurusd_return",
            "gbpusd_return",
        ]
        summary_cols = [c for c in summary_cols if c in scenario_dataset.columns]

        agg_map = {col: "mean" for col in summary_cols}

        for col in summary_cols:
            delta_col = f"{col}_delta"
            if delta_col in scenario_dataset.columns:
                agg_map[delta_col] = "mean"

        summary = (
            scenario_dataset
            .groupby(["scenario_name", "scenario_description"], dropna=False)
            .agg(agg_map)
            .reset_index()
            .sort_values("scenario_name")
            .reset_index(drop=True)
        )

        return summary

    @staticmethod
    def _default_scenarios() -> List[ScenarioDefinition]:
        return [
            ScenarioDefinition(
                name="baseline",
                description="No shock applied. Control scenario.",
                additive_shocks={},
                multiplicative_shocks={},
                rules={},
            ),
            ScenarioDefinition(
                name="rates_up_100bps",
                description="Parallel +100 bps shock to US 2Y and US 10Y yields.",
                additive_shocks={
                    "us2y_yield": 0.01,
                    "us10y_yield": 0.01,
                },
                multiplicative_shocks={},
                rules={
                    "propagate_rates_to_risk": True,
                },
            ),
            ScenarioDefinition(
                name="inflation_up_100bps",
                description="US inflation shock of +1 percentage point YoY.",
                additive_shocks={
                    "us_cpi_yoy": 0.01,
                },
                multiplicative_shocks={},
                rules={
                    "propagate_inflation_to_rates": True,
                },
            ),
            ScenarioDefinition(
                name="vix_spike_10pt",
                description="Volatility shock with VIX rising by 10 points.",
                additive_shocks={
                    "vix_level": 10.0,
                },
                multiplicative_shocks={},
                rules={
                    "propagate_vix_to_volatility": True,
                },
            ),
            ScenarioDefinition(
                name="credit_spread_widening_200bps",
                description="High-yield spread widening by 200 bps.",
                additive_shocks={
                    "high_yield_spread": 0.02,
                },
                multiplicative_shocks={},
                rules={
                    "propagate_credit_to_risk": True,
                },
            ),
            ScenarioDefinition(
                name="hawkish_policy_shock",
                description="Correlated hawkish shock: higher yields, higher inflation, moderately higher VIX, wider spreads.",
                additive_shocks={
                    "us2y_yield": 0.0125,
                    "us10y_yield": 0.0075,
                    "us_cpi_yoy": 0.0075,
                    "vix_level": 4.0,
                    "high_yield_spread": 0.005,
                },
                multiplicative_shocks={},
                rules={
                    "propagate_rates_to_risk": True,
                    "propagate_inflation_to_rates": True,
                    "propagate_vix_to_volatility": True,
                    "propagate_credit_to_risk": True,
                },
            ),
            ScenarioDefinition(
                name="stagflation_regime",
                description="Higher inflation, wider spreads, higher volatility, weaker equities, supportive gold.",
                additive_shocks={
                    "us_cpi_yoy": 0.015,
                    "vix_level": 8.0,
                    "high_yield_spread": 0.015,
                    "gold_return": 0.010,
                },
                multiplicative_shocks={
                    "spx_vol_3m": 0.15,
                    "spx_vol_6m": 0.15,
                    "ndx_vol_3m": 0.18,
                    "ndx_vol_6m": 0.18,
                },
                rules={
                    "propagate_inflation_to_rates": True,
                    "propagate_vix_to_volatility": True,
                    "propagate_credit_to_risk": True,
                },
            ),
            ScenarioDefinition(
                name="disinflation_growth_scare",
                description="Lower inflation, lower yields, weaker growth, weaker equities, mild gold support.",
                additive_shocks={
                    "us_cpi_yoy": -0.01,
                    "vix_level": 5.0,
                },
                multiplicative_shocks={},
                rules={
                    "propagate_growth_scare": True,
                    "propagate_vix_to_volatility": True,
                },
            ),
            ScenarioDefinition(
                name="systemic_crisis",
                description="Severe risk-off crisis with spread widening, volatility spike, large drawdowns, gold support.",
                additive_shocks={
                    "vix_level": 20.0,
                    "high_yield_spread": 0.03,
                },
                multiplicative_shocks={
                    "spx_vol_3m": 0.25,
                    "spx_vol_6m": 0.25,
                    "ndx_vol_3m": 0.30,
                    "ndx_vol_6m": 0.30,
                },
                rules={
                    "propagate_systemic_crisis": True,
                    "propagate_vix_to_volatility": True,
                    "propagate_credit_to_risk": True,
                },
            ),
            ScenarioDefinition(
                name="usd_dollar_squeeze",
                description="Dollar strength regime with weaker EUR/USD and GBP/USD, weaker risk assets, higher stress.",
                additive_shocks={},
                multiplicative_shocks={},
                rules={
                    "propagate_usd_squeeze": True,
                },
            ),
        ]