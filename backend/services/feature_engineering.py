from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class FeatureEngineerConfig:
    input_path: str = "data/processed/merged_monthly.csv"
    output_path_full: str = "data/features/features_monthly.csv"
    output_path_long: str = "data/features/features_monthly_full_history.csv"


class FeatureEngineer:
    """
    Builds model-ready monthly features from merged_monthly.csv.

    Design:
    - run() only creates engineered columns
    - dataset splitting / dropna is done outside this class
    - this preserves full history for alternative output datasets

    Gold-focused additions:
    - real_yield
    - eurusd_level / gbpusd_level
    - vix_squared
    - gold_momentum_3m
    - gold-specific lagged macro relationships
    """

    def __init__(self, input_path: str) -> None:
        self.input_path = Path(input_path)
        self.df = self._load_data()
        self.column_map = self._build_column_map()

    def _load_data(self) -> pd.DataFrame:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        df = pd.read_csv(self.input_path)

        date_candidates = ["Date", "date", "DATE"]
        date_col = next((c for c in date_candidates if c in df.columns), None)
        if date_col is None:
            raise ValueError("No date column found. Expected one of: Date, date, DATE")

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        if df[date_col].isna().all():
            raise ValueError("Date column could not be parsed.")

        df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
        df = df.set_index(date_col)

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    @staticmethod
    def _normalize(name: str) -> str:
        return (
            str(name)
            .lower()
            .replace("^", "")
            .replace("/", "")
            .replace("-", "")
            .replace(" ", "")
            .replace(".", "")
            .replace("(", "")
            .replace(")", "")
        )

    def _match_column(self, options: List[str]) -> str | None:
        normalized = {self._normalize(c): c for c in self.df.columns}
        for option in options:
            norm_option = self._normalize(option)
            if norm_option in normalized:
                return normalized[norm_option]
        return None

    def _build_column_map(self) -> Dict[str, str]:
        candidates = {
            "spx": ["spx", "^spx_d", "sp500", "s&p500", "spx_close", "spx_price"],
            "ndx": ["ndx", "^ndx_d", "nasdaq100", "nasdaq_100", "ndx_close", "ndx_price"],
            "ftse": ["ftse100", "ftse", "ftse_100", "ftse_close", "ftse_price"],
            "gold": ["gold", "xauusd", "gold_price", "xauusd_close"],
            "btc": ["bitcoin", "btc", "btcusd", "btc_price", "btcusd_close"],
            "eurusd": ["eurusd", "eur_usd", "eurusd_close", "eurusd_price"],
            "gbpusd": ["gbpusd", "gbp_usd", "gbpusd_close", "gbpusd_price"],
            "us2y": ["us2y_yield", "dgs2", "us2y", "us_2y", "treasury_2y"],
            "us10y": ["us10y_yield", "dgs10", "us10y", "us_10y", "treasury_10y"],
            "us_cpi": ["us_cpi", "cpiaucsl", "cpi_us", "us_cpi_index"],
            "uk_cpi": ["uk_cpi", "cpi_uk_d", "cpi_uk", "uk_cpi_index"],
            "hy_spread": ["us_hy_oas", "bamlh0a0hym2", "high_yield_spread", "hy_spread", "credit_spread"],
            "vix": ["vix", "vixcls", "vix_level"],
            "ecb": ["ecb_series"],
            "dxy": ["dxy", "dx_y_nyb", "usd_index", "dxy_close"],
            "qqq": ["qqq", "qqq_close", "qqq_ndx_proxy"],
            "fed_funds": ["fed_funds", "fedfunds", "federal_funds_rate"],
            "tips_10y": ["tips_10y", "dfii10", "tips_real_yield"],
            "breakeven_10y": ["breakeven_10y", "t10yie", "breakeven_inflation"],
        }

        column_map: Dict[str, str] = {}
        for logical_name, options in candidates.items():
            found = self._match_column(options)
            if found is not None:
                column_map[logical_name] = found

        missing_important = [k for k in ["spx", "ndx", "us2y", "us10y", "vix"] if k not in column_map]
        if missing_important:
            raise ValueError(
                "Missing required columns for Phase 3: "
                + ", ".join(missing_important)
                + f"\nAvailable columns: {list(self.df.columns)}"
            )

        return column_map

    def _log_return(self, col: str) -> pd.Series:
        series = self.df[col]
        return np.log(series / series.shift(1))

    def _safe_add_return_feature(self, logical_name: str, feature_name: str) -> None:
        if logical_name in self.column_map:
            self.df[feature_name] = self._log_return(self.column_map[logical_name])

    def compute_returns(self) -> None:
        self._safe_add_return_feature("spx", "spx_return")
        self._safe_add_return_feature("ndx", "ndx_return")
        self._safe_add_return_feature("ftse", "ftse_return")
        self._safe_add_return_feature("gold", "gold_return")
        self._safe_add_return_feature("btc", "btc_return")
        self._safe_add_return_feature("eurusd", "eurusd_return")
        self._safe_add_return_feature("gbpusd", "gbpusd_return")

    def compute_rolling_return_features(self) -> None:
        if "gold_return" in self.df.columns:
            self.df["gold_return_3m"] = self.df["gold_return"].rolling(3).sum()
            self.df["gold_momentum_3m"] = self.df["gold_return"].rolling(3).sum()
            self.df["gold_momentum_6m"] = self.df["gold_return"].rolling(6).sum()

        if "spx_return" in self.df.columns:
            self.df["spx_return_3m"] = self.df["spx_return"].rolling(3).sum()

        if "ndx_return" in self.df.columns:
            self.df["ndx_return_3m"] = self.df["ndx_return"].rolling(3).sum()

    def compute_volatility(self) -> None:
        if "spx_return" in self.df.columns:
            self.df["spx_vol_3m"] = self.df["spx_return"].rolling(3).std()
            self.df["spx_vol_6m"] = self.df["spx_return"].rolling(6).std()

        if "ndx_return" in self.df.columns:
            self.df["ndx_vol_3m"] = self.df["ndx_return"].rolling(3).std()
            self.df["ndx_vol_6m"] = self.df["ndx_return"].rolling(6).std()

        if "gold_return" in self.df.columns:
            self.df["gold_vol_3m"] = self.df["gold_return"].rolling(3).std()
            self.df["gold_vol_6m"] = self.df["gold_return"].rolling(6).std()

        if "vix" in self.column_map:
            self.df["vix_level"] = self.df[self.column_map["vix"]]
            self.df["vix_squared"] = self.df["vix_level"] ** 2

    def compute_rates(self) -> None:
        self.df["us2y_yield"] = self.df[self.column_map["us2y"]]
        self.df["us10y_yield"] = self.df[self.column_map["us10y"]]
        self.df["yield_spread"] = self.df["us10y_yield"] - self.df["us2y_yield"]

    def compute_inflation(self) -> None:
        if "us_cpi" in self.column_map:
            us_cpi_col = self.column_map["us_cpi"]
            self.df["us_cpi_yoy"] = self.df[us_cpi_col] / self.df[us_cpi_col].shift(12) - 1

        if "uk_cpi" in self.column_map:
            uk_cpi_col = self.column_map["uk_cpi"]
            self.df["uk_cpi_yoy"] = self.df[uk_cpi_col] / self.df[uk_cpi_col].shift(12) - 1

    def compute_credit_risk(self) -> None:
        if "hy_spread" in self.column_map:
            self.df["high_yield_spread"] = self.df[self.column_map["hy_spread"]]

    def compute_ecb_features(self) -> None:
        if "ecb" in self.column_map:
            ecb_col = self.column_map["ecb"]
            self.df["ecb_level"] = self.df[ecb_col]
            self.df["ecb_yoy"] = self.df[ecb_col] / self.df[ecb_col].shift(12) - 1

    def compute_fx_level_features(self) -> None:
        if "eurusd" in self.column_map:
            self.df["eurusd_level"] = self.df[self.column_map["eurusd"]]

        if "gbpusd" in self.column_map:
            self.df["gbpusd_level"] = self.df[self.column_map["gbpusd"]]

    def compute_gold_macro_features(self) -> None:
        """
        Gold-specific macro features.
        These are high-impact additions for Phase 6.1+ gold modelling.
        """
        if {"us10y_yield", "us_cpi_yoy"}.issubset(self.df.columns):
            self.df["real_yield"] = self.df["us10y_yield"] - self.df["us_cpi_yoy"]

        if "real_yield" in self.df.columns:
            self.df["real_yield_lag1"] = self.df["real_yield"].shift(1)
            self.df["real_yield_change_1m"] = self.df["real_yield"].diff(1)

        if "vix_level" in self.df.columns:
            self.df["vix_lag1"] = self.df["vix_level"].shift(1)
            self.df["vix_change_1m"] = self.df["vix_level"].diff(1)

        if "yield_spread" in self.df.columns:
            self.df["yield_spread_lag1"] = self.df["yield_spread"].shift(1)

        if "high_yield_spread" in self.df.columns:
            self.df["high_yield_spread_lag1"] = self.df["high_yield_spread"].shift(1)

        if "gold_return" in self.df.columns:
            self.df["gold_return_lag1"] = self.df["gold_return"].shift(1)
            self.df["gold_return_lag2"] = self.df["gold_return"].shift(2)

        if "eurusd_level" in self.df.columns:
            self.df["eurusd_level_lag1"] = self.df["eurusd_level"].shift(1)

        if "gbpusd_level" in self.df.columns:
            self.df["gbpusd_level_lag1"] = self.df["gbpusd_level"].shift(1)

    def compute_dxy_features(self) -> None:
        if "dxy" in self.column_map:
            dxy_col = self.column_map["dxy"]
            self.df["dxy_level"] = self.df[dxy_col]
            self.df["dxy_return"] = np.log(self.df[dxy_col] / self.df[dxy_col].shift(1))
            self.df["dxy_return_3m"] = self.df["dxy_return"].rolling(3).sum()

    def compute_qqq_features(self) -> None:
        if "qqq" in self.column_map:
            qqq_col = self.column_map["qqq"]
            self.df["qqq_return"] = np.log(self.df[qqq_col] / self.df[qqq_col].shift(1))

    def compute_fed_features(self) -> None:
        if "fed_funds" in self.column_map:
            col = self.column_map["fed_funds"]
            self.df["fed_funds_level"] = self.df[col]
            self.df["fed_funds_change_1m"] = self.df[col].diff(1)
            self.df["fed_funds_change_3m"] = self.df[col].diff(3)

    def compute_tips_breakeven_features(self) -> None:
        if "tips_10y" in self.column_map:
            col = self.column_map["tips_10y"]
            self.df["tips_10y_level"] = self.df[col]
            self.df["tips_10y_change_1m"] = self.df[col].diff(1)

        if "breakeven_10y" in self.column_map:
            col = self.column_map["breakeven_10y"]
            self.df["breakeven_10y_level"] = self.df[col]
            self.df["breakeven_10y_change_1m"] = self.df[col].diff(1)

        # Forward-looking real yield: TIPS yield is the market-implied real rate
        # (more precise than us10y_yield - us_cpi_yoy which is backward-looking)
        if "tips_10y_level" in self.df.columns:
            self.df["real_yield_tips"] = self.df["tips_10y_level"]
            self.df["real_yield_tips_lag1"] = self.df["tips_10y_level"].shift(1)
            self.df["real_yield_tips_change_1m"] = self.df["tips_10y_level"].diff(1)

    def compute_stress_features(self) -> None:
        if "vix_level" in self.df.columns:
            vix_mean_12 = self.df["vix_level"].rolling(12).mean()
            vix_std_12 = self.df["vix_level"].rolling(12).std()
            self.df["vix_spike"] = (
                self.df["vix_level"] > (vix_mean_12 + 2 * vix_std_12)
            ).astype(int)

        if "spx" in self.column_map:
            spx_price = self.df[self.column_map["spx"]]
            spx_roll_max = spx_price.cummax()
            self.df["spx_drawdown"] = (spx_price - spx_roll_max) / spx_roll_max
            self.df["spx_max_dd_6m"] = self.df["spx_drawdown"].rolling(6).min()

        if "ndx" in self.column_map:
            ndx_price = self.df[self.column_map["ndx"]]
            ndx_roll_max = ndx_price.cummax()
            self.df["ndx_drawdown"] = (ndx_price - ndx_roll_max) / ndx_roll_max
            self.df["ndx_max_dd_6m"] = self.df["ndx_drawdown"].rolling(6).min()

        if "gold" in self.column_map:
            gold_price = self.df[self.column_map["gold"]]
            gold_roll_max = gold_price.cummax()
            self.df["gold_drawdown"] = (gold_price - gold_roll_max) / gold_roll_max
            self.df["gold_max_dd_6m"] = self.df["gold_drawdown"].rolling(6).min()

    def run(self) -> None:
        self.compute_returns()
        self.compute_rolling_return_features()
        self.compute_volatility()
        self.compute_rates()
        self.compute_inflation()
        self.compute_credit_risk()
        self.compute_ecb_features()
        self.compute_fx_level_features()
        self.compute_gold_macro_features()
        self.compute_stress_features()
        self.compute_dxy_features()
        self.compute_qqq_features()
        self.compute_fed_features()
        self.compute_tips_breakeven_features()