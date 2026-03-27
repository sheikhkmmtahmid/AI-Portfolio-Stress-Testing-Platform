from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


LONG_HISTORY_TARGETS = {
    "spx": "spx_return",
    "ndx": "ndx_return",
    "gold": "gold_return",
    "btc": "btc_return",
}

CORE_NUMERIC_FEATURES = [
    "us2y_yield",
    "us10y_yield",
    "yield_spread",
    "us_cpi_yoy",
    "high_yield_spread",
    "vix_level",
    "spx_vol_3m",
    "eurusd_return",
    "gbpusd_return",
    "spx_return",
    "ndx_return",
    "gold_return",
    "regime_confidence",
]

OPTIONAL_NUMERIC_FEATURES = [
    "ecb_level",
    "ecb_yoy",
]

CATEGORICAL_FEATURES = ["regime_label"]

LAG_SOURCE_COLUMNS = [
    "spx_return",
    "ndx_return",
    "gold_return",
    "vix_level",
    "yield_spread",
    "high_yield_spread",
]

GOLD_EXTRA_FEATURES = [
    "real_yield",
    "real_yield_lag1",
    "real_yield_change_1m",
    "eurusd_level",
    "gbpusd_level",
    "eurusd_level_lag1",
    "gbpusd_level_lag1",
    "vix_squared",
    "vix_lag1",
    "vix_change_1m",
    "yield_spread_lag1",
    "high_yield_spread_lag1",
    "gold_momentum_3m",
    "gold_momentum_6m",
    "gold_return_lag1",
    "gold_return_lag2",
]

ELASTIC_PARAM_GRID = {
    "alpha": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
    "l1_ratio": [0.05, 0.2, 0.5, 0.8, 0.95],
}

XGB_PARAM_GRID = {
    "n_estimators": [50, 100],
    "max_depth": [2, 3],
    "learning_rate": [0.01, 0.03],
    "subsample": [0.6, 0.7],
    "colsample_bytree": [0.6, 0.7],
    "reg_alpha": [0.5, 1.0, 2.0],
    "reg_lambda": [2.0, 5.0, 10.0],
}


@dataclass
class Phase61Paths:
    backend_root: Path
    features_file: Path
    regimes_file: Path
    output_dir: Path
    rolling_dir: Path
    tuning_dir: Path
    diagnostics_dir: Path


class Timer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def fmt(self, seconds: float) -> str:
        total_ms = int(round(seconds * 1000))
        hh = total_ms // 3_600_000
        mm = (total_ms % 3_600_000) // 60_000
        ss = (total_ms % 60_000) // 1000
        ms = total_ms % 1000
        return f"{hh:02d}:{mm:02d}:{ss:02d}:{ms:03d}"

    def elapsed(self) -> str:
        return self.fmt(time.perf_counter() - self.t0)

    def log(self, phase: str, msg: str) -> None:
        print(f"[{self.elapsed()}] [{phase}] {msg}", flush=True)


class Phase61Refiner:
    def __init__(
        self,
        backend_root: str | Path | None = None,
        random_state: int = 42,
        features_file_override: Path | None = None,
        target_assets_filter: list | None = None,
    ) -> None:
        if backend_root is None:
            backend_root = Path(__file__).resolve().parents[1]
        self.backend_root = Path(backend_root).resolve()
        self.random_state = random_state
        self.timer = Timer()
        self._features_file_override = Path(features_file_override) if features_file_override is not None else None
        self._target_assets_filter = [a.lower() for a in target_assets_filter] if target_assets_filter is not None else None
        self.paths = self._build_paths(self.backend_root)

    def _build_paths(self, backend_root: Path) -> Phase61Paths:
        output_dir = backend_root / "models" / "phase6_1"
        rolling_dir = output_dir / "rolling_validation"
        tuning_dir = output_dir / "tuning"
        diagnostics_dir = output_dir / "diagnostics"

        for d in [output_dir, rolling_dir, tuning_dir, diagnostics_dir]:
            d.mkdir(parents=True, exist_ok=True)

        features_file = (
            self._features_file_override
            if self._features_file_override is not None
            else backend_root / "data" / "features" / "features_monthly_full_history.csv"
        )

        return Phase61Paths(
            backend_root=backend_root,
            features_file=features_file,
            regimes_file=backend_root / "data" / "regimes" / "regime_dataset.csv",
            output_dir=output_dir,
            rolling_dir=rolling_dir,
            tuning_dir=tuning_dir,
            diagnostics_dir=diagnostics_dir,
        )

    def run(self) -> Dict[str, Any]:
        self.timer.log("START", "Phase 6.1 refinement started")

        active_targets = LONG_HISTORY_TARGETS
        if self._target_assets_filter is not None:
            active_targets = {k: v for k, v in LONG_HISTORY_TARGETS.items() if k in self._target_assets_filter}

        df = self._load_dataset(active_targets)
        df = self._add_lag_features(df)

        summary: Dict[str, Any] = {"assets": {}}

        for asset, source_col in active_targets.items():
            feature_columns = self._get_feature_columns(df, asset)
            self.timer.log("CONFIG", f"{asset.upper()} using {len(feature_columns)} features")
            self.timer.log("ASSET", f"Starting tuning for {asset.upper()}")

            asset_df = self._build_asset_dataset(df.copy(), asset, source_col, feature_columns)

            elastic_best, elastic_trace = self._tune_elastic_net(asset_df, feature_columns, asset)
            xgb_best, xgb_trace = self._tune_xgboost(asset_df, feature_columns, asset)

            rolling_df = self._run_rolling_validation(
                asset_df=asset_df,
                asset=asset,
                feature_columns=feature_columns,
                elastic_params=elastic_best,
                xgb_params=xgb_best,
            )

            rolling_path = self.paths.rolling_dir / f"{asset}_rolling_validation_tuned.csv"
            rolling_df.to_csv(rolling_path, index=False)

            asset_summary = {
                "feature_columns": feature_columns,
                "elastic_net_best_params": elastic_best,
                "xgboost_best_params": xgb_best,
                "rolling_validation_file": str(rolling_path),
                "elastic_net_tuning_file": str(self.paths.tuning_dir / f"{asset}_elastic_tuning_results.csv"),
                "xgboost_tuning_file": str(self.paths.tuning_dir / f"{asset}_xgb_tuning_results.csv"),
                "rolling_summary": self._summarise_rolling(rolling_df),
            }

            with open(self.paths.diagnostics_dir / f"{asset}_elastic_best_params.json", "w", encoding="utf-8") as f:
                json.dump(elastic_best, f, indent=2)

            with open(self.paths.diagnostics_dir / f"{asset}_xgb_best_params.json", "w", encoding="utf-8") as f:
                json.dump(xgb_best, f, indent=2)

            with open(self.paths.diagnostics_dir / f"{asset}_feature_columns.json", "w", encoding="utf-8") as f:
                json.dump({"feature_columns": feature_columns}, f, indent=2)

            elastic_trace.to_csv(self.paths.tuning_dir / f"{asset}_elastic_tuning_results.csv", index=False)
            xgb_trace.to_csv(self.paths.tuning_dir / f"{asset}_xgb_tuning_results.csv", index=False)

            summary["assets"][asset] = asset_summary
            self.timer.log("ASSET", f"{asset.upper()} completed")

        summary_path = self.paths.output_dir / "phase6_1_summary.json"
        # Merge into existing summary so separate runs don't overwrite each other
        existing_summary: dict = {"assets": {}}
        if summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    existing_summary = json.load(f)
            except Exception:
                existing_summary = {"assets": {}}
        existing_summary["assets"].update(summary["assets"])
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(existing_summary, f, indent=2)

        self.timer.log("END", f"Phase 6.1 completed. Summary saved to {summary_path}")
        return summary

    def _load_dataset(self, active_targets: Dict[str, str]) -> pd.DataFrame:
        self.timer.log("LOAD", f"Reading {self.paths.features_file}")
        features_df = pd.read_csv(self.paths.features_file)

        self.timer.log("LOAD", f"Reading {self.paths.regimes_file}")
        regimes_df = pd.read_csv(self.paths.regimes_file)

        features_df["date"] = pd.to_datetime(features_df["date"])
        regimes_df["date"] = pd.to_datetime(regimes_df["date"])

        regimes_keep = ["date", "regime_label", "regime_confidence"]
        regimes_df = regimes_df[regimes_keep].drop_duplicates(subset=["date"]).sort_values("date")

        df = features_df.merge(regimes_df, on="date", how="left").sort_values("date").reset_index(drop=True)

        # Only validate presence of columns needed for the active targets
        required_core = [c for c in CORE_NUMERIC_FEATURES if c in df.columns] + CATEGORICAL_FEATURES + list(active_targets.values())
        missing = [c for c in required_core if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for Phase 6.1: {missing}")

        diag = {
            "rows": int(len(df)),
            "date_min": str(df["date"].min().date()),
            "date_max": str(df["date"].max().date()),
            "columns": list(df.columns),
            "core_numeric_features_present": [c for c in CORE_NUMERIC_FEATURES if c in df.columns],
            "optional_numeric_features_present": [c for c in OPTIONAL_NUMERIC_FEATURES if c in df.columns],
            "optional_numeric_features_missing": [c for c in OPTIONAL_NUMERIC_FEATURES if c not in df.columns],
            "gold_extra_features_present": [c for c in GOLD_EXTRA_FEATURES if c in df.columns],
        }
        with open(self.paths.diagnostics_dir / "phase6_1_dataset_diagnostics.json", "w", encoding="utf-8") as f:
            json.dump(diag, f, indent=2)

        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self.timer.log("FEATURES", "Adding common lag features")
        for col in LAG_SOURCE_COLUMNS:
            if col in df.columns:
                lag1 = f"{col}_lag1"
                lag2 = f"{col}_lag2"
                if lag1 not in df.columns:
                    df[lag1] = df[col].shift(1)
                if lag2 not in df.columns:
                    df[lag2] = df[col].shift(2)
        return df

    def _get_feature_columns(self, df: pd.DataFrame, asset: str) -> List[str]:
        features = []

        for col in CORE_NUMERIC_FEATURES:
            if col in df.columns and col not in features:
                features.append(col)

        for col in OPTIONAL_NUMERIC_FEATURES:
            if col in df.columns and col not in features:
                features.append(col)

        for col in LAG_SOURCE_COLUMNS:
            lag1 = f"{col}_lag1"
            lag2 = f"{col}_lag2"
            if lag1 in df.columns and lag1 not in features:
                features.append(lag1)
            if lag2 in df.columns and lag2 not in features:
                features.append(lag2)

        if asset == "gold":
            for col in GOLD_EXTRA_FEATURES:
                if col in df.columns and col not in features:
                    features.append(col)

        for col in CATEGORICAL_FEATURES:
            if col in df.columns and col not in features:
                features.append(col)

        return features

    def _build_asset_dataset(
        self,
        df: pd.DataFrame,
        asset: str,
        source_col: str,
        feature_columns: List[str],
    ) -> pd.DataFrame:
        target_col = f"{asset}_target"
        df[target_col] = df[source_col].shift(-1)

        use_cols = ["date"] + feature_columns + [target_col]
        asset_df = df[use_cols].copy()
        asset_df = asset_df.dropna(subset=[target_col]).reset_index(drop=True)
        return asset_df

    def _rolling_year_splits(self, asset_df: pd.DataFrame) -> List[Tuple[int, pd.DataFrame, pd.DataFrame]]:
        splits = []

        for split_year in range(2010, 2019):
            train_df = asset_df[asset_df["date"].dt.year <= split_year].copy()
            test_df = asset_df[asset_df["date"].dt.year == split_year + 1].copy()

            if train_df.empty or test_df.empty:
                continue

            splits.append((split_year + 1, train_df, test_df))

        return splits

    def _make_preprocessor(self, feature_columns: List[str]) -> ColumnTransformer:
        numeric_cols = [c for c in feature_columns if c != "regime_label"]
        categorical_cols = [c for c in feature_columns if c == "regime_label"]

        transformers = []

        if numeric_cols:
            transformers.append(
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_cols,
                )
            )

        if categorical_cols:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_cols,
                )
            )

        return ColumnTransformer(transformers=transformers, remainder="drop")

    def _calc_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            "r2": float(r2_score(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "directional_accuracy": float(np.mean(np.sign(y_true.values) == np.sign(y_pred))),
        }

    def _score_for_selection(self, metrics_list: List[Dict[str, float]]) -> float:
        mean_dir = float(np.mean([m["directional_accuracy"] for m in metrics_list]))
        mean_rmse = float(np.mean([m["rmse"] for m in metrics_list]))
        return mean_dir - 0.05 * mean_rmse

    def _tune_elastic_net(
        self,
        asset_df: pd.DataFrame,
        feature_columns: List[str],
        asset: str,
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        self.timer.log("TUNE", f"{asset.upper()} Elastic Net tuning started")
        splits = self._rolling_year_splits(asset_df)

        rows = []
        total = len(ELASTIC_PARAM_GRID["alpha"]) * len(ELASTIC_PARAM_GRID["l1_ratio"])
        done = 0

        best_params: Dict[str, float] | None = None
        best_score = -np.inf

        for alpha in ELASTIC_PARAM_GRID["alpha"]:
            for l1_ratio in ELASTIC_PARAM_GRID["l1_ratio"]:
                done += 1
                self.timer.log("TUNE", f"{asset.upper()} Elastic {done}/{total}")

                fold_metrics = []

                for test_year, train_df, test_df in splits:
                    X_train = train_df[feature_columns]
                    y_train = train_df[f"{asset}_target"]
                    X_test = test_df[feature_columns]
                    y_test = test_df[f"{asset}_target"]

                    preprocessor = self._make_preprocessor(feature_columns)
                    model = ElasticNet(
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                        max_iter=10000,
                        random_state=self.random_state,
                    )

                    X_train_p = preprocessor.fit_transform(X_train)
                    X_test_p = preprocessor.transform(X_test)

                    model.fit(X_train_p, y_train)
                    pred = model.predict(X_test_p)
                    metrics = self._calc_metrics(y_test, pred)
                    metrics["test_year"] = test_year
                    fold_metrics.append(metrics)

                score = self._score_for_selection(fold_metrics)
                mean_dir = float(np.mean([m["directional_accuracy"] for m in fold_metrics]))
                mean_rmse = float(np.mean([m["rmse"] for m in fold_metrics]))
                mean_r2 = float(np.mean([m["r2"] for m in fold_metrics]))

                rows.append(
                    {
                        "alpha": alpha,
                        "l1_ratio": l1_ratio,
                        "mean_directional_accuracy": mean_dir,
                        "mean_rmse": mean_rmse,
                        "mean_r2": mean_r2,
                        "selection_score": score,
                    }
                )

                if score > best_score:
                    best_score = score
                    best_params = {"alpha": float(alpha), "l1_ratio": float(l1_ratio)}

        if best_params is None:
            raise ValueError(f"Elastic tuning failed for {asset}")

        trace_df = pd.DataFrame(rows).sort_values(
            ["mean_directional_accuracy", "mean_r2", "mean_rmse"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

        self.timer.log("TUNE", f"{asset.upper()} Elastic best params: {best_params}")
        return best_params, trace_df

    def _tune_xgboost(
        self,
        asset_df: pd.DataFrame,
        feature_columns: List[str],
        asset: str,
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        self.timer.log("TUNE", f"{asset.upper()} XGBoost tuning started")
        splits = self._rolling_year_splits(asset_df)

        grid = XGB_PARAM_GRID
        total = (
            len(grid["n_estimators"])
            * len(grid["max_depth"])
            * len(grid["learning_rate"])
            * len(grid["subsample"])
            * len(grid["colsample_bytree"])
            * len(grid["reg_alpha"])
            * len(grid["reg_lambda"])
        )

        rows = []
        done = 0
        best_params: Dict[str, float] | None = None
        best_score = -np.inf

        for n_estimators in grid["n_estimators"]:
            for max_depth in grid["max_depth"]:
                for learning_rate in grid["learning_rate"]:
                    for subsample in grid["subsample"]:
                        for colsample_bytree in grid["colsample_bytree"]:
                            for reg_alpha in grid["reg_alpha"]:
                                for reg_lambda in grid["reg_lambda"]:
                                    done += 1
                                    self.timer.log("TUNE", f"{asset.upper()} XGB {done}/{total}")

                                    fold_metrics = []

                                    for test_year, train_df, test_df in splits:
                                        X_train = train_df[feature_columns]
                                        y_train = train_df[f"{asset}_target"]
                                        X_test = test_df[feature_columns]
                                        y_test = test_df[f"{asset}_target"]

                                        preprocessor = self._make_preprocessor(feature_columns)
                                        X_train_p = preprocessor.fit_transform(X_train)
                                        X_test_p = preprocessor.transform(X_test)

                                        if hasattr(X_train_p, "toarray"):
                                            X_train_p = X_train_p.toarray()
                                        if hasattr(X_test_p, "toarray"):
                                            X_test_p = X_test_p.toarray()

                                        model = XGBRegressor(
                                            n_estimators=n_estimators,
                                            max_depth=max_depth,
                                            learning_rate=learning_rate,
                                            subsample=subsample,
                                            colsample_bytree=colsample_bytree,
                                            reg_alpha=reg_alpha,
                                            reg_lambda=reg_lambda,
                                            random_state=self.random_state,
                                            objective="reg:squarederror",
                                            eval_metric="rmse",
                                        )

                                        model.fit(X_train_p, y_train, verbose=False)
                                        pred = model.predict(X_test_p)
                                        metrics = self._calc_metrics(y_test, pred)
                                        metrics["test_year"] = test_year
                                        fold_metrics.append(metrics)

                                    score = self._score_for_selection(fold_metrics)
                                    mean_dir = float(np.mean([m["directional_accuracy"] for m in fold_metrics]))
                                    mean_rmse = float(np.mean([m["rmse"] for m in fold_metrics]))
                                    mean_r2 = float(np.mean([m["r2"] for m in fold_metrics]))

                                    rows.append(
                                        {
                                            "n_estimators": n_estimators,
                                            "max_depth": max_depth,
                                            "learning_rate": learning_rate,
                                            "subsample": subsample,
                                            "colsample_bytree": colsample_bytree,
                                            "reg_alpha": reg_alpha,
                                            "reg_lambda": reg_lambda,
                                            "mean_directional_accuracy": mean_dir,
                                            "mean_rmse": mean_rmse,
                                            "mean_r2": mean_r2,
                                            "selection_score": score,
                                        }
                                    )

                                    if score > best_score:
                                        best_score = score
                                        best_params = {
                                            "n_estimators": int(n_estimators),
                                            "max_depth": int(max_depth),
                                            "learning_rate": float(learning_rate),
                                            "subsample": float(subsample),
                                            "colsample_bytree": float(colsample_bytree),
                                            "reg_alpha": float(reg_alpha),
                                            "reg_lambda": float(reg_lambda),
                                        }

        if best_params is None:
            raise ValueError(f"XGBoost tuning failed for {asset}")

        trace_df = pd.DataFrame(rows).sort_values(
            ["mean_directional_accuracy", "mean_r2", "mean_rmse"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

        self.timer.log("TUNE", f"{asset.upper()} XGBoost best params: {best_params}")
        return best_params, trace_df

    def _run_rolling_validation(
        self,
        asset_df: pd.DataFrame,
        asset: str,
        feature_columns: List[str],
        elastic_params: Dict[str, float],
        xgb_params: Dict[str, float],
    ) -> pd.DataFrame:
        self.timer.log("ROLLING", f"{asset.upper()} rolling validation started")
        splits = self._rolling_year_splits(asset_df)
        rows = []

        for i, (test_year, train_df, test_df) in enumerate(splits, start=1):
            self.timer.log("ROLLING", f"{asset.upper()} fold {i}/{len(splits)} test_year={test_year}")

            X_train = train_df[feature_columns]
            y_train = train_df[f"{asset}_target"]
            X_test = test_df[feature_columns]
            y_test = test_df[f"{asset}_target"]

            pre_elastic = self._make_preprocessor(feature_columns)
            X_train_el = pre_elastic.fit_transform(X_train)
            X_test_el = pre_elastic.transform(X_test)

            elastic = ElasticNet(
                alpha=elastic_params["alpha"],
                l1_ratio=elastic_params["l1_ratio"],
                max_iter=10000,
                random_state=self.random_state,
            )
            elastic.fit(X_train_el, y_train)
            elastic_pred = elastic.predict(X_test_el)

            pre_xgb = self._make_preprocessor(feature_columns)
            X_train_xgb = pre_xgb.fit_transform(X_train)
            X_test_xgb = pre_xgb.transform(X_test)

            if hasattr(X_train_xgb, "toarray"):
                X_train_xgb = X_train_xgb.toarray()
            if hasattr(X_test_xgb, "toarray"):
                X_test_xgb = X_test_xgb.toarray()

            xgb = XGBRegressor(
                n_estimators=xgb_params["n_estimators"],
                max_depth=xgb_params["max_depth"],
                learning_rate=xgb_params["learning_rate"],
                subsample=xgb_params["subsample"],
                colsample_bytree=xgb_params["colsample_bytree"],
                reg_alpha=xgb_params["reg_alpha"],
                reg_lambda=xgb_params["reg_lambda"],
                random_state=self.random_state,
                objective="reg:squarederror",
                eval_metric="rmse",
            )
            xgb.fit(X_train_xgb, y_train, verbose=False)
            xgb_pred = xgb.predict(X_test_xgb)

            naive_pred = np.zeros(len(y_test), dtype=float)

            el_metrics = self._calc_metrics(y_test, elastic_pred)
            xgb_metrics = self._calc_metrics(y_test, xgb_pred)
            naive_metrics = self._calc_metrics(y_test, naive_pred)

            rows.append(
                {
                    "year": int(test_year),
                    "n_test_rows": int(len(test_df)),
                    "elastic_dir": el_metrics["directional_accuracy"],
                    "elastic_rmse": el_metrics["rmse"],
                    "elastic_mae": el_metrics["mae"],
                    "elastic_r2": el_metrics["r2"],
                    "xgb_dir": xgb_metrics["directional_accuracy"],
                    "xgb_rmse": xgb_metrics["rmse"],
                    "xgb_mae": xgb_metrics["mae"],
                    "xgb_r2": xgb_metrics["r2"],
                    "naive_dir": naive_metrics["directional_accuracy"],
                    "naive_rmse": naive_metrics["rmse"],
                    "naive_mae": naive_metrics["mae"],
                    "naive_r2": naive_metrics["r2"],
                }
            )

        return pd.DataFrame(rows)

    def _summarise_rolling(self, rolling_df: pd.DataFrame) -> Dict[str, float]:
        return {
            "elastic_mean_dir": float(rolling_df["elastic_dir"].mean()),
            "elastic_mean_rmse": float(rolling_df["elastic_rmse"].mean()),
            "elastic_mean_r2": float(rolling_df["elastic_r2"].mean()),
            "xgb_mean_dir": float(rolling_df["xgb_dir"].mean()),
            "xgb_mean_rmse": float(rolling_df["xgb_rmse"].mean()),
            "xgb_mean_r2": float(rolling_df["xgb_r2"].mean()),
            "naive_mean_dir": float(rolling_df["naive_dir"].mean()),
            "naive_mean_rmse": float(rolling_df["naive_rmse"].mean()),
            "naive_mean_r2": float(rolling_df["naive_r2"].mean()),
        }