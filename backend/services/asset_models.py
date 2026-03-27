from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


LONG_HISTORY_TARGET_CANDIDATES = {
    "spx": "spx_return",
    "ndx": "ndx_return",
    "gold": "gold_return",
    "ftse": "ftse_return",
    "btc": "btc_return",
}

COLUMN_ALIASES = {
    "spx_return": ["sp500_return", "spx_ret"],
    "ndx_return": ["nasdaq100_return", "ndx_ret"],
    "gold_return": ["xauusd_return", "gold_ret"],
    "ftse_return": ["ftse100_return", "ftse_100_return", "ftse_ret"],
    "btc_return": ["bitcoin_return", "btcusd_return", "btc_usd_return", "btc_ret"],
    "us2y_yield": ["dgs2", "us_2y_yield"],
    "us10y_yield": ["dgs10", "us_10y_yield"],
    "high_yield_spread": ["hy_spread", "high_yield_oad"],
    "vix_level": ["vix", "vix_close"],
    "ecb_level": ["ecb_series", "ecb_value"],
    "regime_confidence": ["max_regime_probability", "cluster_confidence", "confidence"],
    "regime_label": ["regime", "cluster_label", "economic_regime"],
}

LONG_HISTORY_NUMERIC_FEATURE_CANDIDATES = [
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
    "ecb_level",
    "ecb_yoy",
    "regime_confidence",
]

LONG_HISTORY_OPTIONAL_FEATURES = [
    "uk_cpi_yoy",
    "ftse_return",
    "btc_return",
    # New datasets: Fed Funds, TIPS real yield, Breakeven inflation, DXY, QQQ
    "fed_funds_level",
    "fed_funds_change_1m",
    "tips_10y_level",
    "tips_10y_change_1m",
    "real_yield_tips",
    "real_yield_tips_change_1m",
    "breakeven_10y_level",
    "breakeven_10y_change_1m",
    "dxy_return",
    "dxy_return_3m",
    "qqq_return",
]

CATEGORICAL_FEATURE_CANDIDATES = ["regime_label"]

DATE_COLUMN_CANDIDATES = ["date", "Date", "month", "Month"]
REGIME_REQUIRED_COLUMNS = ["date", "regime_label", "regime_confidence"]


@dataclass
class Paths:
    project_root: Path
    features_file: Path
    regimes_file: Path
    phase6_model_dir: Path
    predictions_dir: Path
    feature_importance_dir: Path
    evaluation_dir: Path
    training_cache_dir: Path


class PhaseTimer:
    def __init__(self) -> None:
        self.global_start = time.perf_counter()

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}:{millis:03d}"

    def elapsed(self) -> str:
        return self._format_elapsed(time.perf_counter() - self.global_start)

    def log(self, phase: str, message: str) -> None:
        print(f"[{self.elapsed()}] [{phase}] {message}", flush=True)


class ProgressTracker:
    def __init__(self, timer: PhaseTimer, phase: str, total_steps: int) -> None:
        self.timer = timer
        self.phase = phase
        self.total_steps = max(total_steps, 1)
        self.current_step = 0

    def update(self, message: str) -> None:
        self.current_step += 1
        pct = (self.current_step / self.total_steps) * 100.0
        self.timer.log(self.phase, f"{self.current_step}/{self.total_steps} ({pct:6.2f}%) - {message}")


class AssetSensitivityTrainer:
    def __init__(
        self,
        project_root: str | Path | None = None,
        train_end_date: str = "2018-12-31",
        test_start_date: str = "2019-01-01",
        random_state: int = 42,
        features_file_override: str | Path | None = None,
        target_assets_filter: list | None = None,
    ) -> None:
        self._target_assets_filter = [a.lower() for a in target_assets_filter] if target_assets_filter else None
        self.paths = self._build_paths(project_root)
        if features_file_override is not None:
            self.paths = Paths(
                project_root=self.paths.project_root,
                features_file=Path(features_file_override),
                regimes_file=self.paths.regimes_file,
                phase6_model_dir=self.paths.phase6_model_dir,
                predictions_dir=self.paths.predictions_dir,
                feature_importance_dir=self.paths.feature_importance_dir,
                evaluation_dir=self.paths.evaluation_dir,
                training_cache_dir=self.paths.training_cache_dir,
            )
        self.train_end_date = pd.Timestamp(train_end_date)
        self.test_start_date = pd.Timestamp(test_start_date)
        self.random_state = random_state
        self.timer = PhaseTimer()
        self.run_started_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        self.active_numeric_features: List[str] = []
        self.active_categorical_features: List[str] = []
        self.available_targets: Dict[str, str] = {}
        self.skipped_targets: Dict[str, str] = {}

    @staticmethod
    def _build_paths(project_root: str | Path | None) -> Paths:
        if project_root is None:
            project_root = Path(__file__).resolve().parents[1]
        else:
            project_root = Path(project_root).resolve()

        phase6_model_dir = project_root / "models" / "phase6"
        predictions_dir = phase6_model_dir / "predictions"
        feature_importance_dir = phase6_model_dir / "feature_importance"
        evaluation_dir = phase6_model_dir / "evaluation"
        training_cache_dir = phase6_model_dir / "training_cache"

        for directory in [
            phase6_model_dir,
            predictions_dir,
            feature_importance_dir,
            evaluation_dir,
            training_cache_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        return Paths(
            project_root=project_root,
            features_file=project_root / "data" / "features" / "features_monthly_full_history.csv",
            regimes_file=project_root / "data" / "regimes" / "regime_dataset.csv",
            phase6_model_dir=phase6_model_dir,
            predictions_dir=predictions_dir,
            feature_importance_dir=feature_importance_dir,
            evaluation_dir=evaluation_dir,
            training_cache_dir=training_cache_dir,
        )

    def run(self) -> Dict[str, Any]:
        self.timer.log("START", f"Phase 6 started. Project root: {self.paths.project_root}")
        df = self._load_and_prepare_dataset()

        if not self.available_targets:
            raise ValueError("No valid long-history targets available. Phase 6 cannot proceed.")

        # Apply asset filter if provided (e.g. train only BTC)
        if self._target_assets_filter is not None:
            self.available_targets = {
                k: v for k, v in self.available_targets.items()
                if k.lower() in self._target_assets_filter
            }

        self.timer.log("CONFIG", f"Active numeric features: {self.active_numeric_features}")
        self.timer.log("CONFIG", f"Active categorical features: {self.active_categorical_features}")
        self.timer.log("CONFIG", f"Targets to train: {list(self.available_targets.keys())}")
        if self.skipped_targets:
            self.timer.log("CONFIG", f"Skipped targets: {self.skipped_targets}")

        assets_metadata: List[Dict[str, Any]] = []
        metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

        asset_tracker = ProgressTracker(
            self.timer,
            "TRAINING",
            total_steps=len(self.available_targets),
        )

        for asset_name, target_base_col in self.available_targets.items():
            asset_tracker.update(f"Starting asset pipeline for {asset_name.upper()}")
            asset_df, target_col = self._build_target(df.copy(), asset_name, target_base_col)
            train_df, test_df = self._train_test_split(asset_df)

            train_snapshot_path = self.paths.training_cache_dir / f"train_snapshot_{asset_name}.csv"
            test_snapshot_path = self.paths.training_cache_dir / f"test_snapshot_{asset_name}.csv"
            train_df.to_csv(train_snapshot_path, index=False)
            test_df.to_csv(test_snapshot_path, index=False)
            self.timer.log("DATA", f"Saved train/test snapshots for {asset_name.upper()}")

            feature_columns = self.active_numeric_features + self.active_categorical_features

            X_train = train_df[feature_columns].copy()
            y_train = train_df[target_col].copy()
            X_test = test_df[feature_columns].copy()
            y_test = test_df[target_col].copy()

            elastic_results = self._fit_elastic_net(
                asset_name=asset_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                train_df=train_df,
                test_df=test_df,
                feature_columns=feature_columns,
            )

            xgb_results = self._fit_xgboost(
                asset_name=asset_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                train_df=train_df,
                test_df=test_df,
                feature_columns=feature_columns,
            )

            metrics[asset_name] = {
                "elastic_net": elastic_results["metrics"],
                "xgboost": xgb_results["metrics"],
            }

            assets_metadata.append(
                {
                    "asset": asset_name,
                    "target_column": target_col,
                    "train_rows": int(len(train_df)),
                    "test_rows": int(len(test_df)),
                    "feature_columns": feature_columns,
                    "train_snapshot": str(train_snapshot_path),
                    "test_snapshot": str(test_snapshot_path),
                    "elastic_net_model": elastic_results["model_path"],
                    "elastic_net_preprocessor": elastic_results["preprocessor_path"],
                    "elastic_net_predictions": elastic_results["prediction_path"],
                    "elastic_net_cv_results": elastic_results["cv_results_path"],
                    "xgboost_model": xgb_results["model_path"],
                    "xgboost_predictions": xgb_results["prediction_path"],
                    "xgboost_training_trace": xgb_results["training_trace_path"],
                    "feature_importance_files": {
                        "elastic_net": elastic_results["importance_path"],
                        "xgboost": xgb_results["importance_path"],
                    },
                }
            )

        metrics_path = self.paths.phase6_model_dir / "phase6_metrics.json"
        # Merge into existing file so separate runs (e.g. BTC-only run) don't
        # wipe metrics written by a previous run (e.g. SPX/NDX/Gold run).
        existing: dict = {}
        if metrics_path.exists():
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        existing.update(metrics)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
        self.timer.log("SAVE", f"Saved metrics to {metrics_path}")

        metadata = {
            "phase": "phase6_asset_sensitivity_models",
            "status": "completed",
            "architecture": "long_history_core_models",
            "created_at_utc": self.run_started_at,
            "project_root": str(self.paths.project_root),
            "data_sources": {
                "features_monthly_full_history": str(self.paths.features_file),
                "regime_dataset": str(self.paths.regimes_file),
            },
            "split": {
                "train_end_date": str(self.train_end_date.date()),
                "test_start_date": str(self.test_start_date.date()),
                "split_type": "time_based",
            },
            "features": {
                "active_numeric": self.active_numeric_features,
                "active_categorical": self.active_categorical_features,
                "optional_not_used": [f for f in LONG_HISTORY_OPTIONAL_FEATURES if f not in self.active_numeric_features],
            },
            "targets_trained": self.available_targets,
            "targets_skipped": self.skipped_targets,
            "artifacts": assets_metadata,
            "retraining_note": {
                "elastic_net": "Designed for full retraining on merged historical data. Preprocessing artifacts and snapshots are saved for reproducibility.",
                "xgboost": "Model artifacts, training traces, and snapshots are saved. Future retraining can use merged data or careful warm-start continuation.",
            },
            "future_extensions": {
                "uk_ftse_variant": "Train on a shorter-history dataset that includes ftse_return and uk_cpi_yoy.",
                "crypto_variant": "Train on a shorter-history dataset that includes btc_return.",
            },
        }

        metadata_path = self.paths.phase6_model_dir / "phase6_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        self.timer.log("SAVE", f"Saved metadata to {metadata_path}")

        self.timer.log("END", "Phase 6 completed successfully")
        return {"metrics": metrics, "metadata": metadata}

    def _load_and_prepare_dataset(self) -> pd.DataFrame:
        self.timer.log("LOAD", f"Reading features file: {self.paths.features_file}")
        features_df = pd.read_csv(self.paths.features_file)

        self.timer.log("LOAD", f"Reading regime file: {self.paths.regimes_file}")
        regime_df = pd.read_csv(self.paths.regimes_file)

        features_df = self._ensure_date_column(features_df, source_name="features")
        regime_df = self._ensure_date_column(regime_df, source_name="regimes")

        features_df = self._apply_aliases(features_df)
        regime_df = self._apply_aliases(regime_df)

        missing_regime_columns = [col for col in REGIME_REQUIRED_COLUMNS if col not in regime_df.columns]
        if missing_regime_columns:
            raise ValueError(f"Missing required regime columns: {missing_regime_columns}")

        regime_df = regime_df[REGIME_REQUIRED_COLUMNS].copy()
        regime_df = regime_df.drop_duplicates(subset=["date"]).sort_values("date")

        merged_df = features_df.merge(regime_df, on="date", how="left")
        merged_df = merged_df.sort_values("date").reset_index(drop=True)

        self._write_column_diagnostics(features_df, regime_df, merged_df)

        self.active_numeric_features = [
            col for col in LONG_HISTORY_NUMERIC_FEATURE_CANDIDATES if col in merged_df.columns
        ]
        self.active_categorical_features = [
            col for col in CATEGORICAL_FEATURE_CANDIDATES if col in merged_df.columns
        ]

        missing_core_features = [
            col for col in [
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
                "ecb_level",
                "ecb_yoy",
                "regime_confidence",
                "regime_label",
            ]
            if col not in merged_df.columns
        ]
        if missing_core_features:
            raise ValueError(f"Missing required long-history core features: {missing_core_features}")

        self.available_targets = {}
        self.skipped_targets = {}

        for asset_name, base_col in LONG_HISTORY_TARGET_CANDIDATES.items():
            if base_col in merged_df.columns:
                self.available_targets[asset_name] = base_col
            else:
                self.skipped_targets[asset_name] = f"Target column '{base_col}' not present in long-history dataset"

        # Keep target base columns alongside feature columns so _build_target can shift them
        target_base_cols = [
            col for col in self.available_targets.values()
            if col in merged_df.columns and col not in self.active_numeric_features
        ]
        usable_columns = ["date"] + self.active_numeric_features + self.active_categorical_features + target_base_cols
        merged_df = merged_df[usable_columns].copy()
        merged_df = merged_df.sort_values("date").reset_index(drop=True)

        self.timer.log(
            "LOAD",
            f"Merged dataset ready. Rows={len(merged_df)}, Numeric features={len(self.active_numeric_features)}, Categorical features={len(self.active_categorical_features)}",
        )
        return merged_df

    def _write_column_diagnostics(
        self,
        features_df: pd.DataFrame,
        regime_df: pd.DataFrame,
        merged_df: pd.DataFrame,
    ) -> None:
        diagnostics = {
            "features_file": str(self.paths.features_file),
            "regimes_file": str(self.paths.regimes_file),
            "features_columns": list(features_df.columns),
            "regime_columns": list(regime_df.columns),
            "merged_columns": list(merged_df.columns),
            "long_history_numeric_feature_candidates": LONG_HISTORY_NUMERIC_FEATURE_CANDIDATES,
            "optional_feature_candidates": LONG_HISTORY_OPTIONAL_FEATURES,
            "target_candidates": LONG_HISTORY_TARGET_CANDIDATES,
        }
        diagnostics_path = self.paths.evaluation_dir / "phase6_column_diagnostics.json"
        with open(diagnostics_path, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2)
        self.timer.log("DIAG", f"Saved column diagnostics to {diagnostics_path}")

    def _apply_aliases(self, df: pd.DataFrame) -> pd.DataFrame:
        for canonical_name, aliases in COLUMN_ALIASES.items():
            if canonical_name in df.columns:
                continue
            for alias in aliases:
                if alias in df.columns:
                    df = df.rename(columns={alias: canonical_name})
                    break
        return df

    def _ensure_date_column(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        date_col = None
        for candidate in DATE_COLUMN_CANDIDATES:
            if candidate in df.columns:
                date_col = candidate
                break

        if date_col is None:
            raise ValueError(f"Could not find date column in {source_name} dataset")

        if date_col != "date":
            df = df.rename(columns={date_col: "date"})

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).copy()
        return df

    def _build_target(self, df: pd.DataFrame, asset_name: str, base_col: str) -> Tuple[pd.DataFrame, str]:
        if base_col not in df.columns:
            raise ValueError(f"Cannot build target for {asset_name}. Missing base column: {base_col}")

        self.timer.log("TARGET", f"Building next-month target for {asset_name.upper()} from {base_col}")
        target_col = f"{asset_name}_target"
        df[target_col] = df[base_col].shift(-1)

        required_columns = ["date"] + self.active_numeric_features + self.active_categorical_features + [target_col]
        df = df[required_columns].copy()
        df = df.dropna(subset=[target_col]).reset_index(drop=True)

        self.timer.log("TARGET", f"{asset_name.upper()} dataset prepared with {len(df)} rows after target shift")
        return df, target_col

    def _train_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.timer.log("SPLIT", "Applying time-based train/test split")
        train_df = df[df["date"] <= self.train_end_date].copy()
        test_df = df[df["date"] >= self.test_start_date].copy()

        if train_df.empty or test_df.empty:
            raise ValueError(
                "Train/test split produced an empty partition. Check date range and dataset coverage."
            )

        self.timer.log("SPLIT", f"Train rows={len(train_df)}, Test rows={len(test_df)}")
        return train_df, test_df

    def _make_preprocessor(self) -> ColumnTransformer:
        transformers = []

        if self.active_numeric_features:
            numeric_pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("numeric", numeric_pipe, self.active_numeric_features))

        if self.active_categorical_features:
            categorical_pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            )
            transformers.append(("categorical", categorical_pipe, self.active_categorical_features))

        return ColumnTransformer(transformers=transformers, remainder="drop")

    def _get_feature_names_from_preprocessor(self, preprocessor: ColumnTransformer) -> List[str]:
        feature_names: List[str] = []

        if "numeric" in preprocessor.named_transformers_:
            feature_names.extend(self.active_numeric_features)

        if "categorical" in preprocessor.named_transformers_:
            cat_pipe = preprocessor.named_transformers_["categorical"]
            onehot = cat_pipe.named_steps["onehot"]
            encoded_names = list(onehot.get_feature_names_out(self.active_categorical_features))
            feature_names.extend(encoded_names)

        return feature_names

    def _fit_elastic_net(
        self,
        asset_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_columns: List[str],
    ) -> Dict[str, Any]:
        self.timer.log("ELASTIC", f"{asset_name.upper()} training started")

        preprocessor = self._make_preprocessor()

        train_progress = ProgressTracker(self.timer, f"ELASTIC-{asset_name.upper()}", total_steps=4)
        train_progress.update("Fitting preprocessing")
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        feature_names = self._get_feature_names_from_preprocessor(preprocessor)

        train_progress.update("Running ElasticNetCV fit")
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            alphas=np.logspace(-4, 1, 20),
            cv=5,
            max_iter=10000,
            random_state=self.random_state,
        )
        model.fit(X_train_processed, y_train)

        train_progress.update("Generating train and test predictions")
        train_pred = model.predict(X_train_processed)
        test_pred = model.predict(X_test_processed)

        train_progress.update("Computing validation and test metrics")
        metrics = self._calculate_metrics(train_true=y_train, train_pred=train_pred, test_true=y_test, test_pred=test_pred)

        model_path = self.paths.phase6_model_dir / f"elastic_net_{asset_name}.pkl"
        preprocessor_path = self.paths.phase6_model_dir / f"elastic_net_{asset_name}_preprocessor.pkl"
        feature_map_path = self.paths.phase6_model_dir / f"elastic_net_{asset_name}_feature_columns.json"

        joblib.dump(model, model_path)
        joblib.dump(preprocessor, preprocessor_path)
        with open(feature_map_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "raw_feature_columns": feature_columns,
                    "transformed_feature_columns": feature_names,
                },
                f,
                indent=2,
            )

        prediction_path = self._save_predictions(
            asset_name=asset_name,
            model_name="elastic_net",
            train_df=train_df,
            test_df=test_df,
            train_true=y_train,
            train_pred=train_pred,
            test_true=y_test,
            test_pred=test_pred,
        )

        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": model.coef_,
                "absolute_coefficient": np.abs(model.coef_),
            }
        ).sort_values("absolute_coefficient", ascending=False)

        importance_path = self.paths.feature_importance_dir / f"elastic_net_{asset_name}_importance.csv"
        importance_df.to_csv(importance_path, index=False)

        cv_results = {
            "alpha_selected": float(model.alpha_),
            "l1_ratio_selected": float(model.l1_ratio_),
            "intercept": float(model.intercept_),
            "n_iter": int(model.n_iter_),
            "mse_path_mean_by_alpha": np.mean(model.mse_path_, axis=2).tolist(),
            "alphas_tested": model.alphas_.tolist(),
        }
        cv_results_path = self.paths.evaluation_dir / f"elastic_net_{asset_name}_cv_results.json"
        with open(cv_results_path, "w", encoding="utf-8") as f:
            json.dump(cv_results, f, indent=2)

        self.timer.log("ELASTIC", f"{asset_name.upper()} model saved to {model_path}")
        return {
            "metrics": metrics,
            "model_path": str(model_path),
            "preprocessor_path": str(preprocessor_path),
            "prediction_path": str(prediction_path),
            "cv_results_path": str(cv_results_path),
            "importance_path": str(importance_path),
        }

    def _fit_xgboost(
        self,
        asset_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_columns: List[str],
    ) -> Dict[str, Any]:
        self.timer.log("XGBOOST", f"{asset_name.upper()} training started")

        preprocessor = self._make_preprocessor()

        train_progress = ProgressTracker(self.timer, f"XGB-{asset_name.upper()}", total_steps=5)
        train_progress.update("Fitting preprocessing")
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        if hasattr(X_train_processed, "toarray"):
            X_train_processed = X_train_processed.toarray()
        if hasattr(X_test_processed, "toarray"):
            X_test_processed = X_test_processed.toarray()

        feature_names = self._get_feature_names_from_preprocessor(preprocessor)

        train_progress.update("Training XGBoost model")
        model = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            objective="reg:squarederror",
            eval_metric="rmse",
        )
        eval_set = [(X_train_processed, y_train), (X_test_processed, y_test)]
        model.fit(X_train_processed, y_train, eval_set=eval_set, verbose=False)

        train_progress.update("Generating train predictions")
        train_pred = model.predict(X_train_processed)

        train_progress.update("Generating test predictions")
        test_pred = model.predict(X_test_processed)

        train_progress.update("Computing validation and test metrics")
        metrics = self._calculate_metrics(train_true=y_train, train_pred=train_pred, test_true=y_test, test_pred=test_pred)

        model_path = self.paths.phase6_model_dir / f"xgb_{asset_name}.pkl"
        preprocessor_path = self.paths.phase6_model_dir / f"xgb_{asset_name}_preprocessor.pkl"
        feature_map_path = self.paths.phase6_model_dir / f"xgb_{asset_name}_feature_columns.json"

        joblib.dump(model, model_path)
        joblib.dump(preprocessor, preprocessor_path)
        with open(feature_map_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "raw_feature_columns": feature_columns,
                    "transformed_feature_columns": feature_names,
                },
                f,
                indent=2,
            )

        prediction_path = self._save_predictions(
            asset_name=asset_name,
            model_name="xgboost",
            train_df=train_df,
            test_df=test_df,
            train_true=y_train,
            train_pred=train_pred,
            test_true=y_test,
            test_pred=test_pred,
        )

        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        importance_path = self.paths.feature_importance_dir / f"xgb_{asset_name}_importance.csv"
        importance_df.to_csv(importance_path, index=False)

        training_trace = {
            "evals_result": model.evals_result(),
            "best_iteration": int(getattr(model, "best_iteration", model.n_estimators)),
            "n_estimators": int(model.n_estimators),
        }
        training_trace_path = self.paths.evaluation_dir / f"xgb_{asset_name}_training_trace.json"
        with open(training_trace_path, "w", encoding="utf-8") as f:
            json.dump(training_trace, f, indent=2)

        self.timer.log("XGBOOST", f"{asset_name.upper()} model saved to {model_path}")
        return {
            "metrics": metrics,
            "model_path": str(model_path),
            "preprocessor_path": str(preprocessor_path),
            "prediction_path": str(prediction_path),
            "training_trace_path": str(training_trace_path),
            "importance_path": str(importance_path),
        }

    def _calculate_metrics(
        self,
        train_true: pd.Series,
        train_pred: np.ndarray,
        test_true: pd.Series,
        test_pred: np.ndarray,
    ) -> Dict[str, float]:
        metric_progress = ProgressTracker(self.timer, "METRICS", total_steps=2)

        metric_progress.update("Calculating train metrics")
        train_rmse = float(np.sqrt(mean_squared_error(train_true, train_pred)))
        train_mae = float(mean_absolute_error(train_true, train_pred))
        train_r2 = float(r2_score(train_true, train_pred))
        train_directional_accuracy = float(np.mean(np.sign(train_pred) == np.sign(train_true)))

        metric_progress.update("Calculating test metrics")
        test_rmse = float(np.sqrt(mean_squared_error(test_true, test_pred)))
        test_mae = float(mean_absolute_error(test_true, test_pred))
        test_r2 = float(r2_score(test_true, test_pred))
        test_directional_accuracy = float(np.mean(np.sign(test_pred) == np.sign(test_true)))

        return {
            "train_r2": train_r2,
            "train_rmse": train_rmse,
            "train_mae": train_mae,
            "train_directional_accuracy": train_directional_accuracy,
            "test_r2": test_r2,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "test_directional_accuracy": test_directional_accuracy,
        }

    def _save_predictions(
        self,
        asset_name: str,
        model_name: str,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_true: pd.Series,
        train_pred: np.ndarray,
        test_true: pd.Series,
        test_pred: np.ndarray,
    ) -> Path:
        save_progress = ProgressTracker(self.timer, "PREDICTIONS", total_steps=2)

        save_progress.update(f"Saving train predictions for {asset_name.upper()} {model_name}")
        train_out = pd.DataFrame(
            {
                "date": train_df["date"].values,
                "dataset": "train",
                "actual": train_true.values,
                "predicted": train_pred,
                "residual": train_true.values - train_pred,
            }
        )

        save_progress.update(f"Saving test predictions for {asset_name.upper()} {model_name}")
        test_out = pd.DataFrame(
            {
                "date": test_df["date"].values,
                "dataset": "test",
                "actual": test_true.values,
                "predicted": test_pred,
                "residual": test_true.values - test_pred,
            }
        )

        out_df = pd.concat([train_out, test_out], axis=0, ignore_index=True)
        prediction_path = self.paths.predictions_dir / f"{model_name}_{asset_name}_predictions.csv"
        out_df.to_csv(prediction_path, index=False)
        return prediction_path