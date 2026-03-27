"""
Phase 8 – Explainability Engine (SHAP + ElasticNet Coefficients)

Consumes outputs from Phase 6 (trained models + preprocessors + test data).
No model retraining. No feature engineering.

Produces:
  data/explainability/
    shap_global_xgb_{asset}.csv          - mean |SHAP| per feature (XGBoost)
    shap_global_en_{asset}.csv           - mean |SHAP| per feature (ElasticNet)
    shap_values_xgb_{asset}.csv          - full SHAP matrix, test set (XGBoost)
    shap_values_en_{asset}.csv           - full SHAP matrix, test set (ElasticNet)
    elastic_net_coefficients_{asset}.csv - actual EN coefficients from pkl model
    feature_importance_comparison.csv    - unified table: XGB gain | SHAP | EN coeff
    latest_prediction_explanation.json   - per-asset local explanation, most recent row
    explainability_report.json           - full summary
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import shap

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ASSETS = ["spx", "ndx", "gold", "btc"]
MONTHS_PER_YEAR = 12


class _Timer:
    def __init__(self) -> None:
        self._t0 = time.time()

    def log(self, tag: str, msg: str) -> None:
        elapsed = time.time() - self._t0
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        logger.info(f"[{h:02d}:{m:02d}:{s:02d}] [{tag}] {msg}")


class ExplainabilityEngine:
    """
    Phase 8: SHAP-based explainability + ElasticNet coefficient analysis.

    Loads Phase 6 pkl models + preprocessors, computes SHAP values on the
    out-of-sample test set, and extracts EN coefficients from the model objects.
    """

    def __init__(self, backend_root: str | Path) -> None:
        self.root   = Path(backend_root)
        self.timer  = _Timer()

        self.models_dir       = self.root / "models" / "phase6"
        self.cache_dir        = self.models_dir / "training_cache"
        self.output_dir       = self.root / "data" / "explainability"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> dict:
        self.timer.log("START", "Phase 8 Explainability started")

        report: dict = {"assets": {}, "cross_asset_summary": {}}
        comparison_rows: List[dict] = []

        for asset in ASSETS:
            self.timer.log(asset.upper(), f"Processing {asset.upper()}")
            result = self._process_asset(asset)
            report["assets"][asset] = result
            comparison_rows.extend(result.pop("_comparison_rows"))

        # Cross-asset feature importance comparison
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_df.to_csv(self.output_dir / "feature_importance_comparison.csv", index=False)
        self.timer.log("COMPARE", f"Feature importance comparison saved ({len(comparison_df)} rows)")

        # Cross-asset summary: which features matter most on average
        report["cross_asset_summary"] = self._cross_asset_summary(comparison_df)

        # Write full report
        with open(self.output_dir / "explainability_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        self.timer.log("END", "Phase 8 completed successfully")
        return report

    # ── Per-asset pipeline ────────────────────────────────────────────────────

    def _process_asset(self, asset: str) -> dict:
        # Load artifacts
        feature_cols   = self._load_feature_columns(asset)
        test_df        = self._load_test_snapshot(asset)
        X_test_raw, y_test = self._split_xy(test_df, asset)

        # ── ElasticNet ───────────────────────────────────────────────────────
        en_model, en_pre = self._load_model(asset, "elastic_net")
        X_test_en        = en_pre.transform(X_test_raw)
        transformed_cols = self._get_transformed_columns(asset)

        en_coeffs_df     = self._extract_en_coefficients(en_model, transformed_cols, asset)
        shap_en_df, shap_en_global = self._compute_shap_linear(
            en_model, en_pre, X_test_raw, transformed_cols, asset
        )

        # ── XGBoost ──────────────────────────────────────────────────────────
        xgb_model, xgb_pre = self._load_model(asset, "xgboost")
        X_test_xgb         = xgb_pre.transform(X_test_raw)

        shap_xgb_df, shap_xgb_global = self._compute_shap_tree(
            xgb_model, X_test_xgb, transformed_cols, asset
        )

        xgb_gain_df = self._load_xgb_gain_importance(asset)

        # ── Latest-prediction local explanation ──────────────────────────────
        local_explanation = self._explain_latest_prediction(
            asset, X_test_raw, y_test,
            en_model, en_pre, shap_en_df,
            xgb_model, xgb_pre, shap_xgb_df,
            transformed_cols,
        )

        # ── Comparison rows for cross-asset table ────────────────────────────
        comparison_rows = self._build_comparison_rows(
            asset, transformed_cols, shap_xgb_global, shap_en_global, en_coeffs_df, xgb_gain_df
        )

        self.timer.log(asset.upper(), "Done")

        return {
            "en_active_features":  int((en_coeffs_df["coefficient"] != 0).sum()),
            "en_total_features":   len(en_coeffs_df),
            "xgb_top_feature":     shap_xgb_global.iloc[0]["feature"] if not shap_xgb_global.empty else None,
            "en_top_feature":      shap_en_global.iloc[0]["feature"] if not shap_en_global.empty and shap_en_global.iloc[0]["mean_abs_shap"] > 0 else "all_zero",
            "latest_explanation":  local_explanation,
            "_comparison_rows":    comparison_rows,
        }

    # ── SHAP: XGBoost (TreeExplainer) ────────────────────────────────────────

    def _compute_shap_tree(
        self,
        model,
        X_transformed: np.ndarray,
        feature_names: List[str],
        asset: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)   # (n_test, n_features)

        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        shap_df.to_csv(self.output_dir / f"shap_values_xgb_{asset}.csv", index=False)

        global_df = pd.DataFrame({
            "feature":       feature_names,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        global_df.to_csv(self.output_dir / f"shap_global_xgb_{asset}.csv", index=False)
        self.timer.log(asset.upper(), f"XGBoost SHAP computed — top feature: {global_df.iloc[0]['feature']} ({global_df.iloc[0]['mean_abs_shap']:.5f})")

        return shap_df, global_df

    # ── SHAP: ElasticNet (LinearExplainer) ───────────────────────────────────

    def _compute_shap_linear(
        self,
        model,
        preprocessor,
        X_raw: pd.DataFrame,
        feature_names: List[str],
        asset: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X_transformed = preprocessor.transform(X_raw)

        # Background = training data mean (masker for linear models)
        explainer   = shap.LinearExplainer(model, X_transformed, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_transformed)   # (n_test, n_features)

        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        shap_df.to_csv(self.output_dir / f"shap_values_en_{asset}.csv", index=False)

        global_df = pd.DataFrame({
            "feature":       feature_names,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        global_df.to_csv(self.output_dir / f"shap_global_en_{asset}.csv", index=False)
        top = global_df.iloc[0]
        self.timer.log(asset.upper(), f"ElasticNet SHAP computed — top feature: {top['feature']} ({top['mean_abs_shap']:.5f})")

        return shap_df, global_df

    # ── ElasticNet coefficients ───────────────────────────────────────────────

    def _extract_en_coefficients(
        self,
        model,
        feature_names: List[str],
        asset: str,
    ) -> pd.DataFrame:
        """
        Extract actual coefficients from the ElasticNetCV pkl object.
        The saved CSV has all-zeros due to heavy regularization — the pkl has the truth.
        """
        coef = np.asarray(model.coef_).flatten()
        intercept = float(model.intercept_) if hasattr(model, "intercept_") else 0.0

        df = pd.DataFrame({
            "feature":              feature_names,
            "coefficient":          coef,
            "abs_coefficient":      np.abs(coef),
            "direction":            np.where(coef > 0, "positive", np.where(coef < 0, "negative", "zero")),
        }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

        df.attrs["intercept"] = intercept

        # Add intercept row at end
        intercept_row = pd.DataFrame([{
            "feature":         "__intercept__",
            "coefficient":     intercept,
            "abs_coefficient": abs(intercept),
            "direction":       "positive" if intercept >= 0 else "negative",
        }])
        df = pd.concat([df, intercept_row], ignore_index=True)

        df.to_csv(self.output_dir / f"elastic_net_coefficients_{asset}.csv", index=False)
        n_active = int((np.abs(coef) > 0).sum())
        self.timer.log(asset.upper(), f"EN coefficients extracted — {n_active}/{len(coef)} active, intercept={intercept:.6f}")

        return df

    # ── Latest-prediction local explanation ──────────────────────────────────

    def _explain_latest_prediction(
        self,
        asset: str,
        X_test_raw: pd.DataFrame,
        y_test: pd.Series,
        en_model, en_pre,
        shap_en_df: pd.DataFrame,
        xgb_model, xgb_pre,
        shap_xgb_df: pd.DataFrame,
        feature_names: List[str],
    ) -> dict:
        """
        Explain the most recent (last row of test set) prediction for each model.
        Returns top-5 drivers from each model.
        """
        last_idx = -1   # most recent test observation

        # Raw feature values
        raw_values = X_test_raw.iloc[last_idx].to_dict()

        # EN prediction + SHAP
        X_last_en      = en_pre.transform(X_test_raw.iloc[[last_idx]])
        en_pred         = float(en_model.predict(X_last_en)[0])
        en_shap_last    = shap_en_df.iloc[last_idx].to_dict()
        en_top5         = sorted(en_shap_last.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

        # XGB prediction + SHAP
        X_last_xgb     = xgb_pre.transform(X_test_raw.iloc[[last_idx]])
        xgb_pred        = float(xgb_model.predict(X_last_xgb)[0])
        xgb_shap_last   = shap_xgb_df.iloc[last_idx].to_dict()
        xgb_top5        = sorted(xgb_shap_last.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

        actual = float(y_test.iloc[last_idx])

        explanation = {
            "asset":        asset,
            "actual_return": round(actual, 6),
            "elastic_net": {
                "prediction": round(en_pred, 6),
                "error":      round(en_pred - actual, 6),
                "top_drivers": [
                    {"feature": f, "shap_value": round(v, 6), "direction": "bullish" if v > 0 else "bearish"}
                    for f, v in en_top5
                ],
            },
            "xgboost": {
                "prediction": round(xgb_pred, 6),
                "error":      round(xgb_pred - actual, 6),
                "top_drivers": [
                    {"feature": f, "shap_value": round(v, 6), "direction": "bullish" if v > 0 else "bearish"}
                    for f, v in xgb_top5
                ],
            },
            "raw_feature_snapshot": {k: round(float(v), 6) if isinstance(v, (int, float, np.floating)) else str(v)
                                     for k, v in raw_values.items()},
        }

        return explanation

    # ── Cross-asset summary ───────────────────────────────────────────────────

    def _cross_asset_summary(self, comparison_df: pd.DataFrame) -> dict:
        """Average XGB SHAP importance across all assets — universal drivers."""
        if "xgb_mean_abs_shap" not in comparison_df.columns:
            return {}

        avg = (
            comparison_df.groupby("feature")["xgb_mean_abs_shap"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        avg.columns = ["feature", "avg_xgb_shap_across_assets"]
        avg.to_csv(self.output_dir / "cross_asset_shap_importance.csv", index=False)

        return {
            "top5_universal_drivers": avg.head(5).to_dict(orient="records"),
            "note": "Average XGBoost SHAP importance across SPX, NDX, Gold",
        }

    # ── Comparison table builder ──────────────────────────────────────────────

    def _build_comparison_rows(
        self,
        asset: str,
        feature_names: List[str],
        shap_xgb_global: pd.DataFrame,
        shap_en_global: pd.DataFrame,
        en_coeffs_df: pd.DataFrame,
        xgb_gain_df: pd.DataFrame,
    ) -> List[dict]:
        # Build lookup dicts
        xgb_shap_map = dict(zip(shap_xgb_global["feature"], shap_xgb_global["mean_abs_shap"]))
        en_shap_map  = dict(zip(shap_en_global["feature"],  shap_en_global["mean_abs_shap"]))
        en_coeff_map = {}
        for _, row in en_coeffs_df.iterrows():
            if row["feature"] != "__intercept__":
                en_coeff_map[row["feature"]] = row["coefficient"]

        xgb_gain_map = {}
        if xgb_gain_df is not None:
            xgb_gain_map = dict(zip(xgb_gain_df["feature"], xgb_gain_df["importance"]))

        rows = []
        for feat in feature_names:
            rows.append({
                "asset":              asset,
                "feature":            feat,
                "xgb_gain":           round(xgb_gain_map.get(feat, 0.0), 6),
                "xgb_mean_abs_shap":  round(xgb_shap_map.get(feat, 0.0), 6),
                "en_mean_abs_shap":   round(en_shap_map.get(feat, 0.0), 6),
                "en_coefficient":     round(en_coeff_map.get(feat, 0.0), 6),
                "en_direction":       ("positive" if en_coeff_map.get(feat, 0.0) > 0
                                       else "negative" if en_coeff_map.get(feat, 0.0) < 0
                                       else "zero"),
            })
        return rows

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_model(self, asset: str, model_type: str):
        if model_type == "elastic_net":
            model_path = self.models_dir / f"elastic_net_{asset}.pkl"
            pre_path   = self.models_dir / f"elastic_net_{asset}_preprocessor.pkl"
        else:
            model_path = self.models_dir / f"xgb_{asset}.pkl"
            pre_path   = self.models_dir / f"xgb_{asset}_preprocessor.pkl"

        model = joblib.load(model_path)
        pre   = joblib.load(pre_path)
        return model, pre

    def _load_test_snapshot(self, asset: str) -> pd.DataFrame:
        path = self.cache_dir / f"test_snapshot_{asset}.csv"
        df   = pd.read_csv(path, parse_dates=["date"])
        return df.sort_values("date").reset_index(drop=True)

    def _load_feature_columns(self, asset: str) -> List[str]:
        path = self.models_dir / f"elastic_net_{asset}_feature_columns.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data["raw_feature_columns"]

    def _get_transformed_columns(self, asset: str) -> List[str]:
        path = self.models_dir / f"elastic_net_{asset}_feature_columns.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data["transformed_feature_columns"]

    def _split_xy(self, df: pd.DataFrame, asset: str) -> Tuple[pd.DataFrame, pd.Series]:
        target_col = f"{asset}_target"
        raw_cols   = self._load_feature_columns(asset)

        # Drop non-feature columns
        drop_cols  = [c for c in df.columns if c not in raw_cols and c != target_col]
        X = df[raw_cols].copy()
        y = df[target_col].copy()
        return X, y

    def _load_xgb_gain_importance(self, asset: str) -> Optional[pd.DataFrame]:
        path = self.models_dir / "feature_importance" / f"xgb_{asset}_importance.csv"
        if path.exists():
            return pd.read_csv(path)
        return None
