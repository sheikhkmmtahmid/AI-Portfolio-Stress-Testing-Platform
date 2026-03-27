"""
Phase 8 runner — Explainability (SHAP + ElasticNet Coefficients).

Execution flow:
  1. Load Phase 6 pkl models + preprocessors + test snapshots
  2. Compute SHAP values (TreeExplainer for XGBoost, LinearExplainer for ElasticNet)
  3. Extract ElasticNet coefficients from model objects
  4. Build cross-asset feature importance comparison
  5. Generate latest-prediction local explanations
  6. Save all outputs to data/explainability/
"""

import sys
from pathlib import Path

backend_root = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_root))

from services.explainability_engine import ExplainabilityEngine

EXPLAINABILITY_DIR = backend_root / "data" / "explainability"
ASSETS = ["spx", "ndx", "gold", "btc"]


def _print_section(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print("=" * 65)


def main() -> None:
    print("\nPhase 8: Explainability — SHAP + ElasticNet Coefficients")
    print(f"Backend root: {backend_root}")

    engine = ExplainabilityEngine(backend_root=backend_root)
    report = engine.run()

    import pandas as pd
    import json

    # ── SHAP global importance ────────────────────────────────────────────────
    for asset in ASSETS:
        _print_section(f"XGBoost SHAP — Global Feature Importance  [{asset.upper()}]")
        path = EXPLAINABILITY_DIR / f"shap_global_xgb_{asset}.csv"
        df   = pd.read_csv(path)
        for _, row in df.head(10).iterrows():
            bar = "#" * int(row["mean_abs_shap"] * 600)
            print(f"  {row['feature']:<35} {row['mean_abs_shap']:.5f}  {bar}")

    # ── ElasticNet coefficients ───────────────────────────────────────────────
    _print_section("ElasticNet Coefficients (from pkl model, all assets)")
    for asset in ASSETS:
        path = EXPLAINABILITY_DIR / f"elastic_net_coefficients_{asset}.csv"
        df   = pd.read_csv(path)
        active = df[df["feature"] != "__intercept__"]
        active = active[active["coefficient"] != 0].sort_values("abs_coefficient", ascending=False)
        intercept_row = df[df["feature"] == "__intercept__"]
        intercept_val = intercept_row["coefficient"].iloc[0] if not intercept_row.empty else 0.0
        print(f"\n  {asset.upper()}  (intercept={intercept_val:+.6f},  {len(active)} non-zero features):")
        if active.empty:
            print("    [all coefficients shrunk to zero by regularization]")
        else:
            for _, row in active.head(10).iterrows():
                sign = "+" if row["coefficient"] > 0 else ""
                print(f"    {row['feature']:<35} {sign}{row['coefficient']:.6f}")

    # ── Latest prediction explanation ─────────────────────────────────────────
    _print_section("Latest Prediction — Local SHAP Explanation (most recent test row)")
    for asset in ASSETS:
        exp = report["assets"][asset]["latest_explanation"]
        print(f"\n  {asset.upper()}  actual={exp['actual_return']:+.4f}")
        for model_key in ["elastic_net", "xgboost"]:
            m = exp[model_key]
            print(f"    [{model_key:<12}]  pred={m['prediction']:+.4f}  error={m['error']:+.4f}")
            for driver in m["top_drivers"][:3]:
                arrow = "^" if driver["direction"] == "bullish" else "v"
                print(f"      {arrow} {driver['feature']:<30}  SHAP={driver['shap_value']:+.5f}")

    # ── Cross-asset comparison ────────────────────────────────────────────────
    _print_section("Feature Importance Comparison (XGB Gain | SHAP | EN Coeff)")
    path = EXPLAINABILITY_DIR / "feature_importance_comparison.csv"
    df   = pd.read_csv(path)
    for asset in ASSETS:
        asset_df = df[df["asset"] == asset].sort_values("xgb_mean_abs_shap", ascending=False).head(8)
        print(f"\n  {asset.upper()}  — top 8 by XGB SHAP:")
        print(f"  {'Feature':<35} {'XGB Gain':>10} {'XGB SHAP':>10} {'EN SHAP':>9} {'EN Coeff':>10} Dir")
        print(f"  {'-'*80}")
        for _, row in asset_df.iterrows():
            print(f"  {row['feature']:<35} {row['xgb_gain']:>10.5f} {row['xgb_mean_abs_shap']:>10.5f} "
                  f"{row['en_mean_abs_shap']:>9.5f} {row['en_coefficient']:>10.5f}  {row['en_direction']}")

    # ── Universal drivers ─────────────────────────────────────────────────────
    _print_section("Cross-Asset Universal Drivers (avg SHAP across SPX, NDX, Gold)")
    summary = report["cross_asset_summary"]
    for item in summary.get("top5_universal_drivers", []):
        bar = "#" * int(item["avg_xgb_shap_across_assets"] * 500)
        print(f"  {item['feature']:<35} {item['avg_xgb_shap_across_assets']:.5f}  {bar}")

    # ── Output files ──────────────────────────────────────────────────────────
    _print_section("Output Files")
    for f in sorted(EXPLAINABILITY_DIR.glob("*.csv")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:<55} {size_kb:6.1f} KB")
    for f in sorted(EXPLAINABILITY_DIR.glob("*.json")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:<55} {size_kb:6.1f} KB")

    print("\nPhase 8 complete.\n")


if __name__ == "__main__":
    main()
