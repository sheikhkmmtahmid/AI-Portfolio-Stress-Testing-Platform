from __future__ import annotations

from pathlib import Path

from services.asset_models_phase6_1 import Phase61Refiner


WINDOWS_PROJECT_DATA_ROOT = r"D:\ML-Powered Portfolio Stress Testing Tool with Macroeconomic Scenario Analysis and Market Risk Modelling\Datasets"


def _print_result(summary: dict, label: str) -> None:
    print(f"\n{label} completed.", flush=True)
    for asset, details in summary["assets"].items():
        print(f"\n{asset.upper()}:", flush=True)
        print(f"  Elastic best params: {details['elastic_net_best_params']}", flush=True)
        print(f"  XGBoost best params: {details['xgboost_best_params']}", flush=True)
        print(f"  Rolling summary: {details['rolling_summary']}", flush=True)


def main() -> None:
    backend_root = Path(__file__).resolve().parent

    # ── Run 1: SPX / NDX / Gold  (full history from 2003)
    print("\n[Phase 6.1 - Run 1] SPX / NDX / Gold  (features_monthly_full_history.csv)", flush=True)
    runner_main = Phase61Refiner(
        backend_root=backend_root,
        target_assets_filter=["spx", "ndx", "gold"],
    )
    summary_main = runner_main.run()
    _print_result(summary_main, "Phase 6.1 Run 1")

    # ── Run 2: BTC  (short history from 2010, separate feature file)
    btc_features = backend_root / "data" / "features" / "features_monthly_btc.csv"
    if btc_features.exists():
        print("\n[Phase 6.1 - Run 2] BTC  (features_monthly_btc.csv)", flush=True)
        runner_btc = Phase61Refiner(
            backend_root=backend_root,
            features_file_override=btc_features,
            target_assets_filter=["btc"],
        )
        summary_btc = runner_btc.run()
        _print_result(summary_btc, "Phase 6.1 Run 2")
    else:
        print(f"\n[WARN] BTC feature file not found: {btc_features}", flush=True)
        print("       Run Phase 3 first to generate it.", flush=True)

    print(f"\nBackend root: {backend_root}", flush=True)
    print(f"Reference raw data root: {WINDOWS_PROJECT_DATA_ROOT}", flush=True)
    print(f"Output directory: {backend_root / 'models' / 'phase6_1'}", flush=True)


if __name__ == "__main__":
    main()
