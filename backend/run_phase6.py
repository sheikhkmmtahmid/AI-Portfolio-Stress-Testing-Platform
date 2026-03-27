from __future__ import annotations

from pathlib import Path

from services.asset_models import AssetSensitivityTrainer


WINDOWS_PROJECT_DATA_ROOT = r"D:\ML-Powered Portfolio Stress Testing Tool with Macroeconomic Scenario Analysis and Market Risk Modelling\Datasets"


def _print_result(backend_root: Path, result: dict, label: str) -> None:
    print(f"\n{label} completed.", flush=True)
    print(f"Assets trained: {', '.join(result['metrics'].keys()).upper()}", flush=True)
    skipped = result["metadata"].get("targets_skipped", {})
    for asset, reason in skipped.items():
        print(f"  Skipped {asset.upper()}: {reason}", flush=True)


def main() -> None:
    backend_root = Path(__file__).resolve().parent

    # ── Run 1: SPX / NDX / Gold  (full history from 2003)
    print("\n[Phase 6 - Run 1] SPX / NDX / Gold  (features_monthly_full_history.csv)", flush=True)
    trainer_main = AssetSensitivityTrainer(project_root=backend_root)
    result_main  = trainer_main.run()
    _print_result(backend_root, result_main, "Phase 6 Run 1")

    # ── Run 2: BTC  (short history from 2010, separate feature file)
    btc_features = backend_root / "data" / "features" / "features_monthly_btc.csv"
    if btc_features.exists():
        print("\n[Phase 6 - Run 2] BTC  (features_monthly_btc.csv)", flush=True)
        trainer_btc = AssetSensitivityTrainer(
            project_root=backend_root,
            features_file_override=btc_features,
            target_assets_filter=["btc"],   # only train BTC — don't overwrite SPX/NDX/Gold
        )
        result_btc = trainer_btc.run()
        _print_result(backend_root, result_btc, "Phase 6 Run 2")
    else:
        print(f"\n[WARN] BTC feature file not found: {btc_features}", flush=True)
        print("       Run Phase 3 first to generate it.", flush=True)

    print(f"\nBackend root: {backend_root}", flush=True)
    print(f"Models directory: {backend_root / 'models' / 'phase6'}", flush=True)


if __name__ == "__main__":
    main()