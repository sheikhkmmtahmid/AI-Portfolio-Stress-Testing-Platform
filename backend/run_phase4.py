from __future__ import annotations

import logging
from pathlib import Path

from services.scenario_engine import ScenarioEngine, configure_logging


def main() -> None:
    configure_logging(logging.INFO)

    backend_dir = Path(__file__).resolve().parent
    input_path = backend_dir / "data" / "features" / "features_monthly_full_history.csv"
    output_dir = backend_dir / "data" / "scenarios"

    engine = ScenarioEngine(
        input_path=input_path,
        output_dir=output_dir,
    )

    scenario_dataset, scenario_summary = engine.run()

    print("\n✅ Phase 4 completed successfully.")
    print(f"Scenario dataset shape: {scenario_dataset.shape}")
    print(f"Scenario summary shape: {scenario_summary.shape}")
    print(f"Scenario dataset saved to: {output_dir / 'scenario_dataset.csv'}")
    print(f"Scenario summary saved to: {output_dir / 'scenario_summary.csv'}")


if __name__ == "__main__":
    main()