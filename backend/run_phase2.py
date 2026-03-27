from pathlib import Path
import sys

backend_root = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_root))

from services.data_ingestion import DataIngestionService


def main() -> None:
    service = DataIngestionService(backend_root=backend_root)
    market_df, macro_df, merged_df = service.run()

    print("\nPhase 2 complete.")
    print(f"Market shape: {market_df.shape}")
    print(f"Macro shape: {macro_df.shape}")
    print(f"Merged monthly shape: {merged_df.shape}")

    print("\nDate ranges:")
    print(f"Market: {market_df['date'].min()} -> {market_df['date'].max()}")
    print(f"Macro: {macro_df['date'].min()} -> {macro_df['date'].max()}")
    print(f"Merged: {merged_df['date'].min()} -> {merged_df['date'].max()}")


if __name__ == "__main__":
    main()