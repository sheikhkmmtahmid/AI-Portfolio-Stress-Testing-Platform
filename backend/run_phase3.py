from pathlib import Path

import pandas as pd

from services.feature_engineering import FeatureEngineer, FeatureEngineerConfig


def build_dataset(df_all: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    cols = [col for col in columns if col in df_all.columns]
    df = df_all[cols].replace([float("inf"), float("-inf")], pd.NA).dropna().copy()
    return df


def print_dataset_summary(name: str, path: Path, df: pd.DataFrame) -> None:
    print(f"[OK] {name} saved")
    print(f"Path: {path}")
    print(f"Shape: {df.shape}")
    print(f"Start: {df.index.min()}")
    print(f"End:   {df.index.max()}")


def main() -> None:
    config = FeatureEngineerConfig(
        input_path="data/processed/merged_monthly.csv",
        output_path_full="data/features/features_monthly.csv",
        output_path_long="data/features/features_monthly_full_history.csv",
    )

    output_dir = Path("data/features")
    output_dir.mkdir(parents=True, exist_ok=True)

    engineer = FeatureEngineer(config.input_path)
    engineer.run()

    df_all = engineer.df.copy()

    full_feature_cols = [
        "spx_return",
        "ndx_return",
        "ftse_return",
        "gold_return",
        "gold_return_3m",
        "gold_vol_3m",
        "gold_vol_6m",
        "gold_drawdown",
        "gold_max_dd_6m",
        "btc_return",
        "eurusd_return",
        "gbpusd_return",
        "spx_vol_3m",
        "spx_vol_6m",
        "ndx_vol_3m",
        "ndx_vol_6m",
        "vix_level",
        "us2y_yield",
        "us10y_yield",
        "yield_spread",
        "us_cpi_yoy",
        "uk_cpi_yoy",
        "high_yield_spread",
        "vix_spike",
        "spx_drawdown",
        "spx_max_dd_6m",
        "ndx_drawdown",
        "ndx_max_dd_6m",
        # ECB
        "ecb_level",
        "ecb_yoy",
        # New: Fed Funds, TIPS, Breakeven, DXY, QQQ
        "fed_funds_level",
        "fed_funds_change_1m",
        "tips_10y_level",
        "tips_10y_change_1m",
        "breakeven_10y_level",
        "breakeven_10y_change_1m",
        "real_yield_tips",
        "real_yield_tips_change_1m",
        "dxy_level",
        "dxy_return",
        "dxy_return_3m",
        "qqq_return",
    ]

    long_history_cols = [
        "spx_return",
        "ndx_return",
        "gold_return",
        "gold_return_3m",
        "gold_vol_3m",
        "gold_vol_6m",
        "gold_drawdown",
        "gold_max_dd_6m",
        "eurusd_return",
        "gbpusd_return",
        "spx_vol_3m",
        "spx_vol_6m",
        "ndx_vol_3m",
        "ndx_vol_6m",
        "vix_level",
        "us2y_yield",
        "us10y_yield",
        "yield_spread",
        "us_cpi_yoy",
        "high_yield_spread",
        "vix_spike",
        "spx_drawdown",
        "spx_max_dd_6m",
        "ndx_drawdown",
        "ndx_max_dd_6m",
        # ECB
        "ecb_level",
        "ecb_yoy",
        # New: Fed Funds, TIPS, Breakeven, DXY, QQQ
        "fed_funds_level",
        "fed_funds_change_1m",
        "tips_10y_level",
        "tips_10y_change_1m",
        "breakeven_10y_level",
        "breakeven_10y_change_1m",
        "real_yield_tips",
        "real_yield_tips_change_1m",
        "dxy_level",
        "dxy_return",
        "dxy_return_3m",
        "qqq_return",
    ]

    # BTC uses a SEPARATE short-history file so it doesn't truncate the long-history
    # dataset for SPX/NDX/Gold (BTC data starts 2010, would lose 7 years otherwise)
    btc_cols = [
        "spx_return", "ndx_return", "gold_return",
        "eurusd_return", "gbpusd_return",
        "spx_vol_3m", "ndx_vol_3m",
        "vix_level", "us2y_yield", "us10y_yield", "yield_spread",
        "us_cpi_yoy", "high_yield_spread",
        "vix_spike", "spx_drawdown", "ndx_drawdown",
        "ecb_level", "ecb_yoy",
        "dxy_return", "qqq_return",
        "btc_return",   # target — NaN before 2010, dropna trims automatically
    ]

    full_df = build_dataset(df_all, full_feature_cols)
    long_df = build_dataset(df_all, long_history_cols)
    btc_df  = build_dataset(df_all, btc_cols)

    full_output = Path(config.output_path_full)
    long_output = Path(config.output_path_long)
    btc_output  = Path("data/features/features_monthly_btc.csv")

    full_df.to_csv(full_output)
    long_df.to_csv(long_output)
    btc_df.to_csv(btc_output)

    print_dataset_summary("Full-feature dataset", full_output, full_df)
    print()
    print_dataset_summary("Long-history model dataset", long_output, long_df)
    print()
    print_dataset_summary("BTC short-history dataset", btc_output, btc_df)

    print("\nColumns in full-feature dataset:")
    for col in full_df.columns:
        print(f" - {col}")

    print("\nColumns in long-history model dataset:")
    for col in long_df.columns:
        print(f" - {col}")


if __name__ == "__main__":
    main()
