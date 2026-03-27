from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class DataIngestionService:
    """
    Phase 2: Data ingestion and standardisation

    Responsibilities:
    - Load raw CSVs from market and macro folders
    - Standardise each dataset into a consistent schema
    - Clean dates and numeric values
    - Remove duplicates
    - Handle missing values
    - Align frequencies
    - Save:
        - backend/data/processed/market_clean.csv
        - backend/data/processed/macro_clean.csv
        - backend/data/processed/merged_monthly.csv
    """

    def __init__(self, backend_root: str | Path):
        self.backend_root = Path(backend_root)

        self.raw_market_dir = self.backend_root / "data" / "raw" / "market"
        self.raw_macro_dir = self.backend_root / "data" / "raw" / "macro"
        self.processed_dir = self.backend_root / "data" / "processed"

        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.market_files: Dict[str, str] = {
            "^spx_d.csv": "spx",
            "^ndx_d.csv": "ndx",
            "FTSE 100 Historical Results Price Data.csv": "ftse100",
            "xauusd_d.csv": "gold",
            "btcusd_d.csv": "bitcoin",
            "eurusd_d.csv": "eurusd",
            "gbpusd_d.csv": "gbpusd",
            # NEW: USD Index and NDX ETF proxy
            "dxy_d.csv": "dxy",
            "qqq_d.csv": "qqq",
        }

        self.macro_files: Dict[str, str] = {
            "DGS2.csv": "us2y_yield",
            "DGS10.csv": "us10y_yield",
            "CPIAUCSL.csv": "us_cpi",
            "cpi_uk_d.csv": "uk_cpi",
            "BAMLH0A0HYM2.csv": "us_hy_oas",
            "VIXCLS.csv": "vix",
            # NEW: Fed Funds rate, 10Y TIPS real yield, 10Y breakeven inflation
            "FEDFUNDS.csv": "fed_funds",
            "DFII10.csv": "tips_10y",
            "T10YIE.csv": "breakeven_10y",
        }

        self.min_valid_date = pd.Timestamp("2000-01-01")
        self.max_valid_date = pd.Timestamp("2035-12-31")

    # Public API

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        market_df = self.build_market_dataset()
        macro_df = self.build_macro_dataset()
        merged_monthly_df = self.build_merged_monthly_dataset(market_df, macro_df)

        market_output = self.processed_dir / "market_clean.csv"
        macro_output = self.processed_dir / "macro_clean.csv"
        merged_output = self.processed_dir / "merged_monthly.csv"

        market_df.to_csv(market_output, index=False)
        macro_df.to_csv(macro_output, index=False)
        merged_monthly_df.to_csv(merged_output, index=False)

        print(f"Saved: {market_output}")
        print(f"Saved: {macro_output}")
        print(f"Saved: {merged_output}")

        return market_df, macro_df, merged_monthly_df

    # Dataset builders

    def build_market_dataset(self) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []

        for filename, series_name in self.market_files.items():
            file_path = self.raw_market_dir / filename
            if not file_path.exists():
                print(f"[WARN] Missing market file: {file_path}")
                continue

            df = self._load_and_standardise_file(
                file_path=file_path,
                series_name=series_name,
                data_type="market",
            )
            if not df.empty:
                frames.append(df)

        if not frames:
            raise FileNotFoundError("No valid market files were loaded.")

        combined = self._merge_series_on_date(frames, freq="D")
        combined = self._finalise_dataframe(combined, date_freq="D")
        return combined

    def build_macro_dataset(self) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []

        for filename, series_name in self.macro_files.items():
            file_path = self.raw_macro_dir / filename
            if not file_path.exists():
                print(f"[WARN] Missing macro file: {file_path}")
                continue

            df = self._load_and_standardise_file(
                file_path=file_path,
                series_name=series_name,
                data_type="macro",
            )
            if not df.empty:
                frames.append(df)

        ecb_file = self._find_ecb_file()
        if ecb_file is not None:
            ecb_df = self._load_and_standardise_file(
                file_path=ecb_file,
                series_name="ecb_series",
                data_type="macro",
            )
            if not ecb_df.empty:
                frames.append(ecb_df)

        if not frames:
            raise FileNotFoundError("No valid macro files were loaded.")

        combined = self._merge_series_on_date(frames, freq="MS")
        combined = self._finalise_dataframe(combined, date_freq="MS")
        return combined

    def build_merged_monthly_dataset(
        self,
        market_df: pd.DataFrame,
        macro_df: pd.DataFrame,
    ) -> pd.DataFrame:
        market_monthly = self._convert_market_daily_to_monthly(market_df)
        macro_monthly = self._ensure_monthly_macro(macro_df)

        merged = pd.merge(market_monthly, macro_monthly, on="date", how="outer")
        merged = merged.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

        merged = merged[
            (merged["date"] >= self.min_valid_date) & (merged["date"] <= self.max_valid_date)
        ].copy()

        non_date_cols = [col for col in merged.columns if col != "date"]
        merged[non_date_cols] = merged[non_date_cols].ffill()

        return merged

    # File loading and standardisation

    def _load_and_standardise_file(
        self,
        file_path: Path,
        series_name: str,
        data_type: str,
    ) -> pd.DataFrame:
        # Special handling for Investing.com FTSE file
        if "ftse" in file_path.name.lower():
            df = pd.read_csv(file_path)
            df.columns = [str(col).strip() for col in df.columns]

            if "Date" not in df.columns or "Price" not in df.columns:
                raise ValueError(
                    f"FTSE file format unexpected: {file_path.name}. "
                    f"Detected columns: {list(df.columns)}"
                )

            df = df[["Date", "Price"]].copy()
            df.columns = ["date", "value"]

            # FTSE exported file uses day-first date format
            df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), errors="coerce", dayfirst=True)
            df["value"] = self._clean_numeric_series(df["value"])

        else:
            raw_df = pd.read_csv(file_path)
            df = raw_df.copy()

            df.columns = [str(col).strip() for col in df.columns]

            date_col = self._detect_date_column(df)
            value_col = self._detect_value_column(df, exclude_cols=[date_col] if date_col else [])

            if date_col is None or value_col is None:
                raise ValueError(
                    f"Could not identify required columns in file: {file_path.name}. "
                    f"Detected columns: {list(df.columns)}"
                )

            df = df[[date_col, value_col]].copy()
            df.columns = ["date", "value"]

            df["date"] = self._clean_date_series(df["date"])
            df["value"] = self._clean_numeric_series(df["value"])

        df = df.dropna(subset=["date"]).copy()

        # Date window restriction
        df = df[(df["date"] >= self.min_valid_date) & (df["date"] <= self.max_valid_date)].copy()

        df = df.sort_values("date")
        df = df.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

        self._log_date_range(file_path=file_path, df=df, series_name=series_name)

        if df.empty:
            print(f"[WARN] {file_path.name} -> {series_name}: empty after cleaning")
            return df

        if data_type == "market":
            df = self._standardise_market_frequency(df)
        elif data_type == "macro":
            df = self._standardise_macro_frequency(df)
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")

        df = df.rename(columns={"value": series_name})
        return df

    def _detect_date_column(self, df: pd.DataFrame) -> Optional[str]:
        preferred = [
            "date",
            "Date",
            "DATE",
            "observation_date",
            "time",
            "Time",
            "TIME_PERIOD",
        ]
        for col in preferred:
            if col in df.columns:
                return col

        for col in df.columns:
            col_lower = col.lower()
            if "date" in col_lower or "time" in col_lower:
                return col

        for col in df.columns:
            sample = df[col].dropna().astype(str).head(10)
            if sample.empty:
                continue
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed", dayfirst=True)
            if parsed.notna().sum() >= max(1, len(sample) - 1):
                return col

        return None

    def _detect_value_column(self, df: pd.DataFrame, exclude_cols: List[str]) -> Optional[str]:
        preferred = [
            "value",
            "Value",
            "VALUE",
            "close",
            "Close",
            "Adj Close",
            "Price",
            "price",
            "PX_LAST",
            "OBS_VALUE",
        ]
        for col in preferred:
            if col in df.columns and col not in exclude_cols:
                return col

        numeric_candidates = []
        for col in df.columns:
            if col in exclude_cols:
                continue

            cleaned = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.replace(r"[^\d\.\-]", "", regex=True)
            )
            converted = pd.to_numeric(cleaned, errors="coerce")
            numeric_ratio = converted.notna().mean()

            if numeric_ratio > 0.5:
                numeric_candidates.append((col, numeric_ratio))

        if numeric_candidates:
            numeric_candidates.sort(key=lambda x: x[1], reverse=True)
            return numeric_candidates[0][0]

        return None

    def _clean_date_series(self, series: pd.Series) -> pd.Series:
        series = series.astype(str).str.strip()

        # First try US format (MM/DD/YYYY)
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=False)

        # Retry failed ones with UK format (DD/MM/YYYY)
        mask = parsed.isna()
        if mask.any():
            parsed_alt = pd.to_datetime(series[mask], errors="coerce", dayfirst=True)
            parsed.loc[mask] = parsed_alt

        # Apply date bounds
        parsed = parsed.where(parsed >= self.min_valid_date)
        parsed = parsed.where(parsed <= self.max_valid_date)

        return parsed

    def _clean_numeric_series(self, series: pd.Series) -> pd.Series:
        cleaned = (
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace(r"[^\d\.\-]", "", regex=True)
            .replace("", np.nan)
        )
        return pd.to_numeric(cleaned, errors="coerce")

    def _log_date_range(self, file_path: Path, df: pd.DataFrame, series_name: str) -> None:
        if "date" not in df.columns or df.empty:
            print(f"[WARN] {file_path.name} -> {series_name}: no valid dates after cleaning")
            return

        valid_dates = df["date"].dropna()
        if valid_dates.empty:
            print(f"[WARN] {file_path.name} -> {series_name}: all dates became NaT")
            return

        print(
            f"[INFO] {file_path.name} -> {series_name}: "
            f"{valid_dates.min().date()} to {valid_dates.max().date()} "
            f"({len(valid_dates)} rows)"
        )

    # Frequency standardisation

    def _standardise_market_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        full_daily_index = pd.date_range(df.index.min(), df.index.max(), freq="D")
        df = df.reindex(full_daily_index)
        df.index.name = "date"

        df["value"] = df["value"].ffill()
        df = df.reset_index()

        return df

    def _standardise_macro_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        monthly = df.resample("MS").last()
        monthly["value"] = monthly["value"].ffill()

        monthly = monthly.reset_index()
        return monthly

    def _convert_market_daily_to_monthly(self, market_df: pd.DataFrame) -> pd.DataFrame:
        df = market_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        monthly = df.resample("MS").last().reset_index()
        return monthly

    def _ensure_monthly_macro(self, macro_df: pd.DataFrame) -> pd.DataFrame:
        df = macro_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        monthly = df.resample("MS").last().reset_index()
        non_date_cols = [col for col in monthly.columns if col != "date"]
        monthly[non_date_cols] = monthly[non_date_cols].ffill()

        return monthly

    # Merge helpers

    def _merge_series_on_date(self, frames: List[pd.DataFrame], freq: str) -> pd.DataFrame:
        merged = frames[0].copy()

        for df in frames[1:]:
            merged = pd.merge(merged, df, on="date", how="outer")

        merged["date"] = pd.to_datetime(merged["date"])
        merged = merged.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

        merged = merged.set_index("date")
        full_index = pd.date_range(merged.index.min(), merged.index.max(), freq=freq)
        merged = merged.reindex(full_index)
        merged.index.name = "date"
        merged = merged.reset_index()

        value_cols = [col for col in merged.columns if col != "date"]
        merged[value_cols] = merged[value_cols].ffill()

        return merged

    def _finalise_dataframe(self, df: pd.DataFrame, date_freq: str) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        df = df[(df["date"] >= self.min_valid_date) & (df["date"] <= self.max_valid_date)].copy()

        if df.empty:
            raise ValueError("Dataframe became empty after date filtering.")

        if date_freq == "D":
            full_index = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
        elif date_freq == "MS":
            full_index = pd.date_range(df["date"].min(), df["date"].max(), freq="MS")
        else:
            raise ValueError(f"Unsupported date frequency: {date_freq}")

        df = df.set_index("date").reindex(full_index)
        df.index.name = "date"
        df = df.reset_index()

        value_cols = [col for col in df.columns if col != "date"]
        df[value_cols] = df[value_cols].ffill()

        return df

    # ECB helper

    def _find_ecb_file(self) -> Optional[Path]:
        pattern = re.compile(r"^ECB Data Portal_.*\.csv$", re.IGNORECASE)

        if not self.raw_macro_dir.exists():
            return None

        for file_path in self.raw_macro_dir.iterdir():
            if file_path.is_file() and pattern.match(file_path.name):
                return file_path

        print("[WARN] ECB file not found in macro folder.")
        return None