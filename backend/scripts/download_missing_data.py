"""
Download missing macro and market datasets:
  - FEDFUNDS  : Federal Funds Effective Rate          (FRED)
  - DFII10    : 10Y TIPS Real Yield                   (FRED)
  - T10YIE    : 10Y Breakeven Inflation Rate          (FRED)
  - DXY       : US Dollar Index (DX-Y.NYB)            (Yahoo Finance)
  - NDX P/E   : NASDAQ-100 Trailing P/E (^NDX)       (Yahoo Finance - best-effort)

All files are saved into backend/data/raw/macro/ or backend/data/raw/market/
with the same CSV convention used by the existing pipeline.
"""

import os
import requests
import yfinance as yf
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # project root
RAW_MACRO = BASE_DIR / "backend" / "data" / "raw" / "macro"
RAW_MARKET = BASE_DIR / "backend" / "data" / "raw" / "market"

RAW_MACRO.mkdir(parents=True, exist_ok=True)
RAW_MARKET.mkdir(parents=True, exist_ok=True)

START_DATE = "2000-01-01"
END_DATE   = "2026-03-25"


# ──────────────────────────────────────────────
# 1. FRED — direct CSV download (no API key)
# ──────────────────────────────────────────────

FRED_SERIES = {
    "FEDFUNDS": RAW_MACRO / "FEDFUNDS.csv",
    "DFII10":   RAW_MACRO / "DFII10.csv",
    "T10YIE":   RAW_MACRO / "T10YIE.csv",
}

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="

for series_id, out_path in FRED_SERIES.items():
    print(f"\n[FRED] Downloading {series_id} ...")
    url = FRED_BASE + series_id
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # FRED returns: DATE,<SERIES_ID>
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"] >= START_DATE].copy()
        df = df[df["value"].notna()].copy()
        # FRED uses '.' for missing — drop those rows
        df = df[df["value"].astype(str) != "."].copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])

        df.to_csv(out_path, index=False)
        print(f"  Saved {len(df)} rows ->{out_path.name}")
        print(f"  Range: {df['date'].min().date()} ->{df['date'].max().date()}")
    except Exception as e:
        print(f"  ERROR: {e}")


# ──────────────────────────────────────────────
# 2. Yahoo Finance — DXY (US Dollar Index)
# ──────────────────────────────────────────────

print("\n[Yahoo Finance] Downloading DXY (DX-Y.NYB) ...")
try:
    dxy = yf.download("DX-Y.NYB", start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)
    if dxy.empty:
        # fallback ticker
        dxy = yf.download("DX=F", start=START_DATE, end=END_DATE,
                          auto_adjust=True, progress=False)

    dxy = dxy[["Close"]].copy()
    dxy.index.name = "date"
    dxy.columns = ["close"]
    dxy = dxy.reset_index()
    dxy["date"] = pd.to_datetime(dxy["date"]).dt.tz_localize(None)

    out_path = RAW_MARKET / "dxy_d.csv"
    dxy.to_csv(out_path, index=False)
    print(f"  Saved {len(dxy)} rows ->{out_path.name}")
    print(f"  Range: {dxy['date'].min().date()} ->{dxy['date'].max().date()}")
except Exception as e:
    print(f"  ERROR: {e}")


# ──────────────────────────────────────────────
# 3. Yahoo Finance — NDX P/E Ratio proxy
#    yfinance does not provide historical index P/E,
#    so we download QQQ (NDX ETF) earnings yield via
#    quarterly EPS estimates as the best free proxy.
#    We also capture NDX trailing twelve-month data
#    from the info dict for the current snapshot.
# ──────────────────────────────────────────────

print("\n[Yahoo Finance] Downloading NDX fundamentals (^NDX info snapshot) ...")
try:
    ndx_ticker = yf.Ticker("^NDX")
    info = ndx_ticker.info

    snapshot = {
        "trailingPE":      info.get("trailingPE"),
        "forwardPE":       info.get("forwardPE"),
        "trailingEps":     info.get("trailingEps"),
        "forwardEps":      info.get("forwardEps"),
        "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
        "fiftyTwoWeekLow":  info.get("fiftyTwoWeekLow"),
        "regularMarketPrice": info.get("regularMarketPrice"),
    }

    snap_df = pd.DataFrame([snapshot])
    snap_df.insert(0, "date", pd.Timestamp.today().date())
    out_path = RAW_MACRO / "ndx_fundamentals_snapshot.csv"
    snap_df.to_csv(out_path, index=False)
    print(f"  Saved snapshot ->{out_path.name}")
    for k, v in snapshot.items():
        if v is not None:
            print(f"    {k}: {v}")
except Exception as e:
    print(f"  ERROR (snapshot): {e}")

# QQQ historical P/E proxy via price / trailing EPS trend
print("\n[Yahoo Finance] Downloading QQQ as NDX P/E historical proxy ...")
try:
    # Download QQQ and SPY for earnings yield computation
    qqq = yf.download("QQQ", start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)
    qqq = qqq[["Close"]].copy()
    qqq.index.name = "date"
    qqq.columns = ["qqq_close"]
    qqq = qqq.reset_index()
    qqq["date"] = pd.to_datetime(qqq["date"]).dt.tz_localize(None)

    out_path = RAW_MARKET / "qqq_d.csv"
    qqq.to_csv(out_path, index=False)
    print(f"  Saved {len(qqq)} rows ->{out_path.name}  (QQQ daily close, NDX ETF proxy)")
    print(f"  Range: {qqq['date'].min().date()} ->{qqq['date'].max().date()}")
except Exception as e:
    print(f"  ERROR (QQQ): {e}")


# ──────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────
print("\n" + "="*55)
print("DOWNLOAD COMPLETE — files written:")
for f in sorted(list(RAW_MACRO.glob("*.csv")) + list(RAW_MARKET.glob("*.csv"))):
    size_kb = f.stat().st_size / 1024
    print(f"  {f.parent.name}/{f.name:<45} {size_kb:6.1f} KB")
print("="*55)
