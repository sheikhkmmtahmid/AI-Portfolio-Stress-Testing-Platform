"""
Phase 7 runner — Portfolio Construction & Stress Testing.

Execution flow:
  1. Load Phase 3/5/6 outputs
  2. Compute expected returns  (7.1)
  3. Build covariance matrix   (7.2)
  4. Optimise portfolio        (7.3)
  5. Apply regime adjustment   (7.4)
  6. Run stress tests          (7.5)
  7. Save portfolio metrics    (7.6)
"""

import sys
from pathlib import Path

backend_root = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_root))

from services.portfolio_engine import PortfolioEngine

PORTFOLIO_DIR = backend_root / "data" / "portfolio"


def _print_section(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def _print_weights(label: str, weights: dict) -> None:
    print(f"\n  {label}:")
    for asset, w in weights.items():
        bar = "#" * int(w * 40)
        print(f"    {asset.upper():<6} {w:.1%}  {bar}")


def main() -> None:
    print("\nPhase 7: Portfolio Construction & Stress Testing")
    print(f"Backend root: {backend_root}")

    engine = PortfolioEngine(backend_root=backend_root)
    result = engine.run()

    # ── Print summary ──────────────────────────────────────────────────────

    _print_section("7.1  Expected Returns (next month, model ensemble)")
    er = result["expected_returns"]
    mw = result["model_weights"]
    for asset in ["spx", "ndx", "gold"]:
        en_w  = mw[asset]["elastic_net"]
        xgb_w = mw[asset]["xgboost"]
        print(f"  {asset.upper():<6}  expected={er[asset]:+.4f}  "
              f"(EN weight={en_w:.2f}, XGB weight={xgb_w:.2f}  "
              f"EN RMSE={mw[asset]['en_test_rmse']:.5f}, XGB RMSE={mw[asset]['xgb_test_rmse']:.5f})")

    _print_section("7.2  Covariance Matrix (Ledoit-Wolf, 36-month window)")
    import pandas as pd
    cov = pd.read_csv(PORTFOLIO_DIR / "covariance_matrix.csv", index_col=0)
    print(cov.to_string())

    _print_section("7.3  Optimised Weights (Max Sharpe, MVO)")
    _print_weights("Base weights", result["base_weights"])

    _print_section(f"7.4  Regime-Adjusted Weights  [regime: {result['current_regime']}  confidence: {result['regime_confidence']:.2f}]")
    _print_weights("Adjusted weights", result["adjusted_weights"])

    _print_section("7.5  Stress Test Results")
    stress = pd.read_csv(PORTFOLIO_DIR / "stress_test_results.csv")
    for _, row in stress.iterrows():
        port_ret = row.get("portfolio_total_return", 0) or 0
        sign = "+" if port_ret >= 0 else ""
        print(f"  [{row['stress_type']:<16}] {str(row['scenario']):<35}  "
              f"portfolio={sign}{port_ret:.2%}")

    _print_section("7.6  Portfolio Risk / Return Metrics")
    m = result["portfolio_metrics"]
    print(f"  Expected return (annual):   {m['expected_return_annual']:+.2%}")
    print(f"  Volatility      (annual):   {m['volatility_annual']:.2%}")
    print(f"  Sharpe ratio    (annual):   {m['sharpe_ratio']:.3f}")
    print(f"  Max drawdown    (hist.):    {m['max_drawdown']:.2%}")
    print(f"  VaR 95%         (monthly):  {m['var_95_monthly']:.2%}")
    print(f"  CVaR 95%        (monthly):  {m['cvar_95_monthly']:.2%}")
    print(f"  Diversif. ratio:            {m['diversification_ratio']:.3f}")
    print(f"  Risk-free rate  (annual):   {m['risk_free_rate_annual']:.2%}")

    print("\n  Per-asset annualised stats:")
    for asset in ["spx", "ndx", "gold"]:
        ret = m["asset_expected_returns_annual"][asset]
        vol = m["asset_volatilities_annual"][asset]
        print(f"    {asset.upper():<6}  ret={ret:+.2%}  vol={vol:.2%}")

    _print_section("Output Files")
    for fname in [
        "expected_returns.csv",
        "covariance_matrix.csv",
        "portfolio_weights.csv",
        "portfolio_weights_regime_adjusted.csv",
        "stress_test_results.csv",
        "portfolio_metrics.json",
    ]:
        fpath = PORTFOLIO_DIR / fname
        size_kb = fpath.stat().st_size / 1024 if fpath.exists() else 0
        print(f"  {fname:<45} {size_kb:6.1f} KB")

    print("\nPhase 7 complete.\n")


if __name__ == "__main__":
    main()
