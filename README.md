# ML-Powered Portfolio Stress Testing Tool with Macroeconomic Scenario Analysis and Market Risk Modelling

> **Academic Title:** An Explainable AI Framework for Portfolio Stress Testing Using Macroeconomic Scenario Analysis, Factor Decomposition, and Unsupervised Market Regime Detection

> **Platform Name:** AI Portfolio Stress Testing Platform

---

## Table of Contents

1. [Project Description](#1-project-description)
2. [What It Does](#2-what-it-does)
3. [Data Sources](#3-data-sources)
4. [Dependencies and Installation](#4-dependencies-and-installation)
5. [Running the Pipeline](#5-running-the-pipeline)
6. [Live Data and Auto-Updates](#6-live-data-and-auto-updates)
7. [Model Results and Explanation](#7-model-results-and-explanation)
8. [Code Explanation](#8-code-explanation)
9. [Conclusion](#9-conclusion)

---

## 1. Project Description

This is a full-stack quantitative finance application that stress tests multi-asset investment portfolios under realistic macroeconomic and market shock scenarios. It is designed for anyone who wants to understand how their portfolio might behave during crises like the 2008 financial crash, the COVID-19 selloff, the 2022 inflation surge, or hypothetical future shocks.

The tool combines machine learning with classical portfolio theory to answer one central question: given a set of portfolio weights across equities, gold and Bitcoin, how much could I lose if the macro environment turns hostile, and why?

It is not a black box. Every estimate the system produces is backed by SHAP explainability output, plain-English narratives and transparent factor exposure analysis. The entire workflow runs locally, from raw data ingestion through to an interactive browser-based dashboard. Once the pipeline has been run once, the system keeps itself up to date by automatically fetching live prices and foreign exchange rates from Yahoo Finance every hour and macroeconomic data from the Federal Reserve every month.

---

## 2. What It Does

The system is structured as a sequential pipeline across nine phases. Each phase builds on the outputs of the previous one. After the initial pipeline run, a live data scheduler embedded in the API continuously refreshes market data, FX rates and macro indicators, and re-runs the portfolio construction phase without any manual intervention.

**Phase 2: Data Ingestion**
Loads and cleans raw market and macroeconomic time series from local CSV files. Standardises column names, handles missing values, aligns dates across sources, and resamples everything to a consistent monthly frequency.

**Phase 3: Feature Engineering**
Transforms cleaned price and rate data into roughly fifty predictive features per time period. These include log returns, rolling volatility, yield spreads, momentum signals, drawdown depth, trend strength indicators, credit stress metrics and inflation regime features. Foreign exchange returns for EUR/USD, GBP/USD and the US Dollar Index are included in this feature set and carry significant predictive weight for Gold and Bitcoin.

**Phase 4: Scenario Construction**
Builds a library of twenty-plus named stress scenarios by replaying the actual macro and market conditions that existed during historical crises including the dot-com crash, global financial crisis, European debt crisis, taper tantrum, COVID crash and the 2022 rate shock, plus four additional synthetic macro shock scenarios. Each scenario is stored as a set of feature values representing the conditions at peak stress.

**Phase 5: Regime Detection**
Uses a Gaussian Mixture Model to classify each monthly observation into one of four market regimes: Calm Growth, Inflation Stress, Credit Stress or Crisis. A confidence score accompanies each classification. The model is paired with a KMeans baseline for validation. Regime labels are appended to the feature dataset as categorical inputs for the asset models.

**Phase 5.5: Regime Transition Analysis**
Computes Markov transition probabilities between regimes, persistence metrics showing how long each regime typically lasts, and regime duration distributions.

**Phase 6: Asset Sensitivity Modelling**
Trains two machine learning models per asset: ElasticNet (linear, interpretable) and XGBoost (tree-based, captures non-linearities). For SPX, NDX and Gold the models use a long history going back to 2003. For Bitcoin the history starts in 2010. Each model learns how each asset's monthly return responds to macro and market features. The final sensitivity estimate is a weighted average of both models.

**Phase 7: Portfolio Construction and Stress Testing**
Applies Modern Portfolio Theory through mean-variance optimisation to compute optimal base weights that maximise the Sharpe ratio across four assets: SPX, NDX, Gold and Bitcoin. A second weight set is produced by tilting the base weights according to the current detected regime, for example increasing gold weight in a Crisis regime. Stress test returns are computed by feeding each scenario's feature values through the trained asset models and aggregating with the portfolio weights. The output includes portfolio VaR, CVaR, volatility, expected return and a diversification ratio. As part of this phase, rolling Pearson correlations are computed between gold returns and the three FX factors (EUR/USD, GBP/USD and DXY) over a 36-month window and saved alongside the portfolio metrics. Phase 7 is re-run automatically by the live data scheduler whenever fresh monthly data is appended.

**Phase 8: Explainability**
Runs SHAP (SHapley Additive exPlanations) over both model types to produce per-feature attribution scores. ElasticNet coefficients are also extracted and aligned with the SHAP rankings. A cross-asset importance table aggregates the most influential macro and market factors across the whole portfolio. For Bitcoin, EUR/USD and GBP/USD rank among the top five SHAP drivers, reflecting the strong dollar-channel sensitivity of crypto markets.

**Phase 9: API, Dashboard and Live Data Scheduler**
Launches a FastAPI web server that serves both a REST API and a server-rendered HTML dashboard. On startup, the server automatically launches a background scheduler that fetches live prices and FX rates every hour, full market data every day and FRED macro data on the first of each month. The dashboard has five pages: Dashboard, Portfolio Builder, Scenario Studio, Results and Methodology. The Live Data Feeds section on the dashboard displays real-time rates across three groups: asset prices (SPX, NDX, Gold, BTC, VIX), major FX rates (EUR/USD, GBP/USD, DXY), and an expanded FX panel covering USD Crosses (USD/JPY, USD/CHF, AUD/USD, USD/CAD, NZD/USD, USD/CNY, USD/SEK, USD/NOK) and BDT Rates (USD/BDT, GBP/BDT, EUR/BDT, and Gold/BDT derived in real time from the gold price and the USD/BDT rate). Each feed group has a manual refresh button alongside the automatic hourly schedule. The Narrative section of the Results page now includes a Gold-FX correlation paragraph explaining the relationship between gold and the dollar across the current market regime. Users can customise portfolio weights interactively, save a custom portfolio to the dashboard for side-by-side comparison against the ML-recommended allocation, run any scenario live, and read a dynamically generated plain-English narrative tied to their specific weight configuration.

---

## 3. Data Sources

### Historical Data (Pipeline Input)

All historical data used to train the models is sourced from publicly available financial databases. No paid subscriptions are required.

**Macroeconomic Data: FRED, Federal Reserve Bank of St. Louis**

Website: https://fred.stlouisfed.org

| Series | Description | FRED Code |
|--------|-------------|-----------|
| 2-Year US Treasury Yield | Short-end interest rate | DGS2 |
| 10-Year US Treasury Yield | Long-end interest rate | DGS10 |
| US Consumer Price Index | Inflation measure | CPIAUCSL |
| ICE BofA US High Yield Spread | Credit risk proxy | BAMLH0A0HYM2 |
| Federal Funds Rate | US monetary policy rate | FEDFUNDS |
| 10-Year TIPS Real Yield | Real rate of return | DFII10 |
| 10-Year Breakeven Inflation | Inflation expectations | T10YIE |
| ECB Deposit Facility Rate | European monetary policy | ECBDFR |

**Market Price Data: Yahoo Finance and Investing.com**

Yahoo Finance: https://finance.yahoo.com
Investing.com: https://www.investing.com

| Instrument | Description |
|------------|-------------|
| SPX (^GSPC) | S&P 500 Index |
| NDX (^NDX) | Nasdaq 100 Index |
| XAUUSD | Spot Gold Price in USD |
| BTCUSD | Bitcoin Price in USD |
| EURUSD | Euro to USD Exchange Rate |
| GBPUSD | British Pound to USD Exchange Rate |
| DXY | US Dollar Index |
| QQQ | Invesco Nasdaq 100 ETF |

### Live Data (Auto-Updated)

Once the API is running, the embedded scheduler automatically keeps the following data fresh.

| Frequency | Source | Data Fetched |
|-----------|--------|--------------|
| Every hour | Yahoo Finance via yfinance | SPX, NDX, Gold, BTC, VIX, EUR/USD, GBP/USD, DXY, USD/JPY, USD/CHF, AUD/USD, USD/CAD, NZD/USD, USD/CNY, USD/SEK, USD/NOK, USD/BDT, GBP/BDT, EUR/BDT, and Gold/BDT (derived) |
| Every day | Yahoo Finance via yfinance | SPX, NDX, Gold, BTC, VIX, US 2Y and 10Y yields, DXY, EUR/USD, GBP/USD, QQQ |
| Every month (1st) | FRED via fredapi | CPI, Fed Funds Rate, HY Spread, TIPS 10Y, Breakeven 10Y, ECB Rate, US 2Y and 10Y yield |

Hourly prices and FX rates are displayed in the Live Data Feeds section on the dashboard. Daily and monthly data is appended to the feature dataset and triggers a Phase 7 re-run so portfolio allocations, risk metrics and stress test results stay current automatically.

---

## 4. Dependencies and Installation

### Requirements

The project runs on Python 3.10 or higher.

```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
jinja2>=3.1.4
pydantic>=2.7.0
pandas>=2.2.0
numpy>=1.26.0
scikit-learn>=1.4.0
xgboost>=2.0.0
scipy>=1.13.0
shap>=0.45.0
joblib>=1.4.0
requests>=2.31.0
yfinance>=0.2.40
fredapi>=0.5.0
schedule>=1.2.0
python-multipart>=0.0.9
```

A `requirements.txt` file is included in the project root. Install everything in one step:

```bash
pip install -r requirements.txt
```

### Installation Steps

**Step 1: Clone or download the project**

```bash
git clone <repository-url>
cd "ML-Powered Portfolio Stress Testing Tool with Macroeconomic Scenario Analysis and Market Risk Modelling"
```

**Step 2: Create and activate a virtual environment**

```bash
python -m venv .venv
```

On Windows:
```bash
.venv\Scripts\activate
```

On macOS or Linux:
```bash
source .venv/bin/activate
```

**Step 3: Install all dependencies**

```bash
pip install -r requirements.txt
```

**Step 4: Set your FRED API key (for monthly macro updates)**

Create a file at `backend/.env` with the following content:

```
FRED_API_KEY=your_key_here
```

Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html

The API reads this file automatically on startup. You never need to set the environment variable manually.

**Step 5: Verify raw data files are in place**

Ensure the following directories contain the expected CSV files before running the pipeline:

```
backend/data/raw/market/     -- SPX, NDX, Gold, BTC, FX, DXY
backend/data/raw/macro/      -- Yields, CPI, credit spreads, TIPS, ECB
```

---

## 5. Running the Pipeline

The pipeline phases must be run in sequence. Each reads the outputs of the previous one. All scripts are run from the `backend/` directory or the project root.

**Phase 2: Ingest and clean raw data**
```bash
python backend/run_phase2.py
```
Outputs: `backend/data/processed/market_clean.csv`, `macro_clean.csv`, `merged_monthly.csv`

**Phase 3: Engineer features**
```bash
python backend/run_phase3.py
```
Outputs: `backend/data/features/features_monthly.csv`, `features_monthly_full_history.csv`, `features_monthly_btc.csv`

**Phase 4: Build stress scenarios**
```bash
python backend/run_phase4.py
```
Outputs: `backend/data/scenarios/scenario_dataset.csv`, `scenario_summary.csv`

**Phase 5: Detect market regimes**
```bash
python backend/run_phase5.py
```
Outputs: GMM and KMeans model files in `backend/models/phase5/`, `backend/data/regimes/regime_dataset.csv`

**Phase 5.5: Analyse regime transitions**
```bash
python backend/run_phase5_5.py
```
Outputs: Transition matrices, persistence and duration tables in `backend/data/regimes/`

**Phase 6: Train asset sensitivity models**
```bash
python backend/run_phase6.py
```
Outputs: ElasticNet and XGBoost models in `backend/models/phase6/` for SPX, NDX, Gold and BTC

**Phase 6.1 (optional): Refine models**
```bash
python backend/run_phase6_1.py
```
Outputs: Refined model variants in `backend/models/phase6_1/`

**Phase 7: Construct portfolio and run stress tests**
```bash
python backend/run_phase7.py
```
Outputs: Portfolio weights, covariance matrix, Gold-FX correlation file, stress test results and portfolio metrics in `backend/data/portfolio/`

**Phase 8: Generate SHAP explainability**
```bash
python backend/run_phase8.py
```
Outputs: SHAP values, global importance tables and factor comparison CSV in `backend/data/explainability/`

**Phase 9: Launch the API and dashboard**
```bash
python backend/run_phase9.py
```

Then open your browser and navigate to:
```
http://localhost:8000
```

The live data scheduler starts automatically the moment the API starts. No separate command is needed.

### Standalone Scheduler (Alternative to Phase 9)

If you want to run the live data scheduler separately from the web server, a standalone script is provided:

```bash
python backend/run_scheduler.py
```

This runs the same hourly, daily and monthly data jobs as the embedded scheduler in Phase 9, but without starting the FastAPI web server. It is useful when you want to keep data refreshed in the background while the API server is managed separately or not running. The script logs to both stdout and `backend/data/live/scheduler.log`.

### Data Bootstrap Utility

If any raw CSV files are missing before the first pipeline run, a download utility is included:

```bash
python backend/scripts/download_missing_data.py
```

This script downloads the following datasets automatically and saves them to the correct `backend/data/raw/` subdirectories:

- **FEDFUNDS** (Federal Funds Rate) from FRED via direct CSV download
- **DFII10** (10-Year TIPS Real Yield) from FRED via direct CSV download
- **T10YIE** (10-Year Breakeven Inflation) from FRED via direct CSV download
- **DXY** (US Dollar Index) from Yahoo Finance via yfinance
- **QQQ** daily close prices from Yahoo Finance as an NDX ETF proxy
- **NDX fundamentals snapshot** with current trailing and forward P/E from Yahoo Finance

No API key is required for the FRED downloads in this script; it uses the public CSV endpoint directly.

### Quick-Start After the First Full Run

Once all pipeline outputs have been generated, you only need to restart the server:

```bash
python backend/run_phase9.py
```

The server reads all pre-computed outputs from `backend/data/`, serves them immediately, and the scheduler begins refreshing live data in the background. You do not need to re-run phases 2 through 8 unless the input data changes or you want to retrain the models.

---

## 6. Live Data and Auto-Updates

The live data system is built into the API process. There is no separate process to manage.

### How It Works

When `run_phase9.py` starts, it launches a background thread called `LiveDataScheduler`. On startup it immediately runs all three jobs (hourly, daily and monthly) to ensure data is fresh right away. It then enters a loop that checks for scheduled jobs every 30 seconds.

**Hourly job**
Fetches current prices for 19 instruments from Yahoo Finance: SPX, NDX, Gold, BTC, VIX, EUR/USD, GBP/USD, DXY, USD/JPY, USD/CHF, AUD/USD, USD/CAD, NZD/USD, USD/CNY, USD/SEK, USD/NOK, USD/BDT, GBP/BDT and EUR/BDT. After the fetch completes, a twentieth value, Gold/BDT, is derived in real time by multiplying the gold price in USD by the USD/BDT rate. All twenty values are saved to `backend/data/live/latest_prices.json` and the freshness file is updated with a status and timestamp. The entire Live Data Feeds section on the dashboard reflects these values on every page load. Each feed group also has a manual refresh button that triggers the same fetch immediately without waiting for the next scheduled run.

**Daily job**
Fetches a broader set of 11 instruments including US 2Y and 10Y yields, DXY, EUR/USD, GBP/USD and QQQ. If the calendar has rolled into a new month since the last feature row was written, it constructs a new monthly feature row and appends it to the feature dataset. It then re-runs Phase 7 so the portfolio weights, stress tests, risk metrics and Gold-FX correlations reflect the latest data.

**Monthly job**
Runs on the 1st of each month at 08:00 UTC. Fetches the eight FRED macro series listed in the Data Sources section. Updates the feature dataset with fresh macro readings and re-runs Phase 5 (regime detection) and Phase 7 (portfolio construction) so the regime label and portfolio allocation reflect the current macroeconomic environment.

### FRED API Key

Monthly macro updates require a free FRED API key. Set it once in `backend/.env`:

```
FRED_API_KEY=your_key_here
```

Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html

Without the key, hourly and daily updates continue to work normally. The dashboard will show a warning badge on the Monthly feed card.

### What Updates on the Dashboard

| Dashboard Section | Updated by | Frequency |
|------------------|------------|-----------|
| Asset price tiles (SPX, NDX, Gold, BTC, VIX) | Hourly job | Every hour |
| Major FX tiles (EUR/USD, GBP/USD, DXY) | Hourly job | Every hour |
| USD Crosses (JPY, CHF, AUD, CAD, NZD, CNY, SEK, NOK) | Hourly job | Every hour |
| BDT Rates (USD/BDT, GBP/BDT, EUR/BDT, Gold/BDT) | Hourly job | Every hour |
| Feed status cards | All jobs | Each job run |
| Portfolio Allocation donut | Daily or monthly job (via Phase 7) | When a new month rolls over |
| Market Regime | Monthly job (via Phase 5) | Monthly on the 1st |
| Stress Test results | Daily or monthly job (via Phase 7) | When a new month rolls over |
| Portfolio metrics (Sharpe, VaR, CVaR) | Daily or monthly job (via Phase 7) | When a new month rolls over |

---

## 7. Model Results and Explanation

### Regime Detection (Phase 5)

The GMM regime classifier identifies four distinct market states from the macro and market feature history:

| Regime | Characteristics |
|--------|----------------|
| Calm Growth | Low volatility, positive equity momentum, stable credit spreads |
| Inflation Stress | Rising yields, elevated breakeven inflation, compressed real rates |
| Credit Stress | Widening high-yield spreads, deteriorating credit conditions |
| Crisis | VIX spike, sharp equity drawdowns, flight to safety, USD strength |

The GMM assigns a probability to each regime for every monthly observation, and the highest-probability label is used as the categorical input to the asset models. Asset sensitivities to macro factors differ meaningfully across regimes. Gold behaves very differently in a Crisis regime compared to Calm Growth, and so does the relationship between gold and the US Dollar.

### Asset Sensitivity Models (Phase 6)

For each of the four assets (SPX, NDX, Gold, BTC), two models are trained.

**ElasticNet** is a regularised linear regression combining L1 (lasso) and L2 (ridge) penalties. It is selected for its interpretability. The non-zero coefficients directly indicate which macro factors most linearly influence each asset's monthly return.

**XGBoost** is a gradient-boosted tree ensemble. It captures non-linear interactions between features (for example, the combination of high inflation and rising credit spreads) that a linear model cannot represent. Feature importance is measured by the average information gain at each split.

Together these two models form an ensemble. The final sensitivity estimate used in portfolio stress testing is a blended output from both.

For SPX and NDX, the most influential features tend to be the yield spread (10-year minus 2-year), trailing equity momentum, VIX level and credit spread changes. For Gold, real yields and breakeven inflation dominate, with the US Dollar Index carrying a consistent inverse relationship. For BTC, EUR/USD and GBP/USD rank in the top five SHAP drivers alongside the 10-year Treasury yield, reflecting the strong dollar-channel sensitivity of crypto markets.

### Gold-FX Correlation Analysis

As part of portfolio construction, the system computes rolling 36-month Pearson correlations between gold returns and the three FX factors: EUR/USD return, GBP/USD return and the DXY return. These correlations are stored in `backend/data/portfolio/gold_fx_correlations.json` and fed into the narrative engine.

The plain-English Diversification Analysis narrative on the Results page uses these correlations to explain the gold-dollar relationship in the context of the current regime. In a Calm Growth regime, the typical inverse gold-dollar relationship holds and DXY strength modestly pressures gold. In an Inflation Stress regime, gold can rally despite dollar strength because inflation-hedge demand overrides the usual FX drag. In a Crisis regime, an initial dollar surge tends to pressure gold before safe-haven demand reverses the move. The narrative captures whichever of these dynamics the 36-month data currently reflects.

If a FX factor (DXY, EUR/USD or GBP/USD) ranks as the dominant SHAP driver for the portfolio, the system now also generates a contextual explanation in the Dominant Risk Factor narrative card, describing the directional relationship between that currency move and gold and equity returns.

### Portfolio Optimisation (Phase 7)

The base portfolio weights are computed via mean-variance optimisation targeting the maximum Sharpe ratio, subject to long-only constraints and weights summing to one. A Ledoit-Wolf shrinkage estimator is applied to the covariance matrix to reduce estimation error from the limited monthly sample. The optimisation runs across all four assets: SPX, NDX, Gold and Bitcoin.

A second weight set is computed by applying regime-based tilts on top of the MVO baseline. In a Crisis regime the system increases the gold allocation and reduces equity exposure. In Calm Growth the equity tilt is restored.

**Typical portfolio metrics (will vary with current data):**
- Expected Annual Return: 8 to 15 percent depending on regime
- Sharpe Ratio: 0.8 to 1.6
- Annual Volatility: 10 to 18 percent
- 95 percent monthly VaR: negative 4 to negative 8 percent
- CVaR at 95 percent: negative 6 to negative 11 percent
- Diversification Ratio: above 1.2

### Stress Test Results

Under the twenty-plus stress scenarios, the range of portfolio outcomes broadly follows this pattern:

- **Mild scenarios** (Taper Tantrum 2013, China Slowdown 2015): Portfolio returns of negative 5 to negative 8 percent. Gold partially offsets equity losses.
- **Moderate scenarios** (European Debt Crisis 2011, Oil Crash 2014): Portfolio returns of negative 10 to negative 18 percent. Regime-adjusted weights provide meaningful protection.
- **Severe scenarios** (Global Financial Crisis 2008, COVID Crash 2020): Portfolio returns of negative 20 to negative 35 percent. Gold acts as an anchor but cannot fully offset the depth of equity losses.
- **Extreme scenarios** (hypothetical simultaneous rate shock plus credit crisis): Portfolio returns of negative 30 to negative 45 percent with concentrated losses in equities.

### SHAP Explainability (Phase 8)

SHAP values quantify how much each feature pushed a model's prediction above or below the average for each individual observation. The global importance chart consistently highlights:

- **Yield spread (10-year minus 2-year):** A top driver for all equity assets. A flattening or inverted yield curve precedes losses.
- **VIX level and change:** High fear levels reliably predict negative equity returns.
- **High-yield credit spread change:** Credit stress is an early warning signal for equity weakness.
- **Gold returns lagged 1 month:** Lagged gold performance carries regime information.
- **Breakeven inflation:** Impacts all assets differently depending on the real rate environment.
- **EUR/USD and GBP/USD returns:** Significant drivers for Bitcoin and indirect drivers for Gold through the dollar channel.

---

## 8. Code Explanation

The project is split into backend logic, a REST API layer and a frontend presentation layer.

### backend/services/

This is where the quantitative engine lives.

`data_ingestion.py` handles all raw data loading. It reads CSVs, renames columns to a standard internal schema, forward-fills sparse macro series, and resamples to monthly close.

`feature_engineering.py` takes the clean merged data and computes the feature matrix. Key computations include log return transformations, rolling standard deviations for volatility, lagged values for momentum and carry signals, drawdown calculations, yield curve slope and curvature, and Z-score normalisation. FX return columns for EUR/USD, GBP/USD and the DXY are included in the feature matrix and are available to all downstream models.

`scenario_engine.py` defines each named scenario as a dictionary mapping feature names to their observed values at peak stress. It applies these to the feature dataset and outputs a scenario matrix for the asset models.

`regime_detection.py` fits the GMM and KMeans models on the feature history, assigns regime labels, computes confidence scores via GMM component probabilities, and maps numeric clusters to human-readable regime names based on their macro characteristics.

`asset_models.py` builds preprocessing pipelines (imputation, scaling, one-hot encoding) and trains ElasticNet and XGBoost models per asset. It uses scikit-learn cross-validation for hyperparameter selection.

`asset_models_phase6_1.py` is a refined variant of the asset modelling layer used by `run_phase6_1.py`. It extends the core numeric feature set with optional ECB-related features, an expanded set of lagged columns for SPX, NDX, Gold, VIX, yield spread and high-yield spread, and asset-specific extra features for Gold (real yield and breakeven inflation) and Bitcoin (bitcoin-specific volatility and drawdown metrics). The preprocessing pipeline and ElasticNet / XGBoost architecture mirror Phase 6, but the broader feature set and additional lag windows allow the models to capture slower-moving macro dynamics that the base Phase 6 models may not fully exploit. Outputs are written to `backend/models/phase6_1/` and can be loaded by the portfolio engine as drop-in replacements for the Phase 6 models.

`regime_transitions.py` implements the `RegimeTransitionService` class used by Phase 5.5. It reads the regime dataset produced by the GMM classifier and computes three outputs: a Markov transition matrix showing the probability of moving from each regime to every other regime; a persistence table showing the self-transition probability and average run length for each regime; and a duration distribution table recording the minimum, median and maximum number of consecutive months each regime has historically lasted. These outputs are saved to `backend/data/regimes/` and are displayed on the Methodology page of the dashboard.

`portfolio_engine.py` implements MVO using scipy's constrained optimiser across all four assets (SPX, NDX, Gold and BTC). It applies Ledoit-Wolf shrinkage to the covariance matrix, applies regime tilts, runs all stress test scenarios through the trained models, and computes VaR, CVaR and the diversification ratio. Bitcoin return data is merged from a separate features file covering its shorter available history, with pre-2010 periods handled gracefully. After computing the covariance matrix, the engine runs a second computation that calculates 36-month rolling Pearson correlations between gold returns and the EUR/USD, GBP/USD and DXY return columns from the same feature dataset. The results are saved to `backend/data/portfolio/gold_fx_correlations.json` for use by the narrative engine.

`explainability_engine.py` uses the SHAP library to compute TreeExplainer values for XGBoost and LinearExplainer values for ElasticNet. It aggregates per-sample SHAP values into global importance rankings.

`narrative_engine.py` generates plain-English commentary by translating factor exposures, SHAP rankings and regime states into readable sentences about portfolio risk drivers. It supports dynamic generation for any custom weight configuration. The engine now includes dedicated contextual interpretations for the three FX factors: when DXY, EUR/USD or GBP/USD ranks as the dominant SHAP driver, the Dominant Risk Factor narrative explains the directional link between that currency move and portfolio assets. The Diversification Analysis narrative includes a Gold-FX correlation paragraph that reads the 36-month gold-dollar and gold-EUR/USD correlations and frames them within the current regime context. If the Gold-FX correlation file has not yet been written by the portfolio engine (for example on a fresh server start before Phase 7 has been re-run), the narrative engine computes the correlations inline from the features CSV as a fallback so the narrative is always complete.

`live_data_fetcher.py` handles all live data retrieval. It exposes three functions: `run_hourly` fetches prices and rates for 19 instruments from Yahoo Finance via yfinance, then derives a twentieth value (Gold/BDT) by multiplying the gold price in USD by the USD/BDT rate; `run_daily` fetches a broader market dataset and appends new monthly rows to the feature CSV when the calendar rolls over; `run_monthly` fetches FRED macro series via fredapi, updates the feature dataset and triggers Phase 5 and Phase 7 re-runs. All three functions update `backend/data/live/data_freshness.json` with their status and timestamp after each run.

### backend/api/

`main.py` is the FastAPI application entry point. It registers all API routes, configures CORS and mounts the static file directory. It uses a FastAPI lifespan context manager to start the `LiveDataScheduler` background thread automatically when the server starts and stop it cleanly when the server stops. Key endpoints include `POST /api/portfolio/analyze`, `POST /api/portfolio/save`, `DELETE /api/portfolio/save`, `POST /api/scenario/run`, `GET /api/regime/current` and `POST /api/live-data/refresh/{feed_type}`. The refresh endpoint accepts `hourly`, `daily` or `monthly` as the feed type and runs the corresponding fetcher function on demand, returning the updated prices and status in the response for the dashboard to update in place.

`pages.py` handles the HTML page routes. For each of the five dashboard pages it loads the relevant pre-computed data files (CSVs and JSONs), converts numpy types to native Python for Jinja2 compatibility, and passes the data as template context. The dashboard route calls `_live_data_context()` which loads `latest_prices.json` and `data_freshness.json` to populate the Live Data Feeds section. All dashboard responses are served with no-cache headers to ensure the browser always receives the latest data.

### frontend/

`templates/base.html` defines the master layout: collapsible sidebar, top header bar with regime badge and portfolio KPIs, and a responsive main content zone.

`templates/dashboard.html` is the landing page. It shows portfolio KPI cards, an allocation donut chart, a regime panel with history and distribution, a stress scenario summary, factor exposure bars, a historical drawdown chart, a My Portfolio comparison card showing any user-saved custom allocation, and a Live Data Feeds section at the bottom. The Live Data Feeds section is organised into four rows: asset prices (SPX, NDX, Gold, BTC, VIX), a Major FX row (EUR/USD, GBP/USD, DXY), a USD Crosses row (USD/JPY, USD/CHF, AUD/USD, USD/CAD, NZD/USD, USD/CNY, USD/SEK, USD/NOK), and a BDT Rates row (USD/BDT, GBP/BDT, EUR/BDT, Gold/BDT). FX pairs close to 1.0 are displayed to four decimal places; larger-valued pairs such as USD/JPY and USD/BDT are shown to two decimal places; Gold/BDT is displayed as a whole number with comma formatting. Each feed card has a manual refresh button that triggers a POST to the refresh endpoint and updates all tiles in place without reloading the page.

`templates/portfolio.html` lets the user move sliders to set custom weights for SPX, NDX, Gold and BTC. It validates that weights sum to 100 percent and calls `POST /api/portfolio/analyze` live to update all metrics without a page reload. Users can save their custom allocation to the dashboard with a single click, which stores the weights and metrics to disk and redirects them to the dashboard for side-by-side comparison with the ML-recommended portfolio.

`templates/scenario.html` is the Scenario Studio. Users pick any scenario from a grid of cards and see how their custom weights would perform. The Reverse Stress Test section lets them drag a target loss slider and request the macro shock combination that produces it.

`templates/results.html` provides a tabbed deep-dive with a scenario return table, SHAP heatmap, factor exposures heatmap and a live narrative generation panel tied to the user's weight configuration. The Diversification Analysis card within the narrative panel now includes the Gold-FX correlation paragraph with regime context.

`static/css/theme.css` defines the entire visual language: dark background, gold and scarlet accent system, card components, chart containers and all transition animations.

`static/js/main.js` bootstraps all Chart.js charts, wires slider events to API calls, manages tab switching and handles the fetch lifecycle for all interactive API interactions.

### backend/run_phase9.py

The single entry point for running the application. On startup it reads `backend/.env` to load any environment variables (including `FRED_API_KEY`), then starts uvicorn. The FastAPI lifespan hook launches the scheduler automatically, so one command starts everything: the web server, the REST API, the HTML dashboard and the live data system.

---

## 9. Conclusion

This project started with the question every portfolio manager and serious investor eventually confronts: not just what returns can I expect, but what is the realistic worst case, and what drives it?

The answer the tool builds toward is deliberately layered. At the surface it gives you numbers: portfolio VaR, scenario returns, expected Sharpe ratios. One layer deeper it gives you attribution: which macro factors are responsible and in what proportion. Deeper still it gives you regime context: are the risks elevated right now because we are in a credit stress environment, or is the calm masking a transition that the model has already started to flag?

The foreign exchange layer adds another dimension to that attribution. The gold-dollar relationship is one of the most studied in financial markets, and the system now makes it explicit. The 36-month rolling correlation between gold and the DXY tells you whether that inverse relationship is holding, weakening, or temporarily breaking down (as it often does in crisis regimes when safe-haven flows lift both simultaneously). The live FX rates panel extends this visibility further, showing USD/BDT, GBP/BDT, EUR/BDT and Gold/BDT alongside the major pairs so users with exposure across multiple currencies can read the macro picture in the terms that are most relevant to them.

The reverse stress testing capability flips the usual question on its head. Rather than asking what happens under a historical crisis, you ask what would have to happen for your portfolio to lose 25 percent. The answer is expressed in terms of real, observable macro quantities (yield spread widening, VIX spike, credit spread blow-out) which gives you something actionable: a set of macro conditions to monitor.

The live data layer means the system does not freeze in time after the initial pipeline run. Every hour new prices and FX rates arrive, every month new macro data reshapes the regime picture, and the portfolio allocation adjusts accordingly without any manual steps.

The dynamic narrative layer means you are never left staring at numbers without context. Every weight configuration generates its own commentary, grounded in the same SHAP values and factor exposures that underpin the quantitative outputs.

This is not a trading system and it is not a predictive service. It is an analytical environment for building stress-awareness into portfolio thinking. The models are trained on history, and history does not repeat exactly. But it rhymes often enough that understanding how past crises mapped onto factor exposures, regime states and portfolio losses is one of the most honest risk management exercises available.

The architecture is fully local, fully transparent and fully documented. Every number on the dashboard can be traced back through the pipeline to a data source, a model coefficient or a SHAP value. That traceability is intentional. Good risk management is not magic. It is structured thinking made visible.

---

*Built with Python, FastAPI, scikit-learn, XGBoost, SHAP, yfinance, fredapi and Chart.js.*
*Historical data from FRED, Yahoo Finance, Investing.com and the ECB Data Portal.*
*Live data updated automatically via the Yahoo Finance and FRED APIs.*
