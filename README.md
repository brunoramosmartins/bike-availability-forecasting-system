# Bike Availability Forecasting System

[![CI](https://github.com/brunoramosmartins/bike-availability-forecasting-system/actions/workflows/ci.yml/badge.svg)](https://github.com/brunoramosmartins/bike-availability-forecasting-system/actions/workflows/ci.yml)
[![Ingestion](https://github.com/brunoramosmartins/bike-availability-forecasting-system/actions/workflows/ingest.yml/badge.svg)](https://github.com/brunoramosmartins/bike-availability-forecasting-system/actions/workflows/ingest.yml)
[![Coverage](https://img.shields.io/badge/coverage-96%25-brightgreen)](./pyproject.toml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![PostgreSQL](https://img.shields.io/badge/database-PostgreSQL%20(Neon)-336791)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

An end-to-end data engineering and machine learning system that forecasts bike availability in urban bike-sharing stations using real-time [GBFS](https://gbfs.org/) data from **Bike Itau / Bike Sampa** (Sao Paulo, Brazil).

## Overview

This project builds a continuous data pipeline that ingests high-frequency station data, constructs a time-series dataset, and trains predictive models to estimate future bike availability (t+15 min). It demonstrates the full ML lifecycle: ingestion, storage, feature engineering, modeling, evaluation, monitoring, visualization, and deployment.

### Key Features

- **Real-time ingestion** — Scheduled collection of GBFS station data every 5 minutes via GitHub Actions
- **Structured storage** — PostgreSQL (Neon) with raw tables and a curated **`analytics`** layer for ML and BI
- **Data quality** — Automated DQ metric views and CLI checks (`python -m src.storage.data_quality`)
- **Feature engineering** — Lag, rolling, temporal, and station features with leakage-free pipeline
- **Baseline models** — Naive (last known value) and Linear Regression establishing a performance floor
- **Advanced models** — LightGBM and XGBoost with Optuna hyperparameter tuning and SHAP interpretability
- **Model monitoring** — Drift detection and performance tracking with Evidently AI
- **Visualization** — Interactive Streamlit dashboard with Plotly charts (availability timeline, station heatmap, peak hours, model performance)
- **Prediction API** — FastAPI REST endpoint for real-time availability forecasts (`/predict`, `/stations`, `/anomalies`)
- **Anomaly detection** — Rule-based (stuck stations) + Isolation Forest for identifying malfunctioning stations

### Model Performance

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Naive Baseline | 0.298 | 0.938 | 0.967 |
| Linear Regression | 0.213 | 0.664 | 0.984 |
| **LightGBM (tuned)** | **TBD** | **TBD** | **TBD** |
| XGBoost | TBD | TBD | TBD |

> LightGBM results will be populated after running `python -m src.model` with the tuned pipeline.
> Baseline results are from 240 stations over ~11 days of real data.

### Project Progress

| Phase | Status | Description |
|-------|--------|-------------|
| 0 — Repository Bootstrap | :white_check_mark: Done | Project scaffold, governance files, CI/CD |
| 1 — Data Reconnaissance | :white_check_mark: Done | GBFS feed exploration, schema docs, station map |
| 2 — Data Ingestion Pipeline | :white_check_mark: Done | Fetch -> parse -> load pipeline, GitHub Actions cron |
| 3 — Data Modeling & Storage | :white_check_mark: Done | Analytics views, indexes, DQ metrics |
| 4 — Dataset Construction | :white_check_mark: Done | Feature engineering, temporal splits, Parquet export |
| 5 — Baseline Modeling | :white_check_mark: Done | Naive + Linear Regression baselines, evaluation metrics |
| 6 — Advanced Modeling | :white_check_mark: Done | LightGBM + XGBoost with Optuna tuning, SHAP analysis |
| 7 — Monitoring & Drift | :white_check_mark: Done | Drift detection, Evidently AI reports, alerting |
| 8 — Visualization | :white_check_mark: Done | Streamlit dashboard with Plotly charts |
| 9 — Extensions | :white_check_mark: Done | FastAPI prediction API, anomaly detection |

## Architecture

```
GitHub Actions (cron: */5 * * * *)
        |
        v
  GBFS API (Bike Itau)
        |
        v
  Python Ingestion Service
  fetch -> validate -> load
        |
        v
  PostgreSQL (Neon)
  +------------------+--------------------+
  | raw_station_status| station_information|   <- Raw (Bronze)
  +---------+--------+----------+---------+
            |                   |
            +--------+----------+
                     v
           +---------------------+
           |  schema: analytics   |   <- Curated (Silver)
           |  station_status_*    |
           |  v_dq_* metrics      |
           +---------+-----------+
                     |
                     v
           +---------------------+
           |  Dataset Pipeline    |   <- Features (Gold)
           |  resample -> features|
           |  -> split -> Parquet |
           +---------+-----------+
                     |
        +-------+--------+--------+--------+
        v       v        v        v        v
   ML Models  Dashboard Monitoring  API   Anomaly
  (Phase 5-6) (Phase 8) (Phase 7) (Ph.9) (Ph.9)
        |                            |
        v                            v
  +---------------------+   +------------------+
  | LightGBM | XGBoost  |   | FastAPI          |
  | metrics.json         |   | /predict         |
  | SHAP importance      |   | /stations        |
  +---------------------+   | /anomalies       |
                             +------------------+
```

See [docs/analytics/README.md](./docs/analytics/README.md) for grain, ER/layer diagrams (Mermaid), and the data dictionary.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Ingestion | Python, httpx, GitHub Actions |
| Storage | PostgreSQL (Neon free tier) |
| Processing | pandas, SQL |
| Modeling | scikit-learn, LightGBM, XGBoost |
| Tuning | Optuna (TPE sampler, 50 trials) |
| Interpretability | SHAP (TreeExplainer) |
| Monitoring | Evidently AI, scipy (KS test) |
| Visualization | Streamlit, Plotly, Tableau Public *(planned)* |
| API | FastAPI, Pydantic, uvicorn |
| Anomaly Detection | scikit-learn (Isolation Forest), rule-based |

## Project Structure

```
├── src/
│   ├── ingestion/       # GBFS fetch -> parse -> load pipeline
│   ├── storage/         # DB connection, schema, data quality CLI
│   ├── dataset/         # Resampling, feature engineering, splitting
│   ├── model/           # Baseline + advanced models, evaluation, tuning
│   │   ├── baseline.py  # NaiveBaseline, LinearRegressionModel
│   │   ├── advanced.py  # LightGBMModel, XGBoostModel, tune_lightgbm()
│   │   ├── evaluate.py  # compute_metrics, per_station/hour breakdowns
│   │   └── __main__.py  # CLI: python -m src.model
│   ├── monitoring/      # Drift detection, Evidently reports, prediction store
│   │   ├── drift.py     # rolling_mae, PSI, KS test, analyze_drift()
│   │   ├── store.py     # save/load predictions, backfill actuals
│   │   ├── reporter.py  # Evidently AI HTML report generation
│   │   └── __main__.py  # CLI: python -m src.monitoring
│   ├── dashboard/       # Streamlit interactive dashboard
│   │   ├── data.py      # Pure data helpers (loading, filtering, aggregation)
│   │   ├── app.py       # Multi-page entry point: streamlit run src/dashboard/app.py
│   │   └── views/       # availability, heatmap, peak_hours, performance, drift_monitor
│   ├── anomaly/         # Anomaly detection (stuck stations, Isolation Forest)
│   │   └── detector.py  # detect_stuck_stations(), detect_statistical_anomalies()
│   └── api/             # FastAPI prediction endpoint
│       ├── main.py      # App factory: uvicorn src.api.main:app
│       ├── routes.py    # /health, /stations, /predict, /anomalies
│       └── schemas.py   # Pydantic request/response models
├── tests/               # 200+ unit tests (pytest, 96% coverage)
├── notebooks/
│   ├── 01_data_exploration.ipynb   # GBFS schema, station map (Folium)
│   ├── 02_feature_analysis.ipynb   # Feature distributions, leakage check
│   ├── 03_model_comparison.ipynb   # Baseline model comparison
│   └── 04_advanced_models.ipynb    # LightGBM/XGBoost, SHAP analysis
├── sql/                 # DDL migrations (001_ -> 006_)
│   ├── 001_create_tables.sql       # raw_station_status, station_information
│   ├── 002_create_indexes.sql      # Composite index for time-series
│   ├── 003_analytics_layer.sql     # analytics.station_status_enriched/latest
│   └── 004_data_quality_views.sql  # analytics.v_dq_* metric views
├── data/
│   ├── samples/         # Saved GBFS API responses (Phase 1)
│   └── processed/       # train/val/test Parquet + model artifacts
├── docs/analytics/      # ER diagrams, data dictionary, DQ glossary
└── .github/workflows/
    ├── ci.yml           # Lint (ruff) + tests (pytest, 80% coverage gate)
    └── ingest.yml       # Scheduled ingestion every 5 minutes
```

## Getting Started

### Prerequisites

- Python 3.10+
- PostgreSQL database ([Neon free tier](https://neon.tech/) recommended)
- GitHub account (for Actions-based scheduling)

### Setup

```bash
# Clone the repository
git clone https://github.com/brunoramosmartins/bike-availability-forecasting-system.git
cd bike-availability-forecasting-system

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install dependencies (production + dev tools)
pip install -e .[dev]

# Configure environment
cp .env.example .env
# Edit .env with your DATABASE_URL and other settings
```

### Running the Pipelines

```bash
# 1. Ingest GBFS data into PostgreSQL
python -m src.ingestion

# 2. Build ML-ready dataset (requires 7+ days of ingestion data)
python -m src.dataset

# 3. Train all models (Optuna tuning + evaluation)
python -m src.model

# 4. Run data quality checks
python -m src.storage.data_quality --json

# 5. Launch the interactive dashboard
streamlit run src/dashboard/app.py

# 6. Start the prediction API
uvicorn src.api.main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

### Data model and analytics layer (Phase 3)

Raw tables:

- **`raw_station_status`** — append-only snapshots; unique on `(station_id, last_reported)`.
- **`station_information`** — one row per station (SCD Type 1 upsert from GBFS).

Curated **`analytics`** schema (views):

- **`analytics.station_status_enriched`** — fact grain plus station attributes (join on `station_id`). Primary interface for time-series extracts and feature engineering.
- **`analytics.station_status_latest`** — one row per station (latest `last_reported`, tie-break `ingestion_timestamp`).

Indexes: composite `(station_id, last_reported DESC)` plus existing single-column indexes (see `sql/002_create_indexes.sql`).

**Data quality:** metric views `analytics.v_dq_*` and a CLI:

```bash
python -m src.storage.data_quality
python -m src.storage.data_quality --json
```

Exit code `0` when all checks pass, `1` otherwise. Full documentation: [docs/analytics/README.md](./docs/analytics/README.md).

### How downstream phases consume this layer

| Phase | Consumption pattern | Benefit |
|-------|---------------------|---------|
| **4 — Dataset** | `SELECT ... FROM analytics.station_status_enriched WHERE ...` (time windows per `station_id`) | Stable column contract, no duplicated join logic in Python; explicit grain for resampling and lags. |
| **5-6 — Modeling** | Parquet extracts from the dataset pipeline | Features and target definitions stay aligned with documented semantics. |
| **7 — Monitoring** | Compare predictions to actuals joined on `station_id` + time; optional DQ views in scheduled jobs | Shared vocabulary for "what a row means"; DQ metrics reused for operational trust. |
| **8 — Visualization** | Tableau (or extract) against enriched view or aggregates | One semantic layer for "availability over time" and maps (`lat`/`lon` already on the row). |

**Why it matters:** ingestion owns **raw** tables; **analytics** owns the **contract** everyone else reads. That reduces drift between SQL in notebooks, training code, and dashboards, and centralizes joins and naming.

### Dataset construction pipeline (Phase 4)

The dataset module reads from `analytics.station_status_enriched`, resamples to 15-minute intervals, engineers features, and exports train/val/test splits as Parquet files.

```bash
python -m src.dataset
```

**Features (15 columns)**:

| Group | Features |
|-------|----------|
| Current | `num_bikes_available`, `num_docks_available` |
| Lag (t-15 to t-60 min) | `bikes_lag_1`, `bikes_lag_2`, `bikes_lag_3`, `bikes_lag_4` |
| Rolling (1h window) | `bikes_rolling_mean_1h`, `bikes_rolling_std_1h` |
| Temporal | `hour`, `weekday`, `is_weekend`, `month` |
| Station | `capacity`, `lat`, `lon` |

**Target**: `y` = `num_bikes_available` at t+15 min

**Split**: time-based (train 80%, validation 10%, test 10%) — no shuffle, no temporal overlap.

**Output**: `data/processed/train.parquet`, `val.parquet`, `test.parquet`

### Modeling pipeline (Phases 5-6)

The model module trains all models, runs Optuna hyperparameter tuning for LightGBM, evaluates on the test set, and saves artifacts:

```bash
python -m src.model
```

**Models**:

| Model | Type | Description |
|-------|------|-------------|
| Naive Baseline | Heuristic | Predicts `bikes_lag_1` (last known value) |
| Linear Regression | OLS | scikit-learn on all 15 features |
| LightGBM | Gradient Boosting | Tuned with Optuna (50 trials, TPE sampler) |
| XGBoost | Gradient Boosting | Default hyperparameters, included for comparison |

**Tuning strategy**: Optuna minimizes MAE on the held-out temporal validation set (`val.parquet`). This avoids cross-validation complexity while respecting temporal ordering. LightGBM uses early stopping (patience=50) to prevent overfitting.

**Artifacts saved to `data/processed/`**:

| File | Description |
|------|-------------|
| `metrics.json` | MAE, RMSE, R² for all 4 models |
| `lgbm_best_params.json` | Best hyperparameters from Optuna |
| `lgbm_feature_importance.json` | Feature importance (gain) |
| `naive.joblib`, `lr.joblib`, `lgbm.joblib`, `xgb.joblib` | Serialized models |

### Monitoring and drift detection (Phase 7)

The monitoring module tracks model performance over time, detects data drift, and generates Evidently AI reports.

```bash
# Run drift analysis (requires predictions in the database)
python -m src.monitoring --model lgbm --json

# Generate Evidently HTML reports
python -m src.monitoring --model lgbm --report

# Look back 14 days instead of default 7
python -m src.monitoring --model lgbm --since 14
```

**Drift detection methods:**

| Method | Description | Alert threshold |
|--------|-------------|-----------------|
| Rolling MAE | Sliding window MAE over time (24h window) | MAE > baseline * 1.20 |
| PSI | Population Stability Index between predicted/actual | PSI > 0.2 = significant |
| KS Test | Kolmogorov-Smirnov two-sample test | p-value < 0.05 |

**Exit codes:** `0` = no alert, `1` = MAE degradation > 20%. This enables integration with cron/CI for automated retraining triggers.

**Predictions table** (`analytics.predictions`): stores model predictions alongside actuals for retroactive comparison. Schema in `sql/005_predictions_table.sql`.

**Evidently reports** are saved to `reports/` (gitignored). Includes data drift and regression performance presets.

### Interactive dashboard (Phase 8)

A Streamlit multi-page dashboard for exploring data and model results interactively:

```bash
streamlit run src/dashboard/app.py
```

**Pages:**

| Page | Description |
|------|-------------|
| Availability Timeline | Line chart of bike availability per station with date range filter |
| Station Heatmap | Geographic scatter map (OpenStreetMap) with hour-of-day slider |
| Peak Usage Hours | Weekday vs weekend grouped bar chart + weekday x hour heatmap |
| Model Performance | Metrics comparison table, actual vs predicted scatter, error distribution, per-hour MAE, feature importance |
| Drift Monitor | 3-layer drift monitoring: executive (drift gauge + rolling MAE), analytical (feature ranking by PSI), diagnostic (per-feature distribution overlay) |
| Anomaly Detection | Stuck station detection + Isolation Forest outliers with interactive map, activity scatter, and score distribution |

All data transformation logic lives in `src/dashboard/data.py` (pure functions, no Streamlit dependency) and is fully unit-tested. Caching is applied at the app layer via `@st.cache_data`.

### Prediction API (Phase 9)

A FastAPI REST endpoint for real-time bike availability predictions and anomaly detection:

```bash
# Start the API server
uvicorn src.api.main:app --reload --port 8000

# Interactive docs
open http://localhost:8000/docs
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | System status, model loaded, station count |
| `GET` | `/stations` | List all active stations with metadata |
| `GET` | `/predict?station_id=X` | Predict bikes available at t+15 min |
| `POST` | `/predict/batch` | Batch prediction for multiple stations |
| `GET` | `/anomalies` | Detect anomalous stations (stuck + statistical) |

**Example:**

```bash
# Single prediction
curl "http://localhost:8000/predict?station_id=123"

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"station_ids": ["123", "456", "789"]}'

# Anomaly detection (custom thresholds)
curl "http://localhost:8000/anomalies?stuck_hours=3&contamination=0.1"
```

### Anomaly detection (Phase 9)

Two complementary approaches for identifying malfunctioning or unusual stations:

| Method | Description | Use case |
|--------|-------------|----------|
| **Rule-based** | Flags stations with zero change in `num_bikes_available` for >2 hours | Detect stuck/offline stations |
| **Isolation Forest** | Unsupervised outlier detection on station activity features | Identify unusual patterns vs fleet-wide norms |

**Station activity features** used by Isolation Forest: `avg_bikes`, `std_bikes`, `zero_pct`, `change_rate`, `avg_docks`, `capacity`.

Anomaly flags are stored in `analytics.anomalies` (schema in `sql/006_anomalies_table.sql`).

## Data Source

This project uses the public [GBFS feed](https://gbfs.org/) from **Bike Itau (Bike Sampa)**, operated by Tembici in Sao Paulo, Brazil.

- **240 stations** across the city
- **~5,680 vehicles** (bikes, e-bikes, and scooters)
- **No authentication required** — open data
- **30-second refresh rate**

## Known Limitations & Next Steps

This project was developed as a portfolio piece demonstrating the full ML lifecycle. During analysis, the following insights were identified:

### Feature Dominance

`num_bikes_available` (the current value) dominates feature importance (~1000x more than any other feature). Since the target is `y = bikes at t+15 min`, and **89% of 15-minute intervals show no change**, the model effectively learns an identity function. This is reflected in the Naive Baseline achieving R²=0.957.

### Planned Improvements (v1.1)

| Improvement | Impact | Description |
|-------------|--------|-------------|
| **Increase prediction horizon** | High | Predict at t+1h or t+2h instead of t+15 min — harder problem, more operationally useful |
| **Remove current value from features** | High | Drop `num_bikes_available` from features, forcing the model to learn from lags and temporal patterns |
| **Binary classification** | Medium | Predict `P(bikes=0)` in 30 min — actionable for rebalancing alerts |
| **Predict demand (delta)** | Medium | Target `y = bikes(t+1h) - bikes(t)` — focuses on change, not level |
| **Live data integration** | Medium | Dashboard reads from PostgreSQL directly instead of static Parquet |
| **Automated retraining** | Low | GitHub Action to periodically re-run dataset + model pipeline |

These insights demonstrate the iterative nature of ML development: identifying when a model is "too easy" is as important as building the initial pipeline.

## Contributing

This project follows [Conventional Commits](./COMMIT_CONVENTION.md). See the issue templates in `.github/ISSUE_TEMPLATE/` for task and bug report formats.

## License

This project is licensed under the [MIT License](./LICENSE).
