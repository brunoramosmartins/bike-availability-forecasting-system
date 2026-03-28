# Bike Availability Forecasting System

[![CI](https://github.com/brunoramosmartins/bike-availability-forecasting-system/actions/workflows/ci.yml/badge.svg)](https://github.com/brunoramosmartins/bike-availability-forecasting-system/actions/workflows/ci.yml)
[![Ingestion](https://github.com/brunoramosmartins/bike-availability-forecasting-system/actions/workflows/ingest.yml/badge.svg)](https://github.com/brunoramosmartins/bike-availability-forecasting-system/actions/workflows/ingest.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![PostgreSQL](https://img.shields.io/badge/database-PostgreSQL%20(Neon)-336791)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

An end-to-end data engineering and machine learning system that forecasts bike availability in urban bike-sharing stations using real-time [GBFS](https://gbfs.org/) data from **Bike Itaú / Bike Sampa** (São Paulo, Brazil).

## Overview

This project builds a continuous data pipeline that ingests high-frequency station data, constructs a time-series dataset, and trains predictive models to estimate future bike availability (t+15 min). It also includes monitoring, anomaly detection, and analytical dashboards to support urban mobility insights.

### Key Features

- **Real-time ingestion** — Scheduled collection of GBFS station data every 5 minutes via GitHub Actions
- **Structured storage** — PostgreSQL (Neon) with raw tables and a curated **`analytics`** layer for ML and BI
- **Data quality** — Automated DQ metric views and CLI checks (`python -m src.storage.data_quality`)
- **Feature engineering** — Lag, rolling, temporal, and station features with leakage-free pipeline
- **ML forecasting** — From naive baselines to gradient boosting (LightGBM) *(planned)*
- **Model monitoring** — Drift detection and performance tracking with Evidently AI *(planned)*
- **Visualization** — Interactive dashboards on Tableau Public *(planned)*
- **Prediction API** — FastAPI endpoint for real-time availability forecasts *(planned)*

### Project Progress

| Phase | Status | Description |
|-------|--------|-------------|
| 0 — Repository Bootstrap | :white_check_mark: Done | Project scaffold, governance files, CI/CD |
| 1 — Data Reconnaissance | :white_check_mark: Done | GBFS feed exploration, schema docs, station map |
| 2 — Data Ingestion Pipeline | :white_check_mark: Done | Fetch → parse → load pipeline, GitHub Actions cron |
| 3 — Data Modeling & Storage | :white_check_mark: Done | Analytics views, indexes, DQ metrics |
| 4 — Dataset Construction | :white_check_mark: Done | Feature engineering, temporal splits, Parquet export |
| 5 — Baseline Modeling | :hourglass: Next | Naive + Linear Regression baselines |
| 6 — Advanced Modeling | :construction: Planned | LightGBM with hyperparameter tuning |
| 7 — Monitoring & Drift | :construction: Planned | Evidently AI reports, alerting thresholds |
| 8 — Visualization | :construction: Planned | Tableau Public dashboards |
| 9 — Extensions | :construction: Planned | FastAPI endpoint, anomaly detection |

## Architecture

```
GitHub Actions (cron: */5 * * * *)
        │
        ▼
  GBFS API (Bike Itaú)
        │
        ▼
  Python Ingestion Service
  fetch → validate → load
        │
        ▼
  PostgreSQL (Neon)
  ┌──────────────────┬────────────────────┐
  │ raw_station_status│ station_information│   ← Raw (Bronze)
  └─────────┬────────┴──────────┬─────────┘
            │                   │
            └─────────┬─────────┘
                      ▼
            ┌─────────────────────┐
            │  schema: analytics   │   ← Curated (Silver)
            │  station_status_*    │
            │  v_dq_* metrics      │
            └─────────┬───────────┘
                      │
                      ▼
            ┌─────────────────────┐
            │  Dataset Pipeline    │   ← Features (Gold)
            │  resample → features │
            │  → split → Parquet   │
            └─────────┬───────────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
    ML Models     Tableau     Monitoring
   (Phase 5+)   (Phase 8)   (Phase 7+)
```

See [docs/analytics/README.md](./docs/analytics/README.md) for grain, ER/layer diagrams (Mermaid), and the data dictionary.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Ingestion | Python, httpx, GitHub Actions |
| Storage | PostgreSQL (Neon free tier) |
| Processing | pandas, SQL |
| Modeling | scikit-learn, LightGBM |
| Monitoring | Evidently AI |
| Visualization | Tableau Public |
| API | FastAPI |

## Project Structure

```
├── src/
│   ├── ingestion/       # GBFS fetch → parse → load pipeline
│   ├── storage/         # DB connection, schema, data quality CLI
│   ├── dataset/         # Resampling, feature engineering, splitting
│   ├── model/           # ML training, evaluation, prediction (Phase 5+)
│   ├── monitoring/      # Drift detection and reporting (Phase 7+)
│   └── api/             # FastAPI prediction endpoint (Phase 9)
├── tests/               # 75 unit tests (pytest)
├── notebooks/
│   ├── 01_data_exploration.ipynb   # GBFS schema, station map (Folium)
│   └── 02_feature_analysis.ipynb   # Feature distributions, leakage check
├── sql/                 # DDL migrations (001_ → 004_)
│   ├── 001_create_tables.sql       # raw_station_status, station_information
│   ├── 002_create_indexes.sql      # Composite index for time-series
│   ├── 003_analytics_layer.sql     # analytics.station_status_enriched/latest
│   └── 004_data_quality_views.sql  # analytics.v_dq_* metric views
├── data/
│   ├── samples/         # Saved GBFS API responses (Phase 1)
│   └── processed/       # train/val/test Parquet files (Phase 4)
├── docs/analytics/      # ER diagrams, data dictionary, DQ glossary
└── .github/workflows/
    ├── ci.yml           # Lint (ruff) + tests (pytest) on PRs
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

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your DATABASE_URL and other settings
```

### Running the Ingestion Pipeline

```bash
python -m src.ingestion
```

Applying migrations is part of that run (all `sql/*.sql` files in lexicographic order).

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
| **4 — Dataset** | `SELECT … FROM analytics.station_status_enriched WHERE …` (time windows per `station_id`) | Stable column contract, no duplicated join logic in Python; explicit grain for resampling and lags. |
| **5–6 — Modeling** | Parquet/CSV extracts or direct SQL from the same view | Features and target definitions stay aligned with documented semantics. |
| **7 — Monitoring** | Compare predictions to actuals joined on `station_id` + time; optional DQ views in scheduled jobs | Shared vocabulary for “what a row means”; DQ metrics reused for operational trust. |
| **8 — Visualization** | Tableau (or extract) against enriched view or aggregates | One semantic layer for “availability over time” and maps (`lat`/`lon` already on the row). |

**Why it matters:** ingestion owns **raw** tables; **analytics** owns the **contract** everyone else reads. That reduces drift between SQL in notebooks, training code, and dashboards, and centralizes joins and naming. Materialized views can be added later if query cost grows; start with plain views for simplicity.

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

**Split**: time-based (train 80%, validation 10%, test 20%) — no shuffle, no temporal overlap.

**Output**: `data/processed/train.parquet`, `val.parquet`, `test.parquet`

## Data Source

This project uses the public [GBFS feed](https://gbfs.org/) from **Bike Itaú (Bike Sampa)**, operated by Tembici in São Paulo, Brazil.

- **240 stations** across the city
- **~5,680 vehicles** (bikes, e-bikes, and scooters)
- **No authentication required** — open data
- **30-second refresh rate**

## Contributing

This project follows [Conventional Commits](./COMMIT_CONVENTION.md). See the issue templates in `.github/ISSUE_TEMPLATE/` for task and bug report formats.

## License

This project is licensed under the [MIT License](./LICENSE).
