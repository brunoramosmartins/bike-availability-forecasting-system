# Bike Availability Forecasting System

An end-to-end data engineering and machine learning system that forecasts bike availability in urban bike-sharing stations using real-time [GBFS](https://gbfs.org/) data from **Bike Itaú / Bike Sampa** (São Paulo, Brazil).

## Overview

This project builds a continuous data pipeline that ingests high-frequency station data, constructs a time-series dataset, and trains predictive models to estimate future bike availability (t+15 min). It also includes monitoring, anomaly detection, and analytical dashboards to support urban mobility insights.

### Key Features

- **Real-time ingestion** — Scheduled collection of GBFS station data every 5 minutes via GitHub Actions
- **Structured storage** — PostgreSQL (Neon) with raw tables and a curated **`analytics`** layer for ML and BI
- **ML forecasting** — From naive baselines to gradient boosting (LightGBM)
- **Model monitoring** — Drift detection and performance tracking with Evidently AI
- **Visualization** — Interactive dashboards on Tableau Public
- **Prediction API** — FastAPI endpoint for real-time availability forecasts

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
  │ raw_station_status│ station_information│   ← raw / ingestion layer
  └─────────┬────────┴──────────┬─────────┘
            │                   │
            └─────────┬─────────┘
                      ▼
            ┌─────────────────────┐
            │  schema: analytics   │   ← curated views + DQ metrics
            │  station_status_*    │
            └─────────┬───────────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
    ML pipeline   Tableau     Monitoring
   (Phase 4+)   (Phase 8)   (Phase 7+)
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
│   ├── ingestion/       # GBFS data collection pipeline
│   ├── storage/         # Database connection and schema
│   ├── dataset/         # Feature engineering and splitting
│   ├── model/           # ML training, evaluation, prediction
│   ├── monitoring/      # Drift detection and reporting
│   └── api/             # FastAPI prediction endpoint
├── tests/               # Unit and integration tests
├── notebooks/           # Exploratory analysis and model comparison
├── sql/                 # Database DDL and migrations (ordered 001_, 002_, …)
├── docs/analytics/      # Analytics layer docs (ERD, dictionary, DQ)
├── config/              # Environment-based configuration
└── .github/workflows/   # CI and scheduled ingestion
```

## Getting Started

### Prerequisites

- Python 3.10+
- PostgreSQL database ([Neon free tier](https://neon.tech/) recommended)
- GitHub account (for Actions-based scheduling)

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/bike-availability-forecasting-system.git
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
