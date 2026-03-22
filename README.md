# Bike Availability Forecasting System

An end-to-end data engineering and machine learning system that forecasts bike availability in urban bike-sharing stations using real-time [GBFS](https://gbfs.org/) data from **Bike Itaú / Bike Sampa** (São Paulo, Brazil).

## Overview

This project builds a continuous data pipeline that ingests high-frequency station data, constructs a time-series dataset, and trains predictive models to estimate future bike availability (t+15 min). It also includes monitoring, anomaly detection, and analytical dashboards to support urban mobility insights.

### Key Features

- **Real-time ingestion** — Scheduled collection of GBFS station data every 5 minutes via GitHub Actions
- **Structured storage** — PostgreSQL (Neon) with raw and processed layers
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
  ┌─────────────┬─────────────┐
  │ raw_status  │ station_info│
  └──────┬──────┴──────┬──────┘
         │             │
    ┌────▼────┐   ┌────▼────┐
    │   ML    │   │  Tableau │
    │Pipeline │   │Dashboard│
    └────┬────┘   └─────────┘
         │
    ┌────▼────┐
    │Monitoring│
    │Evidently │
    └─────────┘
```

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
├── sql/                 # Database DDL and migrations
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
