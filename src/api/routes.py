"""FastAPI route definitions for the prediction API.

All routes are defined on an :class:`~fastapi.APIRouter` and mounted in
:mod:`src.api.main`.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.anomaly.detector import analyze_anomalies
from src.api.schemas import (
    AnomaliesResponse,
    AnomalyInfo,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
    StationInfo,
    StationsResponse,
)
from src.dataset.features import FEATURE_COLS

logger = logging.getLogger(__name__)

router = APIRouter()

# -------------------------------------------------------------------------
# Module-level state (loaded once at startup via lifespan)
# -------------------------------------------------------------------------

_state: dict = {
    "model": None,
    "model_name": "lgbm",
    "station_info": {},  # station_id -> dict
    "latest_data": None,  # pd.DataFrame
    "version": "0.9.0",
}

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"


def load_model(model_name: str = "lgbm") -> None:
    """Load a serialized model from data/processed/."""
    model_path = PROCESSED_DIR / f"{model_name}.joblib"
    if not model_path.exists():
        logger.warning("Model file not found: %s", model_path)
        return

    _state["model"] = joblib.load(model_path)
    _state["model_name"] = model_name
    logger.info("Loaded model: %s", model_path)


def load_station_info() -> None:
    """Load station metadata from sample JSON."""
    path = SAMPLES_DIR / "station_information.json"
    if not path.exists():
        logger.warning("Station info not found: %s", path)
        return

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    stations = data.get("data", {}).get("stations", [])
    _state["station_info"] = {
        str(s["station_id"]): {
            "station_id": str(s["station_id"]),
            "name": s.get("name", str(s["station_id"])),
            "lat": s.get("lat", 0.0),
            "lon": s.get("lon", 0.0),
            "capacity": s.get("capacity", 0),
        }
        for s in stations
    }
    logger.info("Loaded %d station records", len(_state["station_info"]))


def load_latest_data() -> None:
    """Load test split as proxy for latest station data."""
    path = PROCESSED_DIR / "test.parquet"
    if not path.exists():
        logger.warning("Test data not found: %s", path)
        return

    _state["latest_data"] = pd.read_parquet(path)
    logger.info("Loaded latest data: %d rows", len(_state["latest_data"]))


def startup() -> None:
    """Initialize all state at application startup."""
    load_model()
    load_station_info()
    load_latest_data()


# -------------------------------------------------------------------------
# GET /health
# -------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="System health check",
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=_state["model"] is not None,
        stations_available=len(_state["station_info"]),
        version=_state["version"],
    )


# -------------------------------------------------------------------------
# GET /stations
# -------------------------------------------------------------------------


@router.get(
    "/stations",
    response_model=StationsResponse,
    tags=["Stations"],
    summary="List all active stations",
)
def list_stations() -> StationsResponse:
    stations = [StationInfo(**info) for info in _state["station_info"].values()]
    return StationsResponse(count=len(stations), stations=stations)


# -------------------------------------------------------------------------
# GET /predict
# -------------------------------------------------------------------------


def _predict_single(station_id: str) -> PredictionResponse:
    """Generate prediction for one station."""
    model = _state["model"]
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python -m src.model` first.",
        )

    df = _state["latest_data"]
    if df is None:
        raise HTTPException(
            status_code=503,
            detail="No data available. Run `python -m src.dataset` first.",
        )

    station_df = df[df["station_id"] == station_id]
    if station_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Station '{station_id}' not found in data.",
        )

    # Use last available row for the station
    last_row = station_df.iloc[[-1]]

    features = last_row[FEATURE_COLS]
    prediction = float(model.predict(features)[0])
    prediction = max(0.0, round(prediction, 2))

    now = datetime.now(tz=timezone.utc)
    target_time = now + timedelta(minutes=15)

    return PredictionResponse(
        station_id=station_id,
        predicted_bikes=prediction,
        target_time=target_time,
        model_name=_state["model_name"],
    )


@router.get(
    "/predict",
    response_model=PredictionResponse,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["Predictions"],
    summary="Predict bike availability for a station (t+15 min)",
)
def predict(
    station_id: str = Query(..., description="Station ID to predict for"),
) -> PredictionResponse:
    return _predict_single(station_id)


# -------------------------------------------------------------------------
# POST /predict/batch
# -------------------------------------------------------------------------


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    responses={503: {"model": ErrorResponse}},
    tags=["Predictions"],
    summary="Batch prediction for multiple stations",
)
def predict_batch(body: BatchPredictionRequest) -> BatchPredictionResponse:
    predictions: list[PredictionResponse] = []
    for sid in body.station_ids:
        try:
            predictions.append(_predict_single(sid))
        except HTTPException:
            continue  # skip unknown stations
    return BatchPredictionResponse(predictions=predictions)


# -------------------------------------------------------------------------
# GET /anomalies
# -------------------------------------------------------------------------


@router.get(
    "/anomalies",
    response_model=AnomaliesResponse,
    tags=["Anomalies"],
    summary="Detect anomalous stations",
)
def detect_anomalies(
    stuck_hours: float = Query(
        2.0, ge=0.5, le=24.0, description="Stuck threshold in hours"
    ),
    contamination: float = Query(
        0.05,
        ge=0.01,
        le=0.50,
        description="Isolation Forest contamination (expected anomaly fraction)",
    ),
) -> AnomaliesResponse:
    df = _state["latest_data"]
    if df is None:
        raise HTTPException(
            status_code=503,
            detail="No data available. Run `python -m src.dataset` first.",
        )

    results = analyze_anomalies(
        df,
        stuck_threshold=timedelta(hours=stuck_hours),
        contamination=contamination,
    )

    anomalies = [AnomalyInfo(**r.to_dict()) for r in results]

    return AnomaliesResponse(count=len(anomalies), anomalies=anomalies)
