"""Pydantic request / response models for the prediction API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

# -------------------------------------------------------------------------
# Health
# -------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """GET /health response."""

    status: str = Field(..., examples=["ok"])
    model_loaded: bool
    stations_available: int
    version: str


# -------------------------------------------------------------------------
# Stations
# -------------------------------------------------------------------------


class StationInfo(BaseModel):
    """Single station metadata."""

    station_id: str
    name: str
    lat: float
    lon: float
    capacity: int


class StationsResponse(BaseModel):
    """GET /stations response."""

    count: int
    stations: list[StationInfo]


# -------------------------------------------------------------------------
# Prediction
# -------------------------------------------------------------------------


class PredictionResponse(BaseModel):
    """GET /predict response."""

    station_id: str
    predicted_bikes: float = Field(
        ..., description="Predicted num_bikes_available at target_time"
    )
    target_time: datetime = Field(..., description="Prediction horizon (t + 15 min)")
    model_name: str


class BatchPredictionRequest(BaseModel):
    """POST /predict/batch request body."""

    station_ids: list[str] = Field(
        ..., min_length=1, max_length=50, description="Station IDs to predict"
    )


class BatchPredictionResponse(BaseModel):
    """POST /predict/batch response."""

    predictions: list[PredictionResponse]


# -------------------------------------------------------------------------
# Anomalies
# -------------------------------------------------------------------------


class AnomalyInfo(BaseModel):
    """Single anomaly record."""

    station_id: str
    is_stuck: bool
    is_statistical_outlier: bool
    stuck_duration_hours: float
    isolation_score: float
    is_anomalous: bool


class AnomaliesResponse(BaseModel):
    """GET /anomalies response."""

    count: int
    anomalies: list[AnomalyInfo]


# -------------------------------------------------------------------------
# Error
# -------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    detail: str
