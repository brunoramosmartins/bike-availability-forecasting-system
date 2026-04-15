"""Tests for src.api — FastAPI prediction endpoint."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routes import _state
from src.dataset.features import TARGET_COL

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_data(n_stations: int = 3, n_rows: int = 20) -> pd.DataFrame:
    """Create synthetic test data matching the feature schema."""
    rng = np.random.default_rng(42)
    rows: list[dict] = []
    for i in range(n_stations):
        sid = f"station_{i}"
        for j in range(n_rows):
            rows.append(
                {
                    "station_id": sid,
                    "timestamp": pd.Timestamp("2024-01-01")
                    + pd.Timedelta(minutes=15 * j),
                    "num_bikes_available": rng.integers(0, 20),
                    "num_docks_available": rng.integers(0, 20),
                    "bikes_lag_1": rng.integers(0, 20),
                    "bikes_lag_2": rng.integers(0, 20),
                    "bikes_lag_3": rng.integers(0, 20),
                    "bikes_lag_4": rng.integers(0, 20),
                    "bikes_rolling_mean_1h": rng.uniform(5, 15),
                    "bikes_rolling_std_1h": rng.uniform(0, 5),
                    "hour": j % 24,
                    "weekday": j % 7,
                    "is_weekend": 1 if j % 7 >= 5 else 0,
                    "month": 1,
                    "capacity": 20,
                    "lat": -23.55 + rng.uniform(-0.01, 0.01),
                    "lon": -46.63 + rng.uniform(-0.01, 0.01),
                    TARGET_COL: rng.integers(0, 20),
                }
            )
    return pd.DataFrame(rows)


class _FakeModel:
    """Minimal model that satisfies the duck-typed interface."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), 7.5)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    """Create a TestClient with mocked state."""
    _state["model"] = _FakeModel()
    _state["model_name"] = "test_model"
    _state["station_info"] = {
        "station_0": {
            "station_id": "station_0",
            "name": "Station Zero",
            "lat": -23.55,
            "lon": -46.63,
            "capacity": 20,
        },
        "station_1": {
            "station_id": "station_1",
            "name": "Station One",
            "lat": -23.56,
            "lon": -46.64,
            "capacity": 15,
        },
    }
    _state["latest_data"] = _make_test_data()
    _state["version"] = "0.9.0"

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def client_no_model() -> TestClient:
    """Client with no model loaded."""
    _state["model"] = None
    _state["model_name"] = ""
    _state["station_info"] = {}
    _state["latest_data"] = _make_test_data()
    _state["version"] = "0.9.0"

    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["stations_available"] == 2

    def test_model_not_loaded(self, client_no_model: TestClient) -> None:
        resp = client_no_model.get("/health")
        data = resp.json()
        assert data["model_loaded"] is False


# ---------------------------------------------------------------------------
# GET /stations
# ---------------------------------------------------------------------------


class TestStations:
    def test_returns_stations(self, client: TestClient) -> None:
        resp = client.get("/stations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert len(data["stations"]) == 2
        assert data["stations"][0]["station_id"] in ("station_0", "station_1")

    def test_station_fields(self, client: TestClient) -> None:
        resp = client.get("/stations")
        station = resp.json()["stations"][0]
        for field in ["station_id", "name", "lat", "lon", "capacity"]:
            assert field in station


# ---------------------------------------------------------------------------
# GET /predict
# ---------------------------------------------------------------------------


class TestPredict:
    def test_returns_prediction(self, client: TestClient) -> None:
        resp = client.get("/predict", params={"station_id": "station_0"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["station_id"] == "station_0"
        assert data["predicted_bikes"] == 7.5
        assert data["model_name"] == "test_model"
        assert "target_time" in data

    def test_unknown_station_returns_404(self, client: TestClient) -> None:
        resp = client.get("/predict", params={"station_id": "nonexistent"})
        assert resp.status_code == 404

    def test_no_model_returns_503(self, client_no_model: TestClient) -> None:
        resp = client_no_model.get("/predict", params={"station_id": "station_0"})
        assert resp.status_code == 503

    def test_missing_station_id_returns_422(self, client: TestClient) -> None:
        resp = client.get("/predict")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /predict/batch
# ---------------------------------------------------------------------------


class TestPredictBatch:
    def test_batch_prediction(self, client: TestClient) -> None:
        resp = client.post(
            "/predict/batch",
            json={"station_ids": ["station_0", "station_1"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) == 2

    def test_batch_skips_unknown(self, client: TestClient) -> None:
        resp = client.post(
            "/predict/batch",
            json={"station_ids": ["station_0", "nonexistent"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) == 1

    def test_batch_empty_list_returns_422(self, client: TestClient) -> None:
        resp = client.post("/predict/batch", json={"station_ids": []})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /anomalies
# ---------------------------------------------------------------------------


class TestAnomalies:
    def test_returns_anomalies(self, client: TestClient) -> None:
        resp = client.get("/anomalies")
        assert resp.status_code == 200
        data = resp.json()
        assert "count" in data
        assert "anomalies" in data
        assert isinstance(data["anomalies"], list)

    def test_custom_threshold(self, client: TestClient) -> None:
        resp = client.get(
            "/anomalies", params={"stuck_hours": 0.5, "contamination": 0.1}
        )
        assert resp.status_code == 200

    def test_anomaly_fields(self, client: TestClient) -> None:
        resp = client.get("/anomalies")
        data = resp.json()
        if data["count"] > 0:
            anomaly = data["anomalies"][0]
            for field in [
                "station_id",
                "is_stuck",
                "is_statistical_outlier",
                "stuck_duration_hours",
                "isolation_score",
                "is_anomalous",
            ]:
                assert field in anomaly
