"""Tests for src.dashboard.data — pure data loading and transformation helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.dashboard.data import (
    compute_aggregate_drift,
    compute_feature_drift_df,
    compute_hourly_availability,
    compute_rolling_mae_series,
    compute_station_summary,
    compute_weekday_hour_heatmap,
    filter_by_stations,
    generate_predictions,
    load_feature_importance,
    load_metrics,
    load_station_names,
)
from src.dataset.features import FEATURE_COLS, TARGET_COL
from src.model.baseline import NaiveBaseline


def _make_df(n_rows: int = 200, n_stations: int = 3) -> pd.DataFrame:
    """Create a synthetic DataFrame resembling Parquet output."""
    rng = np.random.default_rng(42)
    rows_per_station = n_rows // n_stations
    frames = []

    for sid in range(1, n_stations + 1):
        timestamps = pd.date_range(
            datetime(2026, 3, 1, tzinfo=timezone.utc),
            periods=rows_per_station,
            freq="15min",
        )
        bikes = rng.integers(0, 30, size=rows_per_station).astype(float)
        frames.append(
            pd.DataFrame(
                {
                    "station_id": str(sid),
                    "timestamp": timestamps,
                    "num_bikes_available": bikes,
                    "num_docks_available": 60 - bikes,
                    "bikes_lag_1": np.roll(bikes, 1),
                    "bikes_lag_2": np.roll(bikes, 2),
                    "bikes_lag_3": np.roll(bikes, 3),
                    "bikes_lag_4": np.roll(bikes, 4),
                    "bikes_rolling_mean_1h": bikes + rng.normal(0, 1, rows_per_station),
                    "bikes_rolling_std_1h": np.abs(rng.normal(2, 1, rows_per_station)),
                    "hour": timestamps.hour,
                    "weekday": timestamps.weekday,
                    "is_weekend": (timestamps.weekday >= 5).astype(int),
                    "month": timestamps.month,
                    "capacity": 60,
                    "lat": -23.57 + sid * 0.01,
                    "lon": -46.69 + sid * 0.01,
                    TARGET_COL: np.roll(bikes, -1),
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# load_station_names
# ---------------------------------------------------------------------------


class TestLoadStationNames:
    """Tests for load_station_names()."""

    def test_returns_dict(self, tmp_path: Path) -> None:
        data = {
            "data": {
                "stations": [
                    {"station_id": "1", "name": "Station A"},
                    {"station_id": "2", "name": "Station B"},
                ]
            }
        }
        (tmp_path / "station_information.json").write_text(
            json.dumps(data), encoding="utf-8"
        )
        result = load_station_names(tmp_path)
        assert result == {"1": "Station A", "2": "Station B"}

    def test_returns_empty_when_missing(self, tmp_path: Path) -> None:
        result = load_station_names(tmp_path)
        assert result == {}


# ---------------------------------------------------------------------------
# load_metrics
# ---------------------------------------------------------------------------


class TestLoadMetrics:
    """Tests for load_metrics()."""

    def test_loads_correctly(self, tmp_path: Path) -> None:
        metrics = {"lgbm": {"mae": 0.15, "rmse": 0.5, "r2": 0.98}}
        (tmp_path / "metrics.json").write_text(json.dumps(metrics))
        result = load_metrics(tmp_path)
        assert result == metrics

    def test_returns_empty_when_missing(self, tmp_path: Path) -> None:
        result = load_metrics(tmp_path)
        assert result == {}


# ---------------------------------------------------------------------------
# load_feature_importance
# ---------------------------------------------------------------------------


class TestLoadFeatureImportance:
    """Tests for load_feature_importance()."""

    def test_returns_dataframe(self, tmp_path: Path) -> None:
        data = [
            {"feature": "bikes_lag_1", "importance": 100},
            {"feature": "hour", "importance": 50},
        ]
        (tmp_path / "lgbm_feature_importance.json").write_text(json.dumps(data))
        result = load_feature_importance(tmp_path)
        assert isinstance(result, pd.DataFrame)
        assert "feature" in result.columns

    def test_returns_empty_when_missing(self, tmp_path: Path) -> None:
        result = load_feature_importance(tmp_path)
        assert result.empty


# ---------------------------------------------------------------------------
# filter_by_stations
# ---------------------------------------------------------------------------


class TestFilterByStations:
    """Tests for filter_by_stations()."""

    def test_empty_list_returns_all(self) -> None:
        df = _make_df()
        result = filter_by_stations(df, [])
        assert len(result) == len(df)

    def test_filters_correctly(self) -> None:
        df = _make_df()
        result = filter_by_stations(df, ["1"])
        assert all(result["station_id"] == "1")

    def test_nonexistent_id_returns_empty(self) -> None:
        df = _make_df()
        result = filter_by_stations(df, ["999"])
        assert result.empty


# ---------------------------------------------------------------------------
# compute_hourly_availability
# ---------------------------------------------------------------------------


class TestComputeHourlyAvailability:
    """Tests for compute_hourly_availability()."""

    def test_has_required_columns(self) -> None:
        df = _make_df()
        result = compute_hourly_availability(df)
        assert "hour" in result.columns
        assert "day_type" in result.columns
        assert "avg_bikes" in result.columns

    def test_day_types(self) -> None:
        df = _make_df()
        result = compute_hourly_availability(df)
        assert set(result["day_type"]).issubset({"Weekday", "Weekend"})


# ---------------------------------------------------------------------------
# compute_station_summary
# ---------------------------------------------------------------------------


class TestComputeStationSummary:
    """Tests for compute_station_summary()."""

    def test_one_row_per_station(self) -> None:
        df = _make_df(n_stations=3)
        result = compute_station_summary(df)
        assert len(result) == 3

    def test_has_required_columns(self) -> None:
        df = _make_df()
        result = compute_station_summary(df)
        for col in ["station_id", "avg_bikes", "lat", "lon", "capacity", "fill_pct"]:
            assert col in result.columns


# ---------------------------------------------------------------------------
# compute_weekday_hour_heatmap
# ---------------------------------------------------------------------------


class TestComputeWeekdayHourHeatmap:
    """Tests for compute_weekday_hour_heatmap()."""

    def test_returns_pivot_table(self) -> None:
        df = _make_df()
        result = compute_weekday_hour_heatmap(df)
        assert isinstance(result, pd.DataFrame)
        assert result.index.name == "weekday"


# ---------------------------------------------------------------------------
# generate_predictions
# ---------------------------------------------------------------------------


class TestGeneratePredictions:
    """Tests for generate_predictions()."""

    def test_returns_ndarray(self, tmp_path: Path) -> None:
        df = _make_df()
        model = NaiveBaseline().fit(df[FEATURE_COLS], df[TARGET_COL])
        model_path = tmp_path / "naive.joblib"
        joblib.dump(model, model_path)

        preds = generate_predictions(df, model_path)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(df)


# ---------------------------------------------------------------------------
# compute_feature_drift_df
# ---------------------------------------------------------------------------


class TestComputeFeatureDriftDf:
    """Tests for compute_feature_drift_df()."""

    def test_returns_dataframe(self) -> None:
        ref = _make_df(n_rows=300, n_stations=3)
        cur = _make_df(n_rows=300, n_stations=3)
        result = compute_feature_drift_df(ref, cur)
        assert isinstance(result, pd.DataFrame)
        assert "feature" in result.columns
        assert "psi" in result.columns
        assert "drifted" in result.columns

    def test_one_row_per_feature(self) -> None:
        ref = _make_df(n_rows=300, n_stations=3)
        cur = _make_df(n_rows=300, n_stations=3)
        features = ["hour", "weekday", "num_bikes_available"]
        result = compute_feature_drift_df(ref, cur, features=features)
        assert len(result) == len(features)

    def test_empty_when_insufficient_data(self) -> None:
        ref = pd.DataFrame({"hour": [1.0]})
        cur = pd.DataFrame({"hour": [2.0]})
        result = compute_feature_drift_df(ref, cur, features=["hour"])
        assert result.empty


# ---------------------------------------------------------------------------
# compute_aggregate_drift
# ---------------------------------------------------------------------------


class TestComputeAggregateDrift:
    """Tests for compute_aggregate_drift()."""

    def test_returns_expected_keys(self) -> None:
        ref = _make_df(n_rows=300, n_stations=3)
        cur = _make_df(n_rows=300, n_stations=3)
        result = compute_aggregate_drift(ref, cur)
        for key in ["drift_score", "n_features", "n_drifted", "avg_psi", "max_psi"]:
            assert key in result

    def test_drift_score_between_zero_and_one(self) -> None:
        ref = _make_df(n_rows=300, n_stations=3)
        cur = _make_df(n_rows=300, n_stations=3)
        result = compute_aggregate_drift(ref, cur)
        assert 0.0 <= result["drift_score"] <= 1.0


# ---------------------------------------------------------------------------
# compute_rolling_mae_series
# ---------------------------------------------------------------------------


class TestComputeRollingMaeSeries:
    """Tests for compute_rolling_mae_series()."""

    def test_returns_series(self) -> None:
        y_true = pd.Series(np.arange(100, dtype=float))
        y_pred = pd.Series(np.arange(100, dtype=float) + 0.5)
        result = compute_rolling_mae_series(y_true, y_pred, window=10)
        assert isinstance(result, pd.Series)
        assert len(result) == 100

    def test_first_values_are_nan(self) -> None:
        y_true = pd.Series(np.arange(50, dtype=float))
        y_pred = pd.Series(np.arange(50, dtype=float))
        result = compute_rolling_mae_series(y_true, y_pred, window=10)
        assert result.iloc[:9].isna().all()
