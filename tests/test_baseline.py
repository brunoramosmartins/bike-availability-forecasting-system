"""Tests for src.model — baseline models and evaluation metrics."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.dataset.features import FEATURE_COLS, TARGET_COL
from src.model.baseline import LinearRegressionModel, NaiveBaseline
from src.model.evaluate import compute_metrics, per_hour_metrics, per_station_metrics


def _make_ml_df(
    n_rows: int = 100,
    n_stations: int = 2,
) -> pd.DataFrame:
    """Create a synthetic ML-ready DataFrame for testing."""
    rng = np.random.default_rng(42)

    base = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
    rows_per_station = n_rows // n_stations
    frames = []

    for sid in range(1, n_stations + 1):
        timestamps = pd.date_range(base, periods=rows_per_station, freq="15min")
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
                    "lat": -23.57,
                    "lon": -46.69,
                    TARGET_COL: np.roll(bikes, -1),
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# NaiveBaseline
# ---------------------------------------------------------------------------


class TestNaiveBaseline:
    """Tests for NaiveBaseline."""

    def test_fit_returns_self(self) -> None:
        df = _make_ml_df()
        model = NaiveBaseline()
        result = model.fit(df[FEATURE_COLS], df[TARGET_COL])
        assert result is model

    def test_predict_shape(self) -> None:
        df = _make_ml_df()
        model = NaiveBaseline().fit(df[FEATURE_COLS], df[TARGET_COL])
        preds = model.predict(df[FEATURE_COLS])
        assert len(preds) == len(df)

    def test_predict_equals_lag_1(self) -> None:
        df = _make_ml_df()
        model = NaiveBaseline().fit(df[FEATURE_COLS], df[TARGET_COL])
        preds = model.predict(df[FEATURE_COLS])
        np.testing.assert_array_equal(preds, df["bikes_lag_1"].values)

    def test_predict_returns_ndarray(self) -> None:
        df = _make_ml_df()
        model = NaiveBaseline().fit(df[FEATURE_COLS], df[TARGET_COL])
        preds = model.predict(df[FEATURE_COLS])
        assert isinstance(preds, np.ndarray)


# ---------------------------------------------------------------------------
# LinearRegressionModel
# ---------------------------------------------------------------------------


class TestLinearRegressionModel:
    """Tests for LinearRegressionModel."""

    def test_fit_returns_self(self) -> None:
        df = _make_ml_df()
        model = LinearRegressionModel()
        result = model.fit(df[FEATURE_COLS], df[TARGET_COL])
        assert result is model

    def test_predict_shape(self) -> None:
        df = _make_ml_df()
        model = LinearRegressionModel().fit(df[FEATURE_COLS], df[TARGET_COL])
        preds = model.predict(df[FEATURE_COLS])
        assert len(preds) == len(df)

    def test_predict_returns_ndarray(self) -> None:
        df = _make_ml_df()
        model = LinearRegressionModel().fit(df[FEATURE_COLS], df[TARGET_COL])
        preds = model.predict(df[FEATURE_COLS])
        assert isinstance(preds, np.ndarray)

    def test_predict_not_constant(self) -> None:
        df = _make_ml_df()
        model = LinearRegressionModel().fit(df[FEATURE_COLS], df[TARGET_COL])
        preds = model.predict(df[FEATURE_COLS])
        assert np.std(preds) > 0


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    """Tests for compute_metrics()."""

    def test_returns_expected_keys(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        result = compute_metrics(y, y + 0.5)
        assert set(result.keys()) == {"mae", "rmse", "r2"}

    def test_values_are_float(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        result = compute_metrics(y, y + 0.1)
        for v in result.values():
            assert isinstance(v, float)

    def test_perfect_prediction(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_metrics(y, y)
        assert result["mae"] == 0.0
        assert result["rmse"] == 0.0
        assert result["r2"] == 1.0

    def test_mae_is_nonnegative(self) -> None:
        rng = np.random.default_rng(42)
        y_true = rng.normal(10, 3, 100)
        y_pred = rng.normal(10, 3, 100)
        result = compute_metrics(y_true, y_pred)
        assert result["mae"] >= 0
        assert result["rmse"] >= 0


# ---------------------------------------------------------------------------
# per_station_metrics
# ---------------------------------------------------------------------------


class TestPerStationMetrics:
    """Tests for per_station_metrics()."""

    def test_returns_dataframe(self) -> None:
        df = _make_ml_df()
        df["y_pred"] = df[TARGET_COL] + 0.5
        result = per_station_metrics(df, TARGET_COL, "y_pred")
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_station(self) -> None:
        df = _make_ml_df(n_stations=3)
        df["y_pred"] = df[TARGET_COL] + 0.5
        result = per_station_metrics(df, TARGET_COL, "y_pred")
        assert len(result) == 3

    def test_columns_present(self) -> None:
        df = _make_ml_df()
        df["y_pred"] = df[TARGET_COL] + 0.5
        result = per_station_metrics(df, TARGET_COL, "y_pred")
        assert set(result.columns) == {"mae", "rmse", "r2"}


# ---------------------------------------------------------------------------
# per_hour_metrics
# ---------------------------------------------------------------------------


class TestPerHourMetrics:
    """Tests for per_hour_metrics()."""

    def test_returns_dataframe(self) -> None:
        df = _make_ml_df()
        df["y_pred"] = df[TARGET_COL] + 0.5
        result = per_hour_metrics(df, TARGET_COL, "y_pred")
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_hour(self) -> None:
        df = _make_ml_df(n_rows=200)
        df["y_pred"] = df[TARGET_COL] + 0.5
        result = per_hour_metrics(df, TARGET_COL, "y_pred")
        unique_hours = df["hour"].nunique()
        assert len(result) == unique_hours

    def test_columns_present(self) -> None:
        df = _make_ml_df()
        df["y_pred"] = df[TARGET_COL] + 0.5
        result = per_hour_metrics(df, TARGET_COL, "y_pred")
        assert set(result.columns) == {"mae", "rmse", "r2"}
