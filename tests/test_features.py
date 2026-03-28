"""Tests for src.dataset.features — feature engineering pipeline."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.dataset.features import (
    FEATURE_COLS,
    TARGET_COL,
    add_lag_features,
    add_rolling_features,
    add_station_features,
    add_target,
    add_temporal_features,
    build_features,
)


def _make_resampled_df(
    station_id: str = "1",
    n_points: int = 50,
) -> pd.DataFrame:
    """Create a synthetic resampled DataFrame at 15-min intervals."""
    base = datetime(2026, 3, 1, 8, 0, tzinfo=timezone.utc)
    timestamps = pd.date_range(base, periods=n_points, freq="15min")

    rng = np.random.default_rng(42)
    bikes = rng.integers(0, 30, size=n_points)

    return pd.DataFrame(
        {
            "station_id": station_id,
            "timestamp": timestamps,
            "num_bikes_available": bikes,
            "num_docks_available": 60 - bikes,
            "num_bikes_disabled": 0,
            "num_docks_disabled": 0,
            "is_renting": True,
            "is_returning": True,
            "capacity": 60,
            "lat": -23.57,
            "lon": -46.69,
        }
    )


# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------


class TestAddLagFeatures:
    """Tests for add_lag_features()."""

    def test_creates_lag_columns(self) -> None:
        df = _make_resampled_df()
        result = add_lag_features(df.copy())
        for lag in [1, 2, 3, 4]:
            assert f"bikes_lag_{lag}" in result.columns

    def test_lag_values_are_correct(self) -> None:
        df = _make_resampled_df().sort_values(["station_id", "timestamp"])
        result = add_lag_features(df.copy())
        bikes = result["num_bikes_available"].values
        lag1 = result["bikes_lag_1"].values
        # lag_1[i] should equal bikes[i-1] (for the same station)
        assert np.isnan(lag1[0])
        np.testing.assert_array_equal(lag1[1:], bikes[:-1])

    def test_no_future_leakage_in_lags(self) -> None:
        df = _make_resampled_df().sort_values(["station_id", "timestamp"])
        result = add_lag_features(df.copy())
        # lag_4 at index 3 should be NaN (not enough history)
        assert pd.isna(result["bikes_lag_4"].iloc[3])


# ---------------------------------------------------------------------------
# Rolling features
# ---------------------------------------------------------------------------


class TestAddRollingFeatures:
    """Tests for add_rolling_features()."""

    def test_creates_rolling_columns(self) -> None:
        df = _make_resampled_df()
        result = add_rolling_features(df.copy())
        assert "bikes_rolling_mean_1h" in result.columns
        assert "bikes_rolling_std_1h" in result.columns

    def test_rolling_uses_past_only(self) -> None:
        """Rolling window should not include the current row (shift(1))."""
        df = _make_resampled_df(n_points=20).sort_values(
            ["station_id", "timestamp"]
        )
        result = add_rolling_features(df.copy())
        # First 5 rows should have NaN rolling mean (shift + 4-period window)
        assert pd.isna(result["bikes_rolling_mean_1h"].iloc[0])


# ---------------------------------------------------------------------------
# Temporal features
# ---------------------------------------------------------------------------


class TestAddTemporalFeatures:
    """Tests for add_temporal_features()."""

    def test_creates_temporal_columns(self) -> None:
        df = _make_resampled_df()
        result = add_temporal_features(df.copy())
        assert "hour" in result.columns
        assert "weekday" in result.columns
        assert "is_weekend" in result.columns
        assert "month" in result.columns

    def test_hour_values_correct(self) -> None:
        df = _make_resampled_df()
        result = add_temporal_features(df.copy())
        expected_hour = pd.to_datetime(result["timestamp"]).dt.hour
        pd.testing.assert_series_equal(
            result["hour"], expected_hour, check_names=False
        )

    def test_is_weekend_values(self) -> None:
        df = _make_resampled_df()
        result = add_temporal_features(df.copy())
        assert result["is_weekend"].isin([0, 1]).all()


# ---------------------------------------------------------------------------
# Station features
# ---------------------------------------------------------------------------


class TestAddStationFeatures:
    """Tests for add_station_features()."""

    def test_passthrough_when_present(self) -> None:
        df = _make_resampled_df()
        result = add_station_features(df.copy())
        assert "capacity" in result.columns

    def test_raises_on_missing_column(self) -> None:
        df = _make_resampled_df().drop(columns=["capacity"])
        with pytest.raises(KeyError, match="capacity"):
            add_station_features(df)


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------


class TestAddTarget:
    """Tests for add_target()."""

    def test_creates_target_column(self) -> None:
        df = _make_resampled_df()
        result = add_target(df.copy())
        assert TARGET_COL in result.columns

    def test_target_is_next_bikes_value(self) -> None:
        df = _make_resampled_df().sort_values(["station_id", "timestamp"])
        result = add_target(df.copy())
        bikes = result["num_bikes_available"].values
        target = result[TARGET_COL].values
        # target[i] should equal bikes[i+1] for the same station
        np.testing.assert_array_equal(target[:-1], bikes[1:])

    def test_last_row_target_is_nan(self) -> None:
        df = _make_resampled_df().sort_values(["station_id", "timestamp"])
        result = add_target(df.copy())
        assert pd.isna(result[TARGET_COL].iloc[-1])


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class TestBuildFeatures:
    """Tests for the complete build_features() pipeline."""

    def test_no_nan_in_output(self) -> None:
        df = _make_resampled_df(n_points=50)
        result = build_features(df)
        assert result[FEATURE_COLS + [TARGET_COL]].isna().sum().sum() == 0

    def test_all_feature_columns_present(self) -> None:
        df = _make_resampled_df(n_points=50)
        result = build_features(df)
        for col in FEATURE_COLS:
            assert col in result.columns
        assert TARGET_COL in result.columns

    def test_output_shorter_than_input(self) -> None:
        """Rows are dropped at edges due to lag/rolling/target NaN."""
        df = _make_resampled_df(n_points=50)
        result = build_features(df)
        assert len(result) < len(df)

    def test_multiple_stations(self) -> None:
        df1 = _make_resampled_df(station_id="1", n_points=50)
        df2 = _make_resampled_df(station_id="2", n_points=50)
        df = pd.concat([df1, df2], ignore_index=True)
        result = build_features(df)
        assert result["station_id"].nunique() == 2
        assert result[FEATURE_COLS + [TARGET_COL]].isna().sum().sum() == 0
