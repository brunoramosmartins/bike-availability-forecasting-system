"""Tests for src.dataset.resampler — time-series resampling logic."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.dataset.resampler import resample_all, resample_station


def _make_raw_df(
    station_id: str = "1",
    n_points: int = 20,
    freq_minutes: int = 5,
) -> pd.DataFrame:
    """Create a synthetic raw station DataFrame."""
    base = datetime(2026, 3, 1, 8, 0, tzinfo=timezone.utc)
    timestamps = pd.date_range(base, periods=n_points, freq=f"{freq_minutes}min")

    return pd.DataFrame(
        {
            "station_id": station_id,
            "last_reported": timestamps,
            "num_bikes_available": range(10, 10 + n_points),
            "num_docks_available": range(50, 50 - n_points, -1),
            "num_bikes_disabled": 0,
            "num_docks_disabled": 0,
            "is_renting": True,
            "is_returning": True,
            "capacity": 60,
            "lat": -23.57,
            "lon": -46.69,
        }
    )


class TestResampleStation:
    """Tests for resample_station()."""

    def test_output_has_15min_frequency(self) -> None:
        raw = _make_raw_df(n_points=30, freq_minutes=5)
        result = resample_station(raw)
        diffs = result.index.to_series().diff().dropna()
        assert (diffs == pd.Timedelta(minutes=15)).all()

    def test_output_has_expected_columns(self) -> None:
        raw = _make_raw_df(n_points=20, freq_minutes=5)
        result = resample_station(raw)
        assert "num_bikes_available" in result.columns
        assert "capacity" in result.columns
        assert "is_renting" in result.columns

    def test_index_named_timestamp(self) -> None:
        raw = _make_raw_df(n_points=20, freq_minutes=5)
        result = resample_station(raw)
        assert result.index.name == "timestamp"

    def test_forward_fill_within_limit(self) -> None:
        """Gaps of up to 4 intervals (1 hour) should be filled."""
        raw = _make_raw_df(n_points=10, freq_minutes=15)
        # Remove some rows to create a gap of 2 intervals (30 min)
        raw = raw.drop([2, 3]).reset_index(drop=True)
        result = resample_station(raw)
        assert not result["num_bikes_available"].isna().any()

    def test_no_empty_output(self) -> None:
        raw = _make_raw_df(n_points=20, freq_minutes=5)
        result = resample_station(raw)
        assert len(result) > 0


class TestResampleAll:
    """Tests for resample_all()."""

    def test_multiple_stations(self) -> None:
        df1 = _make_raw_df(station_id="1", n_points=20, freq_minutes=5)
        df2 = _make_raw_df(station_id="2", n_points=20, freq_minutes=5)
        raw = pd.concat([df1, df2], ignore_index=True)

        result = resample_all(raw)

        assert result["station_id"].nunique() == 2
        assert "timestamp" in result.columns

    def test_returns_empty_for_empty_input(self) -> None:
        empty = pd.DataFrame(
            columns=[
                "station_id",
                "last_reported",
                "num_bikes_available",
                "num_docks_available",
                "num_bikes_disabled",
                "num_docks_disabled",
                "is_renting",
                "is_returning",
                "capacity",
                "lat",
                "lon",
            ]
        )
        result = resample_all(empty)
        assert len(result) == 0

    def test_output_has_station_id_column(self) -> None:
        raw = _make_raw_df(n_points=20)
        result = resample_all(raw)
        assert "station_id" in result.columns
