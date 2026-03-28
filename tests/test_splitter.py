"""Tests for src.dataset.splitter — time-based dataset splitting."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.dataset.splitter import DatasetSplit, time_based_split


def _make_feature_df(
    n_stations: int = 2,
    n_timestamps: int = 100,
) -> pd.DataFrame:
    """Create a synthetic feature-engineered DataFrame."""
    base = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
    timestamps = pd.date_range(base, periods=n_timestamps, freq="15min")

    rng = np.random.default_rng(42)
    frames = []
    for sid in range(1, n_stations + 1):
        bikes = rng.integers(0, 30, size=n_timestamps).astype(float)
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
                    "bikes_rolling_mean_1h": bikes,
                    "bikes_rolling_std_1h": bikes * 0.1,
                    "hour": timestamps.hour,
                    "weekday": timestamps.weekday,
                    "is_weekend": (timestamps.weekday >= 5).astype(int),
                    "month": timestamps.month,
                    "capacity": 60,
                    "lat": -23.57,
                    "lon": -46.69,
                    "y": np.roll(bikes, -1),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


class TestTimeBasedSplit:
    """Tests for time_based_split()."""

    def test_returns_dataset_split(self) -> None:
        df = _make_feature_df()
        result = time_based_split(df)
        assert isinstance(result, DatasetSplit)

    def test_no_temporal_overlap(self) -> None:
        df = _make_feature_df()
        split = time_based_split(df)
        assert split.train["timestamp"].max() < split.val["timestamp"].min()
        assert split.val["timestamp"].max() < split.test["timestamp"].min()

    def test_all_rows_accounted_for(self) -> None:
        df = _make_feature_df()
        split = time_based_split(df)
        total = len(split.train) + len(split.val) + len(split.test)
        assert total == len(df)

    def test_train_is_largest(self) -> None:
        df = _make_feature_df(n_timestamps=200)
        split = time_based_split(df)
        assert len(split.train) > len(split.val)
        assert len(split.train) > len(split.test)

    def test_temporal_order_preserved(self) -> None:
        df = _make_feature_df()
        split = time_based_split(df)
        for subset in [split.train, split.val, split.test]:
            timestamps = subset["timestamp"].values
            assert (np.diff(timestamps.astype(np.int64)) >= 0).all()

    def test_custom_fractions(self) -> None:
        df = _make_feature_df(n_timestamps=200)
        split = time_based_split(df, train_frac=0.70, val_frac=0.15)
        total = len(split.train) + len(split.val) + len(split.test)
        assert total == len(df)
        # test should be ~30% of timestamps
        assert len(split.test) > 0

    def test_all_stations_can_appear_in_all_splits(self) -> None:
        """Since the split is global (not per-station), all stations should
        appear in all splits when there is enough data."""
        df = _make_feature_df(n_stations=3, n_timestamps=200)
        split = time_based_split(df)
        for subset in [split.train, split.val, split.test]:
            assert subset["station_id"].nunique() == 3
