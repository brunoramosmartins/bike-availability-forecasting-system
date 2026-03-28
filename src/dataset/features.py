"""Feature engineering for bike availability forecasting.

Transforms resampled 15-minute station data into ML-ready features including
lag values, rolling statistics, temporal encodings, and the prediction target.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET_COL = "y"
LAG_PERIODS = [1, 2, 3, 4]  # 15, 30, 45, 60 minutes
ROLLING_WINDOW = 4  # 4 × 15 min = 1 hour


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged values of ``num_bikes_available``.

    Creates columns ``bikes_lag_1`` through ``bikes_lag_4`` representing
    the availability at t-15, t-30, t-45, and t-60 minutes.

    Args:
        df: DataFrame sorted by (station_id, timestamp).

    Returns:
        DataFrame with lag columns added.
    """
    for lag in LAG_PERIODS:
        df[f"bikes_lag_{lag}"] = df.groupby("station_id")[
            "num_bikes_available"
        ].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling mean and standard deviation of bike availability.

    Window size is ``ROLLING_WINDOW`` periods (1 hour at 15-min intervals).
    Rolling statistics use only past values (``shift(1)`` before rolling) to
    prevent leakage.

    Args:
        df: DataFrame sorted by (station_id, timestamp).

    Returns:
        DataFrame with ``bikes_rolling_mean_1h`` and ``bikes_rolling_std_1h``.
    """
    shifted = df.groupby("station_id")["num_bikes_available"].shift(1)
    rolling = shifted.groupby(df["station_id"]).rolling(
        window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW
    )

    df["bikes_rolling_mean_1h"] = rolling.mean().reset_index(level=0, drop=True)
    df["bikes_rolling_std_1h"] = rolling.std().reset_index(level=0, drop=True)

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal encodings from the timestamp.

    Creates: ``hour``, ``weekday`` (0=Mon), ``is_weekend``, ``month``.

    Args:
        df: DataFrame with a ``timestamp`` column.

    Returns:
        DataFrame with temporal feature columns added.
    """
    ts = pd.to_datetime(df["timestamp"])
    df["hour"] = ts.dt.hour
    df["weekday"] = ts.dt.weekday
    df["is_weekend"] = (ts.dt.weekday >= 5).astype(np.int8)
    df["month"] = ts.dt.month
    return df


def add_station_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure station-level features are present.

    The resampled DataFrame already carries ``capacity``, ``lat``, ``lon``
    from the enriched view.  This function is a pass-through that validates
    their presence.

    Args:
        df: Resampled DataFrame.

    Returns:
        The same DataFrame (unchanged).

    Raises:
        KeyError: If any expected station feature is missing.
    """
    for col in ("capacity", "lat", "lon"):
        if col not in df.columns:
            raise KeyError(f"Expected station feature '{col}' not found in DataFrame")
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create the prediction target: ``num_bikes_available`` at t+15 min.

    The target ``y`` is the next observation of ``num_bikes_available`` for
    the same station.

    Args:
        df: DataFrame sorted by (station_id, timestamp).

    Returns:
        DataFrame with the ``y`` column added.
    """
    df[TARGET_COL] = df.groupby("station_id")["num_bikes_available"].shift(-1)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline.

    Pipeline steps:
        1. Sort by (station_id, timestamp)
        2. Add lag features (t-1 … t-4)
        3. Add rolling mean / std (1-hour window)
        4. Add temporal features (hour, weekday, is_weekend, month)
        5. Validate station features (capacity, lat, lon)
        6. Add target (y = bikes at t+15)
        7. Drop rows with NaN (edges of lag/rolling/target windows)

    Args:
        df: Resampled DataFrame from :func:`~src.dataset.resampler.resample_all`.

    Returns:
        Clean ML-ready DataFrame with no NaN values.
    """
    df = df.sort_values(["station_id", "timestamp"]).copy()

    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_temporal_features(df)
    df = add_station_features(df)
    df = add_target(df)

    n_before = len(df)
    df = df.dropna()
    n_dropped = n_before - len(df)

    logger.info(
        "Feature pipeline complete: %d rows (%d dropped due to NaN edges)",
        len(df),
        n_dropped,
    )
    return df


FEATURE_COLS = [
    "num_bikes_available",
    "num_docks_available",
    "bikes_lag_1",
    "bikes_lag_2",
    "bikes_lag_3",
    "bikes_lag_4",
    "bikes_rolling_mean_1h",
    "bikes_rolling_std_1h",
    "hour",
    "weekday",
    "is_weekend",
    "month",
    "capacity",
    "lat",
    "lon",
]
"""Ordered list of feature column names for model training."""
