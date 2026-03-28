"""Resample raw station status data to consistent 15-minute intervals.

Reads from the ``analytics.station_status_enriched`` view and produces a
DataFrame with one row per (station_id, timestamp) at 15-minute granularity.
Missing intervals are forward-filled up to a configurable maximum gap.
"""

from __future__ import annotations

import logging

import pandas as pd
from psycopg2.extensions import connection as PgConnection

logger = logging.getLogger(__name__)

RESAMPLE_FREQ = "15min"
MAX_FFILL_PERIODS = 4  # 4 × 15 min = 1 hour
MIN_DAYS = 7  # drop stations with fewer days of data


def load_raw_status(conn: PgConnection, min_days: int = MIN_DAYS) -> pd.DataFrame:
    """Load enriched station status from the database.

    Args:
        conn: An open PostgreSQL connection.
        min_days: Minimum days of data a station must have to be included.

    Returns:
        DataFrame with columns from ``analytics.station_status_enriched``,
        indexed by ``last_reported`` and sorted chronologically.
    """
    query = """
        SELECT
            station_id,
            last_reported,
            num_bikes_available,
            num_docks_available,
            num_bikes_disabled,
            num_docks_disabled,
            is_renting,
            is_returning,
            capacity,
            lat,
            lon
        FROM analytics.station_status_enriched
        ORDER BY station_id, last_reported
    """
    df = pd.read_sql(query, conn, parse_dates=["last_reported"])

    if df.empty:
        logger.warning("No data returned from analytics.station_status_enriched")
        return df

    # Filter stations with insufficient history
    span = df.groupby("station_id")["last_reported"].agg(
        lambda s: (s.max() - s.min()).days
    )
    valid_stations = span[span >= min_days].index
    n_dropped = len(span) - len(valid_stations)
    if n_dropped > 0:
        logger.info(
            "Dropped %d stations with < %d days of data", n_dropped, min_days
        )
    df = df[df["station_id"].isin(valid_stations)].copy()

    logger.info(
        "Loaded %d rows for %d stations", len(df), df["station_id"].nunique()
    )
    return df


def resample_station(group: pd.DataFrame) -> pd.DataFrame:
    """Resample a single station's data to 15-minute intervals.

    Numeric columns are averaged within each bin; boolean columns use the
    last observed value.  Gaps up to ``MAX_FFILL_PERIODS`` intervals are
    forward-filled; remaining NaN rows are dropped.

    Args:
        group: DataFrame for one station, with ``last_reported`` as a column.

    Returns:
        Resampled DataFrame with a ``timestamp`` DatetimeIndex.
    """
    df = group.set_index("last_reported").sort_index()

    numeric_cols = [
        "num_bikes_available",
        "num_docks_available",
        "num_bikes_disabled",
        "num_docks_disabled",
    ]
    bool_cols = ["is_renting", "is_returning"]
    static_cols = ["capacity", "lat", "lon"]

    resampled = df[numeric_cols].resample(RESAMPLE_FREQ).mean()
    for col in bool_cols:
        resampled[col] = df[col].resample(RESAMPLE_FREQ).last()
    for col in static_cols:
        resampled[col] = df[col].resample(RESAMPLE_FREQ).last()

    resampled = resampled.ffill(limit=MAX_FFILL_PERIODS)
    resampled = resampled.dropna(subset=numeric_cols)
    resampled.index.name = "timestamp"

    return resampled


def resample_all(df: pd.DataFrame) -> pd.DataFrame:
    """Resample all stations to 15-minute intervals.

    Args:
        df: Raw enriched DataFrame from :func:`load_raw_status`.

    Returns:
        DataFrame with ``station_id`` as a column and ``timestamp`` as the
        index, resampled to 15-minute intervals.
    """
    if df.empty:
        return df

    frames: list[pd.DataFrame] = []
    for station_id, group in df.groupby("station_id"):
        resampled = resample_station(group)
        resampled["station_id"] = station_id
        frames.append(resampled)

    result = pd.concat(frames)
    result = result.reset_index()

    logger.info(
        "Resampled to %d rows across %d stations at %s intervals",
        len(result),
        result["station_id"].nunique(),
        RESAMPLE_FREQ,
    )
    return result
