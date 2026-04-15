"""Anomaly detection for bike-sharing stations.

Two detection strategies:

1. **Rule-based** — flag stations with no change in availability for
   an extended period (stuck stations, possible malfunction).
2. **Statistical** — Isolation Forest on station activity patterns
   to identify unusual behavior compared to fleet-wide norms.

All functions are pure (no DB/IO). Persistence helpers live in
:mod:`src.anomaly.store`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Data structures
# -------------------------------------------------------------------------

STUCK_THRESHOLD = timedelta(hours=2)
"""Default duration after which a station with zero change is flagged."""


@dataclass(frozen=True)
class StuckStationResult:
    """Result of rule-based stuck-station detection."""

    station_id: str
    last_change: pd.Timestamp | None
    duration_hours: float
    last_bikes: int
    is_renting: bool | None

    @property
    def flagged(self) -> bool:
        return self.duration_hours >= STUCK_THRESHOLD.total_seconds() / 3600


@dataclass(frozen=True)
class AnomalyResult:
    """Combined anomaly result for a station."""

    station_id: str
    is_stuck: bool
    is_statistical_outlier: bool
    stuck_duration_hours: float
    isolation_score: float

    @property
    def is_anomalous(self) -> bool:
        return self.is_stuck or self.is_statistical_outlier

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        return {
            "station_id": self.station_id,
            "is_stuck": self.is_stuck,
            "is_statistical_outlier": self.is_statistical_outlier,
            "stuck_duration_hours": round(self.stuck_duration_hours, 2),
            "isolation_score": round(self.isolation_score, 4),
            "is_anomalous": self.is_anomalous,
        }


# -------------------------------------------------------------------------
# Rule-based: stuck station detection
# -------------------------------------------------------------------------


def detect_stuck_stations(
    df: pd.DataFrame,
    threshold: timedelta = STUCK_THRESHOLD,
) -> list[StuckStationResult]:
    """Detect stations with no change in ``num_bikes_available``.

    Parameters
    ----------
    df
        Must contain ``station_id``, ``timestamp``, ``num_bikes_available``.
        Should be sorted by ``(station_id, timestamp)``.
    threshold
        Duration of no-change after which a station is flagged.

    Returns
    -------
    list[StuckStationResult]
        One entry per station that has been stuck for >= *threshold*.
    """
    if df.empty:
        return []

    required = {"station_id", "timestamp", "num_bikes_available"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.sort_values(["station_id", "timestamp"]).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    results: list[StuckStationResult] = []
    threshold_hours = threshold.total_seconds() / 3600

    for sid, group in df.groupby("station_id"):
        bikes = group["num_bikes_available"].values
        timestamps = group["timestamp"].values

        # Find last index where value changed
        changes = np.where(np.diff(bikes) != 0)[0]

        if len(changes) == 0:
            # Never changed in the window
            duration = (timestamps[-1] - timestamps[0]) / np.timedelta64(1, "h")
            last_change = None
        else:
            last_change_idx = changes[-1] + 1
            last_change_ts = timestamps[last_change_idx]
            duration = (timestamps[-1] - last_change_ts) / np.timedelta64(1, "h")
            last_change = pd.Timestamp(last_change_ts)

        is_renting = (
            bool(group["is_renting"].iloc[-1])
            if "is_renting" in group.columns
            else None
        )

        duration_float = float(duration)
        if duration_float >= threshold_hours:
            results.append(
                StuckStationResult(
                    station_id=str(sid),
                    last_change=last_change,
                    duration_hours=duration_float,
                    last_bikes=int(bikes[-1]),
                    is_renting=is_renting,
                )
            )

    logger.info(
        "Stuck station detection: %d/%d stations flagged (threshold=%.1fh)",
        len(results),
        df["station_id"].nunique(),
        threshold_hours,
    )
    return results


# -------------------------------------------------------------------------
# Statistical: Isolation Forest
# -------------------------------------------------------------------------


def build_station_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-station activity features for anomaly detection.

    Features computed:
    - ``avg_bikes``: mean availability
    - ``std_bikes``: standard deviation of availability
    - ``zero_pct``: fraction of observations with 0 bikes
    - ``change_rate``: fraction of consecutive intervals with a change
    - ``avg_docks``: mean dock availability
    - ``capacity``: station capacity

    Parameters
    ----------
    df
        Must contain ``station_id``, ``num_bikes_available``.

    Returns
    -------
    pd.DataFrame
        One row per station, indexed by ``station_id``.
    """
    if df.empty:
        return pd.DataFrame()

    stats: list[dict] = []
    for sid, group in df.groupby("station_id"):
        bikes = group["num_bikes_available"].values
        n = len(bikes)

        changes = int(np.sum(np.diff(bikes) != 0)) if n > 1 else 0

        row: dict = {
            "station_id": str(sid),
            "avg_bikes": float(np.mean(bikes)),
            "std_bikes": float(np.std(bikes)) if n > 1 else 0.0,
            "zero_pct": float(np.sum(bikes == 0) / n),
            "change_rate": changes / (n - 1) if n > 1 else 0.0,
        }

        if "num_docks_available" in group.columns:
            row["avg_docks"] = float(group["num_docks_available"].mean())
        if "capacity" in group.columns:
            row["capacity"] = float(group["capacity"].iloc[0])

        stats.append(row)

    return pd.DataFrame(stats).set_index("station_id")


def detect_statistical_anomalies(
    station_features: pd.DataFrame,
    *,
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run Isolation Forest on station-level features.

    Parameters
    ----------
    station_features
        Output of :func:`build_station_features`. Must have numeric columns.
    contamination
        Expected proportion of anomalies (default 5%).
    random_state
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Same index as input with added columns:
        ``anomaly_label`` (-1=outlier, 1=inlier) and
        ``anomaly_score`` (lower = more anomalous).
    """
    if station_features.empty or len(station_features) < 5:
        result = station_features.copy()
        result["anomaly_label"] = 1
        result["anomaly_score"] = 0.0
        return result

    numeric = station_features.select_dtypes(include=[np.number])

    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
    )
    labels = model.fit_predict(numeric)
    scores = model.decision_function(numeric)

    result = station_features.copy()
    result["anomaly_label"] = labels
    result["anomaly_score"] = scores
    return result


# -------------------------------------------------------------------------
# Combined analysis
# -------------------------------------------------------------------------


def analyze_anomalies(
    df: pd.DataFrame,
    *,
    stuck_threshold: timedelta = STUCK_THRESHOLD,
    contamination: float = 0.05,
    random_state: int = 42,
) -> list[AnomalyResult]:
    """Run both rule-based and statistical anomaly detection.

    Parameters
    ----------
    df
        Station status data with ``station_id``, ``timestamp``,
        ``num_bikes_available`` (and optionally ``num_docks_available``,
        ``capacity``, ``is_renting``).
    stuck_threshold
        Duration for stuck-station rule.
    contamination
        Isolation Forest contamination parameter.
    random_state
        Random seed.

    Returns
    -------
    list[AnomalyResult]
        One entry per station that is anomalous (stuck or statistical outlier).
    """
    # Rule-based
    stuck = detect_stuck_stations(df, threshold=stuck_threshold)
    stuck_map = {r.station_id: r for r in stuck}

    # Statistical
    features = build_station_features(df)
    iso_results = detect_statistical_anomalies(
        features,
        contamination=contamination,
        random_state=random_state,
    )

    # Combine
    results: list[AnomalyResult] = []
    all_stations = set(features.index) | set(stuck_map.keys())

    for sid in sorted(all_stations):
        stuck_result = stuck_map.get(sid)
        is_stuck = stuck_result is not None

        is_outlier = False
        iso_score = 0.0
        if sid in iso_results.index:
            row = iso_results.loc[sid]
            is_outlier = bool(row["anomaly_label"] == -1)
            iso_score = float(row["anomaly_score"])

        if is_stuck or is_outlier:
            results.append(
                AnomalyResult(
                    station_id=str(sid),
                    is_stuck=is_stuck,
                    is_statistical_outlier=is_outlier,
                    stuck_duration_hours=(
                        stuck_result.duration_hours if stuck_result else 0.0
                    ),
                    isolation_score=iso_score,
                )
            )

    logger.info(
        "Anomaly analysis: %d anomalous stations (%d stuck, %d statistical outliers)",
        len(results),
        sum(1 for r in results if r.is_stuck),
        sum(1 for r in results if r.is_statistical_outlier),
    )
    return results
