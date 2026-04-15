"""Tests for src.anomaly.detector — anomaly detection logic."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from src.anomaly.detector import (
    AnomalyResult,
    analyze_anomalies,
    build_station_features,
    detect_statistical_anomalies,
    detect_stuck_stations,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_station_df(
    station_id: str = "s1",
    n_intervals: int = 20,
    bikes: list[int] | None = None,
    freq_min: int = 15,
) -> pd.DataFrame:
    """Create a synthetic station DataFrame."""
    if bikes is None:
        rng = np.random.default_rng(42)
        bikes = rng.integers(0, 20, size=n_intervals).tolist()

    timestamps = pd.date_range("2024-01-01", periods=len(bikes), freq=f"{freq_min}min")
    return pd.DataFrame(
        {
            "station_id": station_id,
            "timestamp": timestamps,
            "num_bikes_available": bikes,
            "num_docks_available": [20 - b for b in bikes],
            "capacity": 20,
        }
    )


# ---------------------------------------------------------------------------
# detect_stuck_stations
# ---------------------------------------------------------------------------


class TestDetectStuckStations:
    """Tests for rule-based stuck station detection."""

    def test_returns_empty_for_active_station(self) -> None:
        # Different value every interval — never stuck
        df = _make_station_df(bikes=list(range(10)))
        result = detect_stuck_stations(df, threshold=timedelta(hours=1))
        assert result == []

    def test_flags_stuck_station(self) -> None:
        # Same value for 3 hours at 15-min intervals = 12 intervals
        bikes = [5] * 12
        df = _make_station_df(bikes=bikes)
        result = detect_stuck_stations(df, threshold=timedelta(hours=2))
        assert len(result) == 1
        assert result[0].station_id == "s1"
        assert result[0].duration_hours >= 2.0
        assert result[0].flagged is True

    def test_not_flagged_below_threshold(self) -> None:
        bikes = [5] * 4  # 1 hour at 15-min intervals
        df = _make_station_df(bikes=bikes)
        result = detect_stuck_stations(df, threshold=timedelta(hours=2))
        assert result == []

    def test_returns_empty_for_empty_df(self) -> None:
        df = pd.DataFrame(columns=["station_id", "timestamp", "num_bikes_available"])
        result = detect_stuck_stations(df)
        assert result == []

    def test_multiple_stations(self) -> None:
        # Station A: stuck, Station B: active
        stuck = _make_station_df("A", bikes=[10] * 12)
        active = _make_station_df("B", bikes=list(range(12)))
        df = pd.concat([stuck, active], ignore_index=True)
        result = detect_stuck_stations(df, threshold=timedelta(hours=2))
        assert len(result) == 1
        assert result[0].station_id == "A"

    def test_missing_columns_raises(self) -> None:
        df = pd.DataFrame({"station_id": ["s1"], "timestamp": ["2024-01-01"]})
        with pytest.raises(ValueError, match="Missing columns"):
            detect_stuck_stations(df)

    def test_result_has_last_bikes(self) -> None:
        bikes = [5] * 12
        df = _make_station_df(bikes=bikes)
        result = detect_stuck_stations(df, threshold=timedelta(hours=2))
        assert result[0].last_bikes == 5

    def test_stuck_after_activity(self) -> None:
        # Active first, then stuck
        bikes = [1, 2, 3, 4] + [7] * 12
        df = _make_station_df(bikes=bikes)
        result = detect_stuck_stations(df, threshold=timedelta(hours=2))
        assert len(result) == 1
        assert result[0].last_change is not None


# ---------------------------------------------------------------------------
# build_station_features
# ---------------------------------------------------------------------------


class TestBuildStationFeatures:
    """Tests for station feature aggregation."""

    def test_returns_one_row_per_station(self) -> None:
        df1 = _make_station_df("A", n_intervals=20)
        df2 = _make_station_df("B", n_intervals=20)
        df = pd.concat([df1, df2], ignore_index=True)
        result = build_station_features(df)
        assert len(result) == 2
        assert "A" in result.index
        assert "B" in result.index

    def test_expected_columns(self) -> None:
        df = _make_station_df(n_intervals=20)
        result = build_station_features(df)
        for col in ["avg_bikes", "std_bikes", "zero_pct", "change_rate"]:
            assert col in result.columns

    def test_zero_pct_calculation(self) -> None:
        bikes = [0, 0, 0, 5, 5]  # 60% zero
        df = _make_station_df(bikes=bikes)
        result = build_station_features(df)
        assert abs(result.loc["s1", "zero_pct"] - 0.6) < 0.01

    def test_change_rate_all_same(self) -> None:
        bikes = [5] * 10  # no changes
        df = _make_station_df(bikes=bikes)
        result = build_station_features(df)
        assert result.loc["s1", "change_rate"] == 0.0

    def test_change_rate_all_different(self) -> None:
        bikes = list(range(10))  # every interval changes
        df = _make_station_df(bikes=bikes)
        result = build_station_features(df)
        assert result.loc["s1", "change_rate"] == 1.0

    def test_empty_input(self) -> None:
        df = pd.DataFrame()
        result = build_station_features(df)
        assert result.empty


# ---------------------------------------------------------------------------
# detect_statistical_anomalies (Isolation Forest)
# ---------------------------------------------------------------------------


class TestDetectStatisticalAnomalies:
    """Tests for Isolation Forest anomaly detection."""

    def test_returns_labels_and_scores(self) -> None:
        rng = np.random.default_rng(42)
        features = pd.DataFrame(
            {
                "avg_bikes": rng.normal(10, 2, 50),
                "std_bikes": rng.normal(3, 1, 50),
                "zero_pct": rng.uniform(0, 0.1, 50),
                "change_rate": rng.uniform(0.3, 0.8, 50),
            },
            index=[f"s{i}" for i in range(50)],
        )
        result = detect_statistical_anomalies(features)
        assert "anomaly_label" in result.columns
        assert "anomaly_score" in result.columns
        assert set(result["anomaly_label"].unique()).issubset({-1, 1})

    def test_detects_obvious_outlier(self) -> None:
        # 49 normal + 1 extreme outlier
        rng = np.random.default_rng(42)
        features = pd.DataFrame(
            {
                "avg_bikes": np.append(rng.normal(10, 1, 49), [100.0]),
                "std_bikes": np.append(rng.normal(3, 0.5, 49), [50.0]),
                "zero_pct": np.append(rng.uniform(0, 0.05, 49), [0.99]),
                "change_rate": np.append(rng.uniform(0.3, 0.7, 49), [0.0]),
            },
            index=[f"s{i}" for i in range(50)],
        )
        result = detect_statistical_anomalies(features, contamination=0.05)
        # The extreme outlier should be flagged
        assert result.loc["s49", "anomaly_label"] == -1

    def test_handles_small_dataset(self) -> None:
        features = pd.DataFrame(
            {"avg_bikes": [5.0, 6.0], "std_bikes": [1.0, 1.5]},
            index=["s1", "s2"],
        )
        result = detect_statistical_anomalies(features)
        assert len(result) == 2
        # Should fallback to all inliers
        assert (result["anomaly_label"] == 1).all()


# ---------------------------------------------------------------------------
# analyze_anomalies (combined)
# ---------------------------------------------------------------------------


class TestAnalyzeAnomalies:
    """Tests for combined anomaly analysis."""

    def test_returns_list_of_anomaly_results(self) -> None:
        stuck = _make_station_df("stuck", bikes=[5] * 12)
        active = _make_station_df("active", bikes=list(range(12)))
        df = pd.concat([stuck, active], ignore_index=True)
        results = analyze_anomalies(df, stuck_threshold=timedelta(hours=2))
        assert all(isinstance(r, AnomalyResult) for r in results)

    def test_stuck_station_is_anomalous(self) -> None:
        df = _make_station_df("stuck", bikes=[5] * 12)
        results = analyze_anomalies(df, stuck_threshold=timedelta(hours=2))
        stuck_results = [r for r in results if r.station_id == "stuck"]
        assert len(stuck_results) == 1
        assert stuck_results[0].is_stuck is True
        assert stuck_results[0].is_anomalous is True

    def test_to_dict_serialization(self) -> None:
        result = AnomalyResult(
            station_id="s1",
            is_stuck=True,
            is_statistical_outlier=False,
            stuck_duration_hours=3.5,
            isolation_score=0.1,
        )
        d = result.to_dict()
        assert d["station_id"] == "s1"
        assert d["is_anomalous"] is True
        assert isinstance(d["stuck_duration_hours"], float)

    def test_empty_df(self) -> None:
        df = pd.DataFrame(
            columns=[
                "station_id",
                "timestamp",
                "num_bikes_available",
                "num_docks_available",
                "capacity",
            ]
        )
        results = analyze_anomalies(df)
        assert results == []
