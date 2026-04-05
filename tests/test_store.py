"""Tests for src.monitoring.store — prediction storage functions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd

from src.monitoring.store import (
    backfill_actuals,
    load_baseline_metrics,
    load_predictions,
    save_predictions,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_connection() -> MagicMock:
    """Create a mock PgConnection with cursor context manager."""
    conn = MagicMock()
    cursor = MagicMock()
    cursor.rowcount = 1
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn


# ---------------------------------------------------------------------------
# save_predictions
# ---------------------------------------------------------------------------


class TestSavePredictions:
    """Tests for save_predictions()."""

    def test_returns_zero_for_empty(self) -> None:
        conn = _mock_connection()
        assert save_predictions(conn, []) == 0

    def test_executes_insert(self) -> None:
        conn = _mock_connection()
        records = [
            {
                "station_id": "1",
                "prediction_time": datetime.now(timezone.utc),
                "target_time": datetime.now(timezone.utc),
                "model_name": "lgbm",
                "predicted_value": 5.0,
            }
        ]
        result = save_predictions(conn, records)
        assert result == 1
        conn.commit.assert_called_once()

    def test_multiple_records(self) -> None:
        conn = _mock_connection()
        records = [
            {
                "station_id": str(i),
                "prediction_time": datetime.now(timezone.utc),
                "target_time": datetime.now(timezone.utc),
                "model_name": "lgbm",
                "predicted_value": float(i),
            }
            for i in range(5)
        ]
        result = save_predictions(conn, records)
        assert result == 5


# ---------------------------------------------------------------------------
# backfill_actuals
# ---------------------------------------------------------------------------


class TestBackfillActuals:
    """Tests for backfill_actuals()."""

    def test_returns_zero_for_empty(self) -> None:
        conn = _mock_connection()
        df = pd.DataFrame(columns=["station_id", "target_time", "actual_value"])
        assert backfill_actuals(conn, df) == 0

    def test_executes_update(self) -> None:
        conn = _mock_connection()
        df = pd.DataFrame(
            {
                "station_id": ["1", "2"],
                "target_time": [
                    datetime.now(timezone.utc),
                    datetime.now(timezone.utc),
                ],
                "actual_value": [5.0, 10.0],
            }
        )
        result = backfill_actuals(conn, df)
        assert result == 2
        conn.commit.assert_called_once()


# ---------------------------------------------------------------------------
# load_predictions
# ---------------------------------------------------------------------------


class TestLoadPredictions:
    """Tests for load_predictions()."""

    def test_returns_empty_when_no_rows(self) -> None:
        conn = _mock_connection()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = []

        df = load_predictions(conn, "lgbm")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == [
            "station_id",
            "target_time",
            "predicted_value",
            "actual_value",
        ]

    def test_returns_dataframe_with_rows(self) -> None:
        conn = _mock_connection()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = [
            ("1", datetime.now(timezone.utc), 5.0, 4.8),
            ("2", datetime.now(timezone.utc), 10.0, 9.5),
        ]

        df = load_predictions(conn, "lgbm")
        assert len(df) == 2
        assert "predicted_value" in df.columns
        assert "actual_value" in df.columns

    def test_since_parameter_adds_filter(self) -> None:
        conn = _mock_connection()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = []

        since = datetime.now(timezone.utc)
        load_predictions(conn, "lgbm", since=since)

        # Verify the SQL was called with the since parameter
        call_args = cursor.execute.call_args
        assert len(call_args[0][1]) == 2  # model_name + since


# ---------------------------------------------------------------------------
# load_baseline_metrics
# ---------------------------------------------------------------------------


class TestLoadBaselineMetrics:
    """Tests for load_baseline_metrics()."""

    def test_loads_metrics(self, tmp_path: object) -> None:
        metrics = {"lgbm": {"mae": 0.15, "rmse": 0.5, "r2": 0.98}}
        metrics_file = tmp_path / "metrics.json"  # type: ignore[operator]
        metrics_file.write_text(json.dumps(metrics))

        with patch("src.monitoring.store.DATA_DIR", tmp_path):
            result = load_baseline_metrics()

        assert result == metrics

    def test_raises_when_missing(self, tmp_path: object) -> None:
        import pytest

        with (
            patch("src.monitoring.store.DATA_DIR", tmp_path),
            pytest.raises(FileNotFoundError),
        ):
            load_baseline_metrics()
