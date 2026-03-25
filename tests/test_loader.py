"""Tests for src.ingestion.loader — database upsert operations.

These tests verify SQL generation and cursor interaction without requiring
a live database. psycopg2 connections and cursors are fully mocked.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.ingestion.loader import upsert_station_information, upsert_station_status


def _make_mock_conn() -> MagicMock:
    """Create a mock psycopg2 connection with a context-managed cursor."""
    conn = MagicMock()
    cursor = MagicMock()
    cursor.rowcount = 1
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn, cursor


# ---------------------------------------------------------------------------
# upsert_station_status
# ---------------------------------------------------------------------------


class TestUpsertStationStatus:
    """Tests for upsert_station_status()."""

    def test_returns_zero_for_empty_records(self) -> None:
        conn, _ = _make_mock_conn()
        assert upsert_station_status(conn, []) == 0

    def test_executes_insert_for_each_record(self) -> None:
        conn, cursor = _make_mock_conn()
        records = [
            {
                "station_id": "1",
                "num_bikes_available": 20,
                "num_docks_available": 63,
                "num_bikes_disabled": 0,
                "num_docks_disabled": 0,
                "last_reported": datetime(2026, 3, 22, tzinfo=timezone.utc),
                "is_renting": True,
                "is_returning": True,
                "status": "IN_SERVICE",
                "ingestion_timestamp": datetime(2026, 3, 22, tzinfo=timezone.utc),
            },
        ]

        result = upsert_station_status(conn, records)

        assert cursor.execute.call_count == 1
        sql = cursor.execute.call_args[0][0]
        assert "INSERT INTO raw_station_status" in sql
        assert "ON CONFLICT" in sql
        assert "DO NOTHING" in sql
        assert result == 1

    def test_uses_parameterized_query(self) -> None:
        conn, cursor = _make_mock_conn()
        records = [
            {
                "station_id": "5",
                "num_bikes_available": 10,
                "num_docks_available": 5,
                "num_bikes_disabled": 1,
                "num_docks_disabled": 0,
                "last_reported": datetime(2026, 3, 22, tzinfo=timezone.utc),
                "is_renting": True,
                "is_returning": False,
                "status": "IN_SERVICE",
                "ingestion_timestamp": datetime(2026, 3, 22, tzinfo=timezone.utc),
            },
        ]

        upsert_station_status(conn, records)

        _, params = cursor.execute.call_args[0]
        assert params["station_id"] == "5"
        assert params["num_bikes_available"] == 10

    def test_commits_after_insert(self) -> None:
        conn, cursor = _make_mock_conn()
        records = [
            {
                "station_id": "1",
                "num_bikes_available": 0,
                "num_docks_available": 10,
                "num_bikes_disabled": 0,
                "num_docks_disabled": 0,
                "last_reported": datetime(2026, 3, 22, tzinfo=timezone.utc),
                "is_renting": True,
                "is_returning": True,
                "status": "IN_SERVICE",
                "ingestion_timestamp": datetime(2026, 3, 22, tzinfo=timezone.utc),
            },
        ]

        upsert_station_status(conn, records)

        conn.commit.assert_called_once()


# ---------------------------------------------------------------------------
# upsert_station_information
# ---------------------------------------------------------------------------


class TestUpsertStationInformation:
    """Tests for upsert_station_information()."""

    def test_returns_zero_for_empty_records(self) -> None:
        conn, _ = _make_mock_conn()
        assert upsert_station_information(conn, []) == 0

    def test_executes_upsert_for_each_record(self) -> None:
        conn, cursor = _make_mock_conn()
        records = [
            {
                "station_id": "1",
                "name": "Largo da Batata",
                "lat": -23.5668,
                "lon": -46.6937,
                "capacity": 83,
                "address": "Av. Faria Lima",
                "groups": ["G3 - Pinheiros/Jardins"],
            },
        ]

        result = upsert_station_information(conn, records)

        assert cursor.execute.call_count == 1
        sql = cursor.execute.call_args[0][0]
        assert "INSERT INTO station_information" in sql
        assert "ON CONFLICT (station_id) DO UPDATE" in sql
        assert result == 1

    def test_uses_parameterized_query(self) -> None:
        conn, cursor = _make_mock_conn()
        records = [
            {
                "station_id": "7",
                "name": "Test Station",
                "lat": -23.55,
                "lon": -46.63,
                "capacity": 20,
                "address": None,
                "groups": [],
            },
        ]

        upsert_station_information(conn, records)

        _, params = cursor.execute.call_args[0]
        assert params["station_id"] == "7"
        assert params["name"] == "Test Station"

    def test_commits_after_upsert(self) -> None:
        conn, cursor = _make_mock_conn()
        records = [
            {
                "station_id": "1",
                "name": "Station 1",
                "lat": -23.5,
                "lon": -46.6,
                "capacity": 10,
                "address": "Rua X",
                "groups": [],
            },
        ]

        upsert_station_information(conn, records)

        conn.commit.assert_called_once()
