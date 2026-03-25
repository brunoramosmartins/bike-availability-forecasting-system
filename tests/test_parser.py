"""Tests for src.ingestion.parser — GBFS response parsing and validation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.ingestion.parser import (
    ParseError,
    parse_station_information,
    parse_station_status,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_STATUS_PAYLOAD = {
    "last_updated": 1774467136,
    "ttl": 30,
    "data": {
        "stations": [
            {
                "station_id": "1",
                "num_bikes_available": 20,
                "num_docks_available": 63,
                "num_bikes_disabled": 0,
                "num_docks_disabled": 0,
                "last_reported": 1774467106,
                "is_renting": True,
                "is_returning": True,
                "status": "IN_SERVICE",
            },
            {
                "station_id": "3",
                "num_bikes_available": 7,
                "num_docks_available": 8,
                "num_bikes_disabled": 0,
                "num_docks_disabled": 0,
                "last_reported": 1774467000,
                "is_renting": True,
                "is_returning": True,
                "status": "IN_SERVICE",
            },
        ]
    },
}

VALID_INFO_PAYLOAD = {
    "last_updated": 1774467135,
    "ttl": 29,
    "data": {
        "stations": [
            {
                "station_id": "1",
                "name": "1 - Largo da Batata",
                "lat": -23.5668,
                "lon": -46.6937,
                "capacity": 83,
                "address": "Av. Brigadeiro Faria Lima",
                "groups": ["G3 - Pinheiros/Jardins"],
            },
        ]
    },
}


# ---------------------------------------------------------------------------
# parse_station_status
# ---------------------------------------------------------------------------


class TestParseStationStatus:
    """Tests for parse_station_status()."""

    def test_extracts_all_records(self) -> None:
        records = parse_station_status(VALID_STATUS_PAYLOAD)
        assert len(records) == 2

    def test_includes_required_fields(self) -> None:
        record = parse_station_status(VALID_STATUS_PAYLOAD)[0]
        expected_keys = {
            "station_id",
            "num_bikes_available",
            "num_docks_available",
            "num_bikes_disabled",
            "num_docks_disabled",
            "last_reported",
            "is_renting",
            "is_returning",
            "status",
            "ingestion_timestamp",
        }
        assert set(record.keys()) == expected_keys

    def test_station_id_is_string(self) -> None:
        record = parse_station_status(VALID_STATUS_PAYLOAD)[0]
        assert isinstance(record["station_id"], str)

    def test_last_reported_is_utc_datetime(self) -> None:
        record = parse_station_status(VALID_STATUS_PAYLOAD)[0]
        assert isinstance(record["last_reported"], datetime)
        assert record["last_reported"].tzinfo == timezone.utc

    def test_ingestion_timestamp_is_utc(self) -> None:
        record = parse_station_status(VALID_STATUS_PAYLOAD)[0]
        assert isinstance(record["ingestion_timestamp"], datetime)
        assert record["ingestion_timestamp"].tzinfo == timezone.utc

    def test_skips_records_with_missing_fields(self) -> None:
        payload = {
            "data": {
                "stations": [
                    {"station_id": "99"},  # missing most fields
                    VALID_STATUS_PAYLOAD["data"]["stations"][0],
                ]
            }
        }
        records = parse_station_status(payload)
        assert len(records) == 1
        assert records[0]["station_id"] == "1"

    def test_raises_parse_error_on_bad_structure(self) -> None:
        with pytest.raises(ParseError, match="data.stations"):
            parse_station_status({"bad": "payload"})

    def test_defaults_disabled_to_zero(self) -> None:
        payload = {
            "data": {
                "stations": [
                    {
                        "station_id": "5",
                        "num_bikes_available": 10,
                        "num_docks_available": 5,
                        "last_reported": 1774467106,
                        "is_renting": True,
                        "is_returning": True,
                        "status": "IN_SERVICE",
                    },
                ]
            }
        }
        record = parse_station_status(payload)[0]
        assert record["num_bikes_disabled"] == 0
        assert record["num_docks_disabled"] == 0


# ---------------------------------------------------------------------------
# parse_station_information
# ---------------------------------------------------------------------------


class TestParseStationInformation:
    """Tests for parse_station_information()."""

    def test_extracts_all_records(self) -> None:
        records = parse_station_information(VALID_INFO_PAYLOAD)
        assert len(records) == 1

    def test_includes_required_fields(self) -> None:
        record = parse_station_information(VALID_INFO_PAYLOAD)[0]
        expected_keys = {
            "station_id",
            "name",
            "lat",
            "lon",
            "capacity",
            "address",
            "groups",
        }
        assert set(record.keys()) == expected_keys

    def test_lat_lon_are_floats(self) -> None:
        record = parse_station_information(VALID_INFO_PAYLOAD)[0]
        assert isinstance(record["lat"], float)
        assert isinstance(record["lon"], float)

    def test_groups_is_list(self) -> None:
        record = parse_station_information(VALID_INFO_PAYLOAD)[0]
        assert isinstance(record["groups"], list)

    def test_skips_records_with_missing_fields(self) -> None:
        payload = {
            "data": {
                "stations": [
                    {"station_id": "99"},  # missing name, lat, lon, capacity
                    VALID_INFO_PAYLOAD["data"]["stations"][0],
                ]
            }
        }
        records = parse_station_information(payload)
        assert len(records) == 1

    def test_raises_parse_error_on_bad_structure(self) -> None:
        with pytest.raises(ParseError, match="data.stations"):
            parse_station_information({})

    def test_handles_missing_optional_fields(self) -> None:
        payload = {
            "data": {
                "stations": [
                    {
                        "station_id": "7",
                        "name": "Test Station",
                        "lat": -23.55,
                        "lon": -46.63,
                        "capacity": 20,
                    },
                ]
            }
        }
        record = parse_station_information(payload)[0]
        assert record["address"] is None
        assert record["groups"] == []
