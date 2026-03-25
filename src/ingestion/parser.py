"""GBFS response parser and field validator.

Extracts station records from nested JSON payloads and validates that all
required fields are present before passing data downstream.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

REQUIRED_STATUS_FIELDS = {
    "station_id",
    "num_bikes_available",
    "num_docks_available",
    "last_reported",
    "is_renting",
    "is_returning",
    "status",
}

REQUIRED_INFO_FIELDS = {
    "station_id",
    "name",
    "lat",
    "lon",
    "capacity",
}


class ParseError(Exception):
    """Raised when the GBFS payload is malformed or missing required data."""


def _extract_stations(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the stations list from a GBFS response envelope.

    Args:
        payload: Raw JSON response from the GBFS API.

    Returns:
        List of station dictionaries.

    Raises:
        ParseError: If the expected nested structure is missing.
    """
    try:
        return payload["data"]["stations"]
    except (KeyError, TypeError) as exc:
        raise ParseError("Payload missing expected 'data.stations' structure") from exc


def _validate_fields(
    record: dict[str, Any],
    required: set[str],
) -> bool:
    """Check that a record contains all required fields with non-None values."""
    return all(record.get(field) is not None for field in required)


def parse_station_status(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse and validate station status records.

    Each valid record is enriched with an ``ingestion_timestamp`` in UTC.
    Records missing required fields are logged and skipped.

    Args:
        payload: Raw JSON response from ``station_status.json``.

    Returns:
        List of validated and enriched station-status dictionaries.

    Raises:
        ParseError: If the payload structure is invalid.
    """
    stations = _extract_stations(payload)
    now = datetime.now(timezone.utc)
    results: list[dict[str, Any]] = []

    for record in stations:
        if not _validate_fields(record, REQUIRED_STATUS_FIELDS):
            logger.warning(
                "Skipping station_status record with missing fields: %s",
                record.get("station_id", "unknown"),
            )
            continue

        results.append(
            {
                "station_id": str(record["station_id"]),
                "num_bikes_available": int(record["num_bikes_available"]),
                "num_docks_available": int(record["num_docks_available"]),
                "num_bikes_disabled": int(record.get("num_bikes_disabled", 0)),
                "num_docks_disabled": int(record.get("num_docks_disabled", 0)),
                "last_reported": datetime.fromtimestamp(
                    record["last_reported"], tz=timezone.utc
                ),
                "is_renting": bool(record["is_renting"]),
                "is_returning": bool(record["is_returning"]),
                "status": str(record["status"]),
                "ingestion_timestamp": now,
            }
        )

    logger.info("Parsed %d/%d station_status records", len(results), len(stations))
    return results


def parse_station_information(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Parse and validate station information records.

    Args:
        payload: Raw JSON response from ``station_information.json``.

    Returns:
        List of validated station-information dictionaries.

    Raises:
        ParseError: If the payload structure is invalid.
    """
    stations = _extract_stations(payload)
    results: list[dict[str, Any]] = []

    for record in stations:
        if not _validate_fields(record, REQUIRED_INFO_FIELDS):
            logger.warning(
                "Skipping station_information record with missing fields: %s",
                record.get("station_id", "unknown"),
            )
            continue

        results.append(
            {
                "station_id": str(record["station_id"]),
                "name": str(record["name"]),
                "lat": float(record["lat"]),
                "lon": float(record["lon"]),
                "capacity": int(record["capacity"]),
                "address": record.get("address"),
                "groups": record.get("groups", []),
            }
        )

    logger.info(
        "Parsed %d/%d station_information records",
        len(results),
        len(stations),
    )
    return results
