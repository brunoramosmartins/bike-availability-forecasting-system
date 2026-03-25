"""Database loader for GBFS data.

Performs idempotent inserts into PostgreSQL using parameterized queries.
Station status rows are append-only (``ON CONFLICT … DO NOTHING``).
Station information rows are upserted (``ON CONFLICT … DO UPDATE``).
"""

from __future__ import annotations

import logging
from typing import Any

from psycopg2.extensions import connection as PgConnection

logger = logging.getLogger(__name__)


def upsert_station_status(
    conn: PgConnection,
    records: list[dict[str, Any]],
) -> int:
    """Insert station status records, skipping duplicates.

    Uniqueness is defined by ``(station_id, last_reported)``.

    Args:
        conn: An open PostgreSQL connection.
        records: Parsed station-status dictionaries.

    Returns:
        Number of rows actually inserted (excluding conflicts).
    """
    if not records:
        return 0

    query = """
        INSERT INTO raw_station_status (
            station_id, num_bikes_available, num_docks_available,
            num_bikes_disabled, num_docks_disabled, last_reported,
            is_renting, is_returning, status, ingestion_timestamp
        )
        VALUES (
            %(station_id)s, %(num_bikes_available)s, %(num_docks_available)s,
            %(num_bikes_disabled)s, %(num_docks_disabled)s, %(last_reported)s,
            %(is_renting)s, %(is_returning)s, %(status)s, %(ingestion_timestamp)s
        )
        ON CONFLICT (station_id, last_reported) DO NOTHING
    """

    inserted = 0
    with conn.cursor() as cur:
        for record in records:
            cur.execute(query, record)
            inserted += cur.rowcount
    conn.commit()

    logger.info(
        "Inserted %d/%d station_status rows (duplicates skipped)",
        inserted,
        len(records),
    )
    return inserted


def upsert_station_information(
    conn: PgConnection,
    records: list[dict[str, Any]],
) -> int:
    """Upsert station information records.

    On conflict the existing row is updated with the latest values.

    Args:
        conn: An open PostgreSQL connection.
        records: Parsed station-information dictionaries.

    Returns:
        Number of rows inserted or updated.
    """
    if not records:
        return 0

    query = """
        INSERT INTO station_information (
            station_id, name, lat, lon, capacity, address, groups
        )
        VALUES (
            %(station_id)s, %(name)s, %(lat)s, %(lon)s,
            %(capacity)s, %(address)s, %(groups)s
        )
        ON CONFLICT (station_id) DO UPDATE SET
            name     = EXCLUDED.name,
            lat      = EXCLUDED.lat,
            lon      = EXCLUDED.lon,
            capacity = EXCLUDED.capacity,
            address  = EXCLUDED.address,
            groups   = EXCLUDED.groups,
            updated_at = NOW()
    """

    affected = 0
    with conn.cursor() as cur:
        for record in records:
            cur.execute(query, record)
            affected += cur.rowcount
    conn.commit()

    logger.info(
        "Upserted %d/%d station_information rows",
        affected,
        len(records),
    )
    return affected
