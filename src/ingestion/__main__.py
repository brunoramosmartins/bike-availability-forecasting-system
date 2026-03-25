"""CLI entry point for the GBFS data ingestion pipeline.

Usage::

    python -m src.ingestion

Orchestrates the full fetch → parse → load cycle with structured JSON
logging and proper exit codes.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

from src.ingestion.fetcher import (
    FetchError,
    fetch_station_information,
    fetch_station_status,
)
from src.ingestion.loader import (
    upsert_station_information,
    upsert_station_status,
)
from src.ingestion.parser import (
    ParseError,
    parse_station_information,
    parse_station_status,
)
from src.storage.connection import get_connection
from src.storage.schema import apply_schema


class _JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def _configure_logging() -> None:
    """Set up root logger with structured JSON output to stderr."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter())
    logging.root.handlers.clear()
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)


def run() -> None:
    """Execute the full ingestion pipeline.

    Steps:
        1. Ensure database schema exists.
        2. Fetch station_status and station_information from the GBFS API.
        3. Parse and validate the responses.
        4. Load records into PostgreSQL.

    Raises:
        FetchError: If API requests fail after retries.
        ParseError: If responses have unexpected structure.
    """
    logger = logging.getLogger("src.ingestion")
    start = time.monotonic()

    logger.info("Starting ingestion pipeline")

    with get_connection() as conn:
        apply_schema(conn)

        status_payload = fetch_station_status()
        info_payload = fetch_station_information()

        status_records = parse_station_status(status_payload)
        info_records = parse_station_information(info_payload)

        status_inserted = upsert_station_status(conn, status_records)
        info_upserted = upsert_station_information(conn, info_records)

    elapsed = time.monotonic() - start
    logger.info(
        "Ingestion complete in %.2fs — status_inserted=%d, info_upserted=%d",
        elapsed,
        status_inserted,
        info_upserted,
    )


def main() -> None:
    """Entry point with error handling and exit-code management."""
    load_dotenv()
    _configure_logging()
    logger = logging.getLogger("src.ingestion")

    try:
        run()
    except (FetchError, ParseError) as exc:
        logger.error("Pipeline failed: %s", exc)
        sys.exit(1)
    except Exception:
        logger.exception("Unexpected error in ingestion pipeline")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
