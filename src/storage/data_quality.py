"""Data quality checks for the analytics foundation.

Runs assertions against ``analytics.v_dq_*`` metric views and exits non-zero
when any **hard** check fails. Intended for manual runs, CI with a database,
or post-ingestion validation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

from psycopg2.extensions import connection as PgConnection

logger = logging.getLogger(__name__)

CHECKS: tuple[tuple[str, str], ...] = (
    (
        "orphan_status_rows",
        "SELECT violation_count FROM analytics.v_dq_orphan_status_count",
    ),
    (
        "negative_availability",
        "SELECT violation_count FROM analytics.v_dq_negative_availability_count",
    ),
    (
        "ingestion_before_last_reported",
        "SELECT violation_count FROM analytics.v_dq_ingestion_before_report_count",
    ),
    (
        "duplicate_grain_keys",
        "SELECT violation_count FROM analytics.v_dq_duplicate_grain_count",
    ),
)


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single data-quality check."""

    name: str
    violation_count: int
    passed: bool


def _fetch_violation_count(conn: PgConnection, query: str) -> int:
    with conn.cursor() as cur:
        cur.execute(query)
        row = cur.fetchone()
        if row is None or row[0] is None:
            return 0
        return int(row[0])


def run_checks(conn: PgConnection) -> list[CheckResult]:
    """Execute all configured checks and return per-check results.

    Args:
        conn: Open PostgreSQL connection.

    Returns:
        List of :class:`CheckResult` in stable order.
    """
    results: list[CheckResult] = []
    for name, sql in CHECKS:
        count = _fetch_violation_count(conn, sql)
        passed = count == 0
        results.append(CheckResult(name=name, violation_count=count, passed=passed))
        if passed:
            logger.info("DQ check %s passed (violations=0)", name)
        else:
            logger.warning("DQ check %s failed (violations=%d)", name, count)
    return results


def all_passed(results: list[CheckResult]) -> bool:
    """Return True if every check passed."""
    return all(r.passed for r in results)


def results_to_json(results: list[CheckResult]) -> str:
    """Serialize results as a single JSON object (one line)."""
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": [
            {
                "name": r.name,
                "violation_count": r.violation_count,
                "passed": r.passed,
            }
            for r in results
        ],
        "overall_passed": all_passed(results),
    }
    return json.dumps(payload)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: load .env, connect, run checks, print JSON summary."""
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(
        description="Run data-quality checks against analytics DQ views.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print one JSON line with all check results (stderr still has logs).",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from src.storage.connection import get_connection

    with get_connection() as conn:
        results = run_checks(conn)

    if args.json:
        print(results_to_json(results))

    return 0 if all_passed(results) else 1


if __name__ == "__main__":
    sys.exit(main())
