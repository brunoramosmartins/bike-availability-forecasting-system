"""Database schema management.

Reads DDL files from the ``sql/`` directory and applies them to the database.
"""

from __future__ import annotations

import logging
from pathlib import Path

from psycopg2.extensions import connection as PgConnection

logger = logging.getLogger(__name__)

SQL_DIR = Path(__file__).resolve().parent.parent.parent / "sql"


def apply_schema(conn: PgConnection) -> None:
    """Execute all SQL migration files in order.

    Files are sorted lexicographically so that numbered prefixes
    (e.g. ``001_``, ``002_``) determine execution order.

    Args:
        conn: An open PostgreSQL connection.
    """
    sql_files = sorted(SQL_DIR.glob("*.sql"))
    if not sql_files:
        logger.warning("No SQL files found in %s", SQL_DIR)
        return

    with conn.cursor() as cur:
        for sql_file in sql_files:
            logger.info("Applying %s", sql_file.name)
            cur.execute(sql_file.read_text(encoding="utf-8"))
    conn.commit()
    logger.info("Schema applied successfully")
