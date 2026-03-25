"""Database connection factory for PostgreSQL (Neon).

Provides a context-managed connection using DATABASE_URL from the environment.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager

import psycopg2
from psycopg2.extensions import connection as PgConnection


def get_database_url() -> str:
    """Read DATABASE_URL from environment variables.

    Returns:
        The PostgreSQL connection string.

    Raises:
        RuntimeError: If DATABASE_URL is not set.
    """
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL environment variable is not set. "
            "See .env.example for the expected format."
        )
    return url


@contextmanager
def get_connection() -> Generator[PgConnection, None, None]:
    """Yield a PostgreSQL connection and ensure it is closed afterward.

    The connection has ``autocommit=False`` by default; callers are
    responsible for committing or rolling back as needed.

    Yields:
        A live ``psycopg2`` connection.
    """
    conn = psycopg2.connect(get_database_url())
    try:
        yield conn
    finally:
        conn.close()
