"""Tests for src.storage.connection — database connection factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.storage.connection import get_connection, get_database_url


class TestGetDatabaseUrl:
    """Tests for get_database_url()."""

    @patch.dict("os.environ", {"DATABASE_URL": "postgresql://user:pass@host/db"})
    def test_returns_url_from_env(self) -> None:
        assert get_database_url() == "postgresql://user:pass@host/db"

    @patch.dict("os.environ", {}, clear=True)
    def test_raises_when_not_set(self) -> None:
        with pytest.raises(RuntimeError, match="DATABASE_URL"):
            get_database_url()


class TestGetConnection:
    """Tests for get_connection() context manager."""

    @patch("src.storage.connection.psycopg2.connect")
    @patch.dict("os.environ", {"DATABASE_URL": "postgresql://test"})
    def test_yields_connection_and_closes(self, mock_connect: MagicMock) -> None:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        with get_connection() as conn:
            assert conn is mock_conn

        mock_conn.close.assert_called_once()

    @patch("src.storage.connection.psycopg2.connect")
    @patch.dict("os.environ", {"DATABASE_URL": "postgresql://test"})
    def test_closes_on_exception(self, mock_connect: MagicMock) -> None:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        with pytest.raises(ValueError, match="boom"), get_connection():
            raise ValueError("boom")

        mock_conn.close.assert_called_once()
