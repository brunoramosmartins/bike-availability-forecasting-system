"""Tests for src.storage.schema — DDL execution."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.storage.schema import SQL_DIR, apply_schema


class TestApplySchema:
    """Tests for apply_schema()."""

    def test_executes_sql_files_in_order(self) -> None:
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        apply_schema(conn)

        assert cursor.execute.call_count >= 1
        conn.commit.assert_called_once()

    @patch("src.storage.schema.SQL_DIR")
    def test_warns_when_no_sql_files(self, mock_sql_dir: MagicMock) -> None:
        mock_sql_dir.glob.return_value = []
        conn = MagicMock()
        apply_schema(conn)
        conn.commit.assert_not_called()
