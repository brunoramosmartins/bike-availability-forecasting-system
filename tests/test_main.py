"""Tests for src.ingestion.__main__ — CLI entry point and orchestration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.__main__ import main, run
from src.ingestion.fetcher import FetchError


class TestRun:
    """Tests for the run() orchestration function."""

    @patch("src.ingestion.__main__.upsert_station_information", return_value=240)
    @patch("src.ingestion.__main__.upsert_station_status", return_value=240)
    @patch("src.ingestion.__main__.parse_station_information", return_value=[])
    @patch("src.ingestion.__main__.parse_station_status", return_value=[])
    @patch(
        "src.ingestion.__main__.fetch_station_information",
        return_value={"data": {"stations": []}},
    )
    @patch(
        "src.ingestion.__main__.fetch_station_status",
        return_value={"data": {"stations": []}},
    )
    @patch("src.ingestion.__main__.apply_schema")
    @patch("src.ingestion.__main__.get_connection")
    def test_orchestrates_full_pipeline(
        self,
        mock_conn: MagicMock,
        mock_schema: MagicMock,
        mock_fetch_status: MagicMock,
        mock_fetch_info: MagicMock,
        mock_parse_status: MagicMock,
        mock_parse_info: MagicMock,
        mock_upsert_status: MagicMock,
        mock_upsert_info: MagicMock,
    ) -> None:
        mock_conn.return_value.__enter__ = MagicMock()
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)

        run()

        mock_schema.assert_called_once()
        mock_fetch_status.assert_called_once()
        mock_fetch_info.assert_called_once()
        mock_parse_status.assert_called_once()
        mock_parse_info.assert_called_once()
        mock_upsert_status.assert_called_once()
        mock_upsert_info.assert_called_once()


class TestMain:
    """Tests for the main() entry point."""

    @patch("src.ingestion.__main__.run")
    @patch("src.ingestion.__main__.load_dotenv")
    def test_exits_zero_on_success(
        self,
        mock_dotenv: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

    @patch("src.ingestion.__main__.run", side_effect=FetchError("API down"))
    @patch("src.ingestion.__main__.load_dotenv")
    def test_exits_one_on_fetch_error(
        self,
        mock_dotenv: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    @patch("src.ingestion.__main__.run", side_effect=RuntimeError("unexpected"))
    @patch("src.ingestion.__main__.load_dotenv")
    def test_exits_one_on_unexpected_error(
        self,
        mock_dotenv: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
