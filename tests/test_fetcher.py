"""Tests for src.ingestion.fetcher — HTTP fetching with retry logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.ingestion.fetcher import (
    BACKOFF_BASE,
    MAX_RETRIES,
    FetchError,
    fetch_station_information,
    fetch_station_status,
)


def _mock_response(json_data: dict, status_code: int = 200) -> httpx.Response:
    """Build a fake httpx.Response with the given JSON body."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = json_data
    response.raise_for_status.return_value = None
    return response


class TestFetchStationStatus:
    """Tests for fetch_station_status()."""

    @patch("src.ingestion.fetcher.httpx.Client")
    def test_returns_json_on_success(self, mock_client_cls: MagicMock) -> None:
        payload = {"data": {"stations": [{"station_id": "1"}]}}
        mock_client = MagicMock()
        mock_client.get.return_value = _mock_response(payload)
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        result = fetch_station_status()

        assert result == payload
        mock_client.get.assert_called_once()

    @patch("src.ingestion.fetcher.time.sleep")
    @patch("src.ingestion.fetcher.httpx.Client")
    def test_retries_on_http_error(
        self,
        mock_client_cls: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        payload = {"data": {"stations": []}}
        mock_client = MagicMock()
        mock_client.get.side_effect = [
            httpx.RequestError("connection reset"),
            _mock_response(payload),
        ]
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        result = fetch_station_status()

        assert result == payload
        assert mock_client.get.call_count == 2
        mock_sleep.assert_called_once_with(BACKOFF_BASE**1)

    @patch("src.ingestion.fetcher.time.sleep")
    @patch("src.ingestion.fetcher.httpx.Client")
    def test_raises_fetch_error_after_max_retries(
        self,
        mock_client_cls: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.RequestError("timeout")
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        with pytest.raises(FetchError, match=f"after {MAX_RETRIES} attempts"):
            fetch_station_status()

        assert mock_client.get.call_count == MAX_RETRIES


class TestFetchStationInformation:
    """Tests for fetch_station_information()."""

    @patch("src.ingestion.fetcher.httpx.Client")
    def test_returns_json_on_success(self, mock_client_cls: MagicMock) -> None:
        payload = {"data": {"stations": [{"station_id": "10", "name": "Test"}]}}
        mock_client = MagicMock()
        mock_client.get.return_value = _mock_response(payload)
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        result = fetch_station_information()

        assert result == payload
        assert "station_information.json" in mock_client.get.call_args[0][0]
