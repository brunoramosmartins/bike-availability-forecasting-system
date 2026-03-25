"""GBFS data fetcher with retry logic.

Fetches ``station_status`` and ``station_information`` endpoints from the
Bike Itaú GBFS feed using *httpx* with configurable timeout and exponential
backoff.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://saopaulo.publicbikesystem.net/customer/gbfs/v2/en"
REQUEST_TIMEOUT = 10.0
MAX_RETRIES = 3
BACKOFF_BASE = 2.0


class FetchError(Exception):
    """Raised when a GBFS endpoint cannot be fetched after all retries."""


def _get_base_url() -> str:
    return os.environ.get("GBFS_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def _fetch_with_retry(url: str) -> dict[str, Any]:
    """Fetch a URL with exponential-backoff retries.

    Args:
        url: The absolute URL to fetch.

    Returns:
        Parsed JSON response as a dictionary.

    Raises:
        FetchError: If all retry attempts fail.
    """
    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("Fetching %s (attempt %d/%d)", url, attempt, MAX_RETRIES)
            with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                response = client.get(url)
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                wait = BACKOFF_BASE**attempt
                logger.warning(
                    "Attempt %d failed (%s). Retrying in %.1fs…",
                    attempt,
                    exc,
                    wait,
                )
                time.sleep(wait)

    raise FetchError(
        f"Failed to fetch {url} after {MAX_RETRIES} attempts"
    ) from last_exc


def fetch_station_status() -> dict[str, Any]:
    """Fetch the ``station_status.json`` endpoint.

    Returns:
        Raw JSON payload from the GBFS feed.

    Raises:
        FetchError: If the request fails after retries.
    """
    url = f"{_get_base_url()}/station_status.json"
    return _fetch_with_retry(url)


def fetch_station_information() -> dict[str, Any]:
    """Fetch the ``station_information.json`` endpoint.

    Returns:
        Raw JSON payload from the GBFS feed.

    Raises:
        FetchError: If the request fails after retries.
    """
    url = f"{_get_base_url()}/station_information.json"
    return _fetch_with_retry(url)
