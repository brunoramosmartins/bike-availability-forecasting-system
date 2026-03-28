"""CLI entry point for the dataset construction pipeline.

Usage::

    python -m src.dataset

Connects to the database, resamples raw data, engineers features, and
exports train/val/test splits as Parquet files to ``data/processed/``.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from src.dataset.features import FEATURE_COLS, TARGET_COL, build_features
from src.dataset.resampler import load_raw_status, resample_all
from src.dataset.splitter import time_based_split
from src.storage.connection import get_connection


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
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter())
    logging.root.handlers.clear()
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)


OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


def run() -> None:
    """Execute the full dataset construction pipeline.

    Steps:
        1. Load enriched station status from the database.
        2. Resample to 15-minute intervals.
        3. Engineer features (lag, rolling, temporal, target).
        4. Split into train / validation / test sets.
        5. Export as Parquet files.
    """
    logger = logging.getLogger("src.dataset")
    start = time.monotonic()

    logger.info("Starting dataset construction pipeline")

    with get_connection() as conn:
        raw_df = load_raw_status(conn)

    if raw_df.empty:
        logger.error("No data available — cannot build dataset")
        sys.exit(1)

    resampled = resample_all(raw_df)
    features_df = build_features(resampled)

    if features_df.empty:
        logger.error("Feature engineering produced an empty dataset")
        sys.exit(1)

    split = time_based_split(features_df)

    # Export
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_cols = ["station_id", "timestamp"] + FEATURE_COLS + [TARGET_COL]
    for name, subset in [
        ("train", split.train),
        ("val", split.val),
        ("test", split.test),
    ]:
        path = OUTPUT_DIR / f"{name}.parquet"
        subset[all_cols].to_parquet(path, index=False)
        logger.info("Saved %s → %s (%d rows)", name, path, len(subset))

    elapsed = time.monotonic() - start
    logger.info(
        "Dataset pipeline complete in %.2fs — "
        "train=%d, val=%d, test=%d, features=%d",
        elapsed,
        len(split.train),
        len(split.val),
        len(split.test),
        len(FEATURE_COLS),
    )


def main() -> None:
    """Entry point with error handling and exit-code management."""
    load_dotenv()
    _configure_logging()
    logger = logging.getLogger("src.dataset")

    try:
        run()
    except Exception:
        logger.exception("Unexpected error in dataset pipeline")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
