"""CLI entry point for baseline model training and evaluation.

Usage::

    python -m src.model

Loads Parquet splits from ``data/processed/``, trains baseline models,
evaluates on the test set, and saves metrics and serialised models.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv

from src.dataset.features import FEATURE_COLS, TARGET_COL
from src.model.baseline import LinearRegressionModel, NaiveBaseline
from src.model.evaluate import compute_metrics


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


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


def run() -> None:
    """Train baseline models and evaluate on the test set.

    Steps:
        1. Load train and test Parquet splits.
        2. Train NaiveBaseline and LinearRegressionModel.
        3. Evaluate both on the test set.
        4. Save metrics JSON and serialised models.
    """
    logger = logging.getLogger("src.model")
    start = time.monotonic()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    train_path = DATA_DIR / "train.parquet"
    test_path = DATA_DIR / "test.parquet"

    if not train_path.exists() or not test_path.exists():
        logger.error(
            "Parquet files not found in %s. "
            "Run `python -m src.dataset` first to generate them.",
            DATA_DIR,
        )
        sys.exit(1)

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    logger.info("Loaded train=%d rows, test=%d rows", len(train_df), len(test_df))

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    # ------------------------------------------------------------------
    # 2. Train models
    # ------------------------------------------------------------------
    models = {
        "naive": NaiveBaseline(),
        "lr": LinearRegressionModel(),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        logger.info("Trained model: %s", name)

    # ------------------------------------------------------------------
    # 3. Evaluate
    # ------------------------------------------------------------------
    all_metrics: dict[str, dict[str, float]] = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test.values, y_pred)
        all_metrics[name] = metrics
        logger.info(
            "Model %s — MAE=%.4f  RMSE=%.4f  R²=%.4f",
            name,
            metrics["mae"],
            metrics["rmse"],
            metrics["r2"],
        )

    # ------------------------------------------------------------------
    # 4. Save artifacts
    # ------------------------------------------------------------------
    metrics_path = DATA_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Metrics saved → %s", metrics_path)

    for name, model in models.items():
        model_path = DATA_DIR / f"{name}.joblib"
        joblib.dump(model, model_path)
        logger.info("Model saved → %s", model_path)

    elapsed = time.monotonic() - start
    logger.info("Model pipeline complete in %.2fs", elapsed)


def main() -> None:
    """Entry point with error handling and exit-code management."""
    load_dotenv()
    _configure_logging()
    logger = logging.getLogger("src.model")

    try:
        run()
    except Exception:
        logger.exception("Unexpected error in model pipeline")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
