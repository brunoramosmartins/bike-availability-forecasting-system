"""CLI entry point for model training and evaluation.

Usage::

    python -m src.model

Loads Parquet splits from ``data/processed/``, trains baseline and
advanced models (with Optuna hyperparameter tuning for LightGBM),
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
from src.model.advanced import LightGBMModel, XGBoostModel, tune_lightgbm
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
    """Train all models and evaluate on the test set.

    Steps:
        1. Load train, val, and test Parquet splits.
        2. Tune LightGBM with Optuna (val set as objective).
        3. Train all models (Naive, LR, LightGBM, XGBoost).
        4. Evaluate on the test set.
        5. Save metrics, feature importance, and serialised models.
    """
    logger = logging.getLogger("src.model")
    start = time.monotonic()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    train_path = DATA_DIR / "train.parquet"
    val_path = DATA_DIR / "val.parquet"
    test_path = DATA_DIR / "test.parquet"

    for path in (train_path, val_path, test_path):
        if not path.exists():
            logger.error(
                "Parquet file not found: %s. Run `python -m src.dataset` first.",
                path,
            )
            sys.exit(1)

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    logger.info(
        "Loaded train=%d, val=%d, test=%d rows",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_val = val_df[FEATURE_COLS]
    y_val = val_df[TARGET_COL]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    # ------------------------------------------------------------------
    # 2. Tune LightGBM
    # ------------------------------------------------------------------
    logger.info("Starting Optuna hyperparameter search (50 trials)...")
    best_params = tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=50)

    params_path = DATA_DIR / "lgbm_best_params.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    logger.info("Best params saved -> %s", params_path)

    # ------------------------------------------------------------------
    # 3. Train models
    # ------------------------------------------------------------------
    models = {
        "naive": NaiveBaseline(),
        "lr": LinearRegressionModel(),
        "lgbm": LightGBMModel(**best_params),
        "xgb": XGBoostModel(),
    }

    for name, model in models.items():
        if hasattr(model, "fit") and name in ("lgbm", "xgb"):
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        else:
            model.fit(X_train, y_train)
        logger.info("Trained model: %s", name)

    # ------------------------------------------------------------------
    # 4. Evaluate
    # ------------------------------------------------------------------
    all_metrics: dict[str, dict[str, float]] = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test.values, y_pred)
        all_metrics[name] = metrics
        logger.info(
            "Model %s -- MAE=%.4f  RMSE=%.4f  R2=%.4f",
            name,
            metrics["mae"],
            metrics["rmse"],
            metrics["r2"],
        )

    # ------------------------------------------------------------------
    # 5. Save artifacts
    # ------------------------------------------------------------------
    metrics_path = DATA_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Metrics saved -> %s", metrics_path)

    # Feature importance (LightGBM)
    lgbm_model = models["lgbm"]
    importance_df = lgbm_model.feature_importance("gain")
    importance_path = DATA_DIR / "lgbm_feature_importance.json"
    importance_df.to_json(importance_path, orient="records", indent=2)
    logger.info("Feature importance saved -> %s", importance_path)

    for name, model in models.items():
        model_path = DATA_DIR / f"{name}.joblib"
        joblib.dump(model, model_path)
        logger.info("Model saved -> %s", model_path)

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
