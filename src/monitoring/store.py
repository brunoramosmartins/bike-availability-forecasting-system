"""Prediction storage — save, backfill, and load from PostgreSQL.

Follows the same pattern as :mod:`src.ingestion.loader`: functions
receive an open ``PgConnection`` and execute parameterized queries.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from psycopg2.extensions import connection as PgConnection

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


# -------------------------------------------------------------------------
# Save predictions
# -------------------------------------------------------------------------


def save_predictions(
    conn: PgConnection,
    predictions: list[dict[str, Any]],
) -> int:
    """Bulk insert predictions into ``analytics.predictions``.

    Each dict must have keys: ``station_id``, ``prediction_time``,
    ``target_time``, ``model_name``, ``predicted_value``.

    Duplicates (same station + target_time + model) are skipped.

    Returns
    -------
    int
        Number of rows actually inserted.
    """
    if not predictions:
        return 0

    query = """
        INSERT INTO analytics.predictions (
            station_id, prediction_time, target_time,
            model_name, predicted_value
        )
        VALUES (
            %(station_id)s, %(prediction_time)s, %(target_time)s,
            %(model_name)s, %(predicted_value)s
        )
        ON CONFLICT (station_id, target_time, model_name) DO NOTHING
    """

    inserted = 0
    with conn.cursor() as cur:
        for record in predictions:
            cur.execute(query, record)
            inserted += cur.rowcount
    conn.commit()

    logger.info(
        "Inserted %d/%d prediction rows (duplicates skipped)",
        inserted,
        len(predictions),
    )
    return inserted


# -------------------------------------------------------------------------
# Backfill actuals
# -------------------------------------------------------------------------


def backfill_actuals(
    conn: PgConnection,
    actuals_df: pd.DataFrame,
) -> int:
    """Update ``actual_value`` in predictions from observed actuals.

    Parameters
    ----------
    actuals_df
        Must contain columns: ``station_id``, ``target_time``,
        ``actual_value``.

    Returns
    -------
    int
        Number of rows updated.
    """
    if actuals_df.empty:
        return 0

    query = """
        UPDATE analytics.predictions
        SET actual_value = %(actual_value)s
        WHERE station_id = %(station_id)s
          AND target_time = %(target_time)s
          AND actual_value IS NULL
    """

    updated = 0
    with conn.cursor() as cur:
        for _, row in actuals_df.iterrows():
            cur.execute(
                query,
                {
                    "station_id": str(row["station_id"]),
                    "target_time": row["target_time"],
                    "actual_value": float(row["actual_value"]),
                },
            )
            updated += cur.rowcount
    conn.commit()

    logger.info("Backfilled %d actual values", updated)
    return updated


# -------------------------------------------------------------------------
# Load predictions
# -------------------------------------------------------------------------


def load_predictions(
    conn: PgConnection,
    model_name: str,
    since: datetime | None = None,
) -> pd.DataFrame:
    """Load prediction/actual pairs for drift analysis.

    Only returns rows where ``actual_value IS NOT NULL`` so drift
    analysis operates on complete pairs.

    Parameters
    ----------
    conn
        Open PostgreSQL connection.
    model_name
        Model to filter by (e.g. ``"lgbm"``).
    since
        Optional lower bound on ``target_time``.

    Returns
    -------
    pd.DataFrame
        Columns: ``station_id``, ``target_time``, ``predicted_value``,
        ``actual_value``.
    """
    query = """
        SELECT station_id, target_time, predicted_value, actual_value
        FROM analytics.predictions
        WHERE model_name = %s
          AND actual_value IS NOT NULL
    """
    params: list[Any] = [model_name]

    if since is not None:
        query += " AND target_time >= %s"
        params.append(since)

    query += " ORDER BY target_time"

    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    if not rows:
        logger.info("No predictions found for model=%s", model_name)
        return pd.DataFrame(
            columns=["station_id", "target_time", "predicted_value", "actual_value"]
        )

    df = pd.DataFrame(
        rows,
        columns=["station_id", "target_time", "predicted_value", "actual_value"],
    )
    logger.info("Loaded %d prediction rows for model=%s", len(df), model_name)
    return df


# -------------------------------------------------------------------------
# Load baseline metrics from disk
# -------------------------------------------------------------------------


def load_baseline_metrics() -> dict[str, dict[str, float]]:
    """Read ``data/processed/metrics.json`` and return the model metrics dict.

    Raises
    ------
    FileNotFoundError
        If the metrics file does not exist.
    """
    metrics_path = DATA_DIR / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Baseline metrics not found: {metrics_path}. "
            "Run `python -m src.model` first."
        )

    with open(metrics_path, encoding="utf-8") as f:
        metrics: dict[str, dict[str, float]] = json.load(f)

    logger.info(
        "Loaded baseline metrics for models: %s",
        ", ".join(metrics.keys()),
    )
    return metrics


# -------------------------------------------------------------------------
# Build predictions from a model + dataset
# -------------------------------------------------------------------------


def build_prediction_records(
    model: Any,
    df: pd.DataFrame,
    model_name: str,
    feature_cols: list[str],
) -> list[dict[str, Any]]:
    """Generate prediction records ready for :func:`save_predictions`.

    Parameters
    ----------
    model
        A fitted model with a ``predict(X)`` method.
    df
        DataFrame with ``station_id``, ``timestamp``, features, and
        optionally ``y`` (actual target).
    model_name
        Identifier for the model (e.g. ``"lgbm"``).
    feature_cols
        List of feature column names.

    Returns
    -------
    list[dict]
        Each dict has: ``station_id``, ``prediction_time``,
        ``target_time``, ``model_name``, ``predicted_value``.
    """
    predictions = model.predict(df[feature_cols])
    now = datetime.now(timezone.utc)

    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        records.append(
            {
                "station_id": str(row["station_id"]),
                "prediction_time": now,
                "target_time": row["timestamp"] + pd.Timedelta(minutes=15),
                "model_name": model_name,
                "predicted_value": float(predictions[i]),
            }
        )

    return records
