"""Model evaluation metrics for bike availability forecasting.

Pure functions that compute regression metrics. Independent of any model
implementation so it can be reused across baseline and advanced models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute MAE, RMSE, and R² for a set of predictions.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Dictionary with keys ``mae``, ``rmse``, ``r2``.
    """
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def per_station_metrics(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
) -> pd.DataFrame:
    """Compute metrics grouped by ``station_id``.

    Args:
        df: DataFrame containing predictions, with a ``station_id`` column.
        y_true_col: Column name for ground truth values.
        y_pred_col: Column name for predicted values.

    Returns:
        DataFrame indexed by ``station_id`` with columns ``mae``, ``rmse``,
        ``r2``.
    """
    rows = []
    for station_id, group in df.groupby("station_id"):
        m = compute_metrics(group[y_true_col].values, group[y_pred_col].values)
        rows.append({"station_id": station_id, **m})
    return pd.DataFrame(rows).set_index("station_id")


def per_hour_metrics(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
) -> pd.DataFrame:
    """Compute metrics grouped by ``hour``.

    Args:
        df: DataFrame containing predictions, with an ``hour`` column.
        y_true_col: Column name for ground truth values.
        y_pred_col: Column name for predicted values.

    Returns:
        DataFrame indexed by ``hour`` (0–23) with columns ``mae``, ``rmse``,
        ``r2``.
    """
    rows = []
    for hour, group in df.groupby("hour"):
        m = compute_metrics(group[y_true_col].values, group[y_pred_col].values)
        rows.append({"hour": hour, **m})
    return pd.DataFrame(rows).set_index("hour")
