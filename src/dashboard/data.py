"""Pure data loading and transformation helpers for the dashboard.

All functions are free of Streamlit dependencies so they can be
unit-tested. Caching is applied at the app layer using
``@st.cache_data``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.dataset.features import FEATURE_COLS
from src.monitoring.drift import (
    compute_drift_score,
    compute_feature_drift,
    rolling_mae,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"


def load_parquet_data(data_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Load and concatenate train/val/test Parquet files.

    Adds a ``split`` column (``"train"`` / ``"val"`` / ``"test"``).

    Raises
    ------
    FileNotFoundError
        If any Parquet file is missing.
    """
    splits: list[pd.DataFrame] = []
    for name in ("train", "val", "test"):
        path = data_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m src.dataset` first."
            )
        df = pd.read_parquet(path)
        df["split"] = name
        splits.append(df)

    combined = pd.concat(splits, ignore_index=True)
    logger.info("Loaded %d rows from Parquet files", len(combined))
    return combined


def load_station_names(
    samples_dir: Path = SAMPLES_DIR,
) -> dict[str, str]:
    """Load station_id -> name mapping from sample JSON.

    Returns an empty dict if the file does not exist.
    """
    path = samples_dir / "station_information.json"
    if not path.exists():
        return {}

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    stations = data.get("data", {}).get("stations", [])
    return {str(s["station_id"]): s.get("name", str(s["station_id"])) for s in stations}


def load_metrics(data_dir: Path = PROCESSED_DIR) -> dict[str, dict[str, float]]:
    """Read metrics.json and return the model metrics dict."""
    path = data_dir / "metrics.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_feature_importance(
    data_dir: Path = PROCESSED_DIR,
) -> pd.DataFrame:
    """Read lgbm_feature_importance.json into a DataFrame."""
    path = data_dir / "lgbm_feature_importance.json"
    if not path.exists():
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.read_json(path)


def filter_by_stations(
    df: pd.DataFrame,
    station_ids: list[str],
) -> pd.DataFrame:
    """Filter DataFrame to selected station IDs.

    Returns all rows if *station_ids* is empty.
    """
    if not station_ids:
        return df
    return df[df["station_id"].isin(station_ids)].copy()


def compute_hourly_availability(df: pd.DataFrame) -> pd.DataFrame:
    """Compute average bike availability grouped by hour and day type.

    Returns a DataFrame with columns: ``hour``, ``day_type``, ``avg_bikes``.
    """
    result = (
        df.groupby(["hour", "is_weekend"])["num_bikes_available"].mean().reset_index()
    )
    result.rename(columns={"num_bikes_available": "avg_bikes"}, inplace=True)
    result["day_type"] = result["is_weekend"].map({0: "Weekday", 1: "Weekend"})
    return result


def compute_station_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-station summary statistics.

    Returns one row per station with: ``station_id``, ``avg_bikes``,
    ``min_bikes``, ``max_bikes``, ``lat``, ``lon``, ``capacity``.
    """
    agg = df.groupby("station_id").agg(
        avg_bikes=("num_bikes_available", "mean"),
        min_bikes=("num_bikes_available", "min"),
        max_bikes=("num_bikes_available", "max"),
        lat=("lat", "first"),
        lon=("lon", "first"),
        capacity=("capacity", "first"),
    )
    agg["fill_pct"] = (agg["avg_bikes"] / agg["capacity"] * 100).round(1)
    return agg.reset_index()


def compute_weekday_hour_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Compute average availability by weekday x hour.

    Returns a pivot table with weekdays as rows (0=Mon..6=Sun)
    and hours (0-23) as columns.
    """
    agg = df.groupby(["weekday", "hour"])["num_bikes_available"].mean().reset_index()
    pivot = agg.pivot(index="weekday", columns="hour", values="num_bikes_available")
    return pivot.fillna(0)


def compute_feature_drift_df(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Compute per-feature drift and return a sorted DataFrame.

    Columns: ``feature``, ``psi``, ``ks_statistic``, ``ks_p_value``, ``drifted``.
    """
    if features is None:
        features = FEATURE_COLS

    results = compute_feature_drift(reference, current, features)
    if not results:
        return pd.DataFrame(
            columns=["feature", "psi", "ks_statistic", "ks_p_value", "drifted"]
        )

    return pd.DataFrame(
        [
            {
                "feature": r.feature,
                "psi": r.psi,
                "ks_statistic": r.ks_statistic,
                "ks_p_value": r.ks_p_value,
                "drifted": r.drifted,
            }
            for r in results
        ]
    )


def compute_aggregate_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str] | None = None,
) -> dict:
    """Compute aggregated drift summary.

    Returns
    -------
    dict
        Keys: ``drift_score``, ``n_features``, ``n_drifted``,
        ``avg_psi``, ``max_psi``.
    """
    if features is None:
        features = FEATURE_COLS

    results = compute_feature_drift(reference, current, features)
    if not results:
        return {
            "drift_score": 0.0,
            "n_features": 0,
            "n_drifted": 0,
            "avg_psi": 0.0,
            "max_psi": 0.0,
        }

    score = compute_drift_score(results)
    psi_values = [r.psi for r in results]
    n_drifted = sum(1 for r in results if r.drifted)

    return {
        "drift_score": score,
        "n_features": len(results),
        "n_drifted": n_drifted,
        "avg_psi": float(np.mean(psi_values)),
        "max_psi": float(np.max(psi_values)),
    }


def compute_rolling_mae_series(
    y_true: pd.Series,
    y_pred: pd.Series,
    window: int = 96,
) -> pd.Series:
    """Thin wrapper around :func:`rolling_mae` for dashboard use."""
    return rolling_mae(y_true, y_pred, window=window)


def generate_predictions(
    test_df: pd.DataFrame,
    model_path: Path,
) -> np.ndarray:
    """Load a serialized model and generate predictions.

    Returns
    -------
    np.ndarray
        Predicted values for the test set.
    """
    model = joblib.load(model_path)
    return model.predict(test_df[FEATURE_COLS])
