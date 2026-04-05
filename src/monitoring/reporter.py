"""Evidently AI report generation for data drift and model performance.

Generates HTML reports comparing a reference (training) distribution
against a current (production) distribution.

Uses the Evidently v2 API (``evidently >= 0.5``).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from evidently import Dataset, Report
from evidently.core.datasets import DataDefinition, Regression
from evidently.presets import DataDriftPreset, RegressionPreset

logger = logging.getLogger(__name__)


def generate_data_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Create an Evidently data drift report.

    Compares feature distributions between reference (training) and
    current (production) data.

    Parameters
    ----------
    reference_df
        Training/reference data with feature columns.
    current_df
        Current/production data with the same columns.
    output_path
        Path to save the HTML report.

    Returns
    -------
    Path
        The path where the report was saved.
    """
    report = Report([DataDriftPreset()])
    snapshot = report.run(
        reference_data=reference_df,
        current_data=current_df,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output_path))

    logger.info("Data drift report saved -> %s", output_path)
    return output_path


def generate_model_performance_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: Path,
    target_col: str = "target",
    prediction_col: str = "prediction",
) -> Path:
    """Create an Evidently regression performance report.

    Both DataFrames must contain columns named by *target_col* and
    *prediction_col*.

    Parameters
    ----------
    reference_df
        Reference data with target and prediction columns.
    current_df
        Current data with the same columns.
    output_path
        Path to save the HTML report.
    target_col
        Name of the target column (default: ``"target"``).
    prediction_col
        Name of the prediction column (default: ``"prediction"``).

    Returns
    -------
    Path
        The path where the report was saved.
    """
    data_def = DataDefinition(
        regression=[Regression(target=target_col, prediction=prediction_col)]
    )

    ref_ds = Dataset.from_pandas(reference_df, data_definition=data_def)
    cur_ds = Dataset.from_pandas(current_df, data_definition=data_def)

    report = Report([RegressionPreset()])
    snapshot = report.run(reference_data=ref_ds, current_data=cur_ds)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output_path))

    logger.info("Model performance report saved -> %s", output_path)
    return output_path


def generate_all_reports(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    reports_dir: Path,
    feature_cols: list[str] | None = None,
    target_col: str = "target",
    prediction_col: str = "prediction",
) -> list[Path]:
    """Generate all monitoring reports (data drift + model performance).

    Parameters
    ----------
    reference_df
        Reference data with features, *target_col*, and *prediction_col*.
    current_df
        Current data with the same schema.
    reports_dir
        Directory to save reports.
    feature_cols
        If provided, only compare these features for drift.
    target_col
        Name of the target column.
    prediction_col
        Name of the prediction column.

    Returns
    -------
    list[Path]
        Paths of all generated reports.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M")
    paths: list[Path] = []

    # Data drift on features
    if feature_cols is not None:
        ref_features = reference_df[feature_cols]
        cur_features = current_df[feature_cols]
    else:
        ref_features = reference_df
        cur_features = current_df

    drift_path = reports_dir / f"data_drift_{timestamp}.html"
    paths.append(generate_data_drift_report(ref_features, cur_features, drift_path))

    # Model performance
    perf_path = reports_dir / f"model_performance_{timestamp}.html"
    paths.append(
        generate_model_performance_report(
            reference_df,
            current_df,
            perf_path,
            target_col=target_col,
            prediction_col=prediction_col,
        )
    )

    logger.info("Generated %d reports in %s", len(paths), reports_dir)
    return paths
