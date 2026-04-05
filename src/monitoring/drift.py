"""Drift detection and model performance monitoring.

Pure functions — no database access or file I/O. Operates on NumPy
arrays and pandas DataFrames, returning structured results.

Key components:

- :func:`rolling_mae` — sliding-window MAE over time-ordered predictions.
- :func:`compute_psi` — Population Stability Index between distributions.
- :func:`compute_ks_test` — Kolmogorov–Smirnov two-sample test wrapper.
- :func:`check_mae_alert` — threshold-based alerting (>20% degradation).
- :func:`analyze_drift` — orchestrator returning a :class:`DriftReport`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Rolling MAE
# -------------------------------------------------------------------------


def rolling_mae(
    y_true: pd.Series,
    y_pred: pd.Series,
    window: int = 96,
) -> pd.Series:
    """Compute rolling MAE over a sliding window.

    Parameters
    ----------
    y_true, y_pred
        Aligned series of actual and predicted values.
    window
        Number of observations per window (default: 96 = 24h at 15-min).

    Returns
    -------
    pd.Series
        Rolling MAE; first ``window - 1`` values are NaN.
    """
    errors = (y_true - y_pred).abs()
    return errors.rolling(window=window, min_periods=window).mean()


# -------------------------------------------------------------------------
# Population Stability Index (PSI)
# -------------------------------------------------------------------------


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Population Stability Index between two 1-D distributions.

    PSI > 0.1 suggests moderate drift; PSI > 0.2 is significant.

    Parameters
    ----------
    reference
        Baseline (training) distribution.
    current
        Recent (production) distribution.
    n_bins
        Number of equal-width bins.

    Returns
    -------
    float
        PSI value (non-negative).
    """
    eps = 1e-4

    # Shared bin edges from the reference distribution
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())
    bins = np.linspace(min_val, max_val, n_bins + 1)

    ref_counts, _ = np.histogram(reference, bins=bins)
    cur_counts, _ = np.histogram(current, bins=bins)

    ref_pct = ref_counts / ref_counts.sum() + eps
    cur_pct = cur_counts / cur_counts.sum() + eps

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


# -------------------------------------------------------------------------
# Kolmogorov–Smirnov test
# -------------------------------------------------------------------------


@dataclass(frozen=True)
class KSResult:
    """Result of a two-sample KS test."""

    statistic: float
    p_value: float
    drifted: bool


def compute_ks_test(
    reference: np.ndarray,
    current: np.ndarray,
    alpha: float = 0.05,
) -> KSResult:
    """Two-sample Kolmogorov–Smirnov test.

    Parameters
    ----------
    reference, current
        1-D arrays to compare.
    alpha
        Significance level for drift detection.

    Returns
    -------
    KSResult
        Test statistic, p-value, and whether drift was detected.
    """
    stat, p_value = stats.ks_2samp(reference, current)
    return KSResult(
        statistic=float(stat),
        p_value=float(p_value),
        drifted=bool(p_value < alpha),
    )


# -------------------------------------------------------------------------
# MAE alerting
# -------------------------------------------------------------------------


def check_mae_alert(
    current_mae: float,
    baseline_mae: float,
    threshold: float = 0.20,
) -> bool:
    """Return True if current MAE exceeds baseline by more than *threshold*.

    Default threshold is 20% degradation.
    """
    if baseline_mae <= 0:
        return current_mae > 0
    return current_mae > baseline_mae * (1 + threshold)


# -------------------------------------------------------------------------
# Drift report
# -------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftReport:
    """Aggregated drift analysis results."""

    model_name: str
    current_mae: float
    baseline_mae: float
    mae_alert: bool
    psi_score: float
    ks_statistic: float
    ks_p_value: float
    distribution_drift: bool
    n_predictions: int

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        return {
            "model_name": self.model_name,
            "current_mae": round(self.current_mae, 6),
            "baseline_mae": round(self.baseline_mae, 6),
            "mae_alert": self.mae_alert,
            "psi_score": round(self.psi_score, 6),
            "ks_statistic": round(self.ks_statistic, 6),
            "ks_p_value": round(self.ks_p_value, 6),
            "distribution_drift": self.distribution_drift,
            "n_predictions": self.n_predictions,
        }


MIN_PREDICTIONS = 96  # at least 24h of data at 15-min intervals


def analyze_drift(
    predictions_df: pd.DataFrame,
    baseline_mae: float,
    model_name: str,
) -> DriftReport | None:
    """Run full drift analysis on a predictions DataFrame.

    Parameters
    ----------
    predictions_df
        Must contain columns ``predicted_value`` and ``actual_value``.
    baseline_mae
        MAE from the test set evaluation (loaded from metrics.json).
    model_name
        Model identifier (e.g. ``"lgbm"``).

    Returns
    -------
    DriftReport or None
        None if insufficient data (< 96 rows).
    """
    if len(predictions_df) < MIN_PREDICTIONS:
        logger.warning(
            "Insufficient predictions (%d < %d). Skipping drift analysis.",
            len(predictions_df),
            MIN_PREDICTIONS,
        )
        return None

    y_true = predictions_df["actual_value"].values
    y_pred = predictions_df["predicted_value"].values

    # Current MAE
    current_mae = float(np.mean(np.abs(y_true - y_pred)))

    # MAE alert
    mae_alert = check_mae_alert(current_mae, baseline_mae)

    # Distribution drift on predicted vs actual
    psi = compute_psi(y_true, y_pred)
    ks = compute_ks_test(y_true, y_pred)

    report = DriftReport(
        model_name=model_name,
        current_mae=current_mae,
        baseline_mae=baseline_mae,
        mae_alert=mae_alert,
        psi_score=psi,
        ks_statistic=ks.statistic,
        ks_p_value=ks.p_value,
        distribution_drift=ks.drifted,
        n_predictions=len(predictions_df),
    )

    logger.info(
        "Drift report: MAE=%.4f (baseline=%.4f, alert=%s), PSI=%.4f, KS=%.4f (p=%.4f)",
        report.current_mae,
        report.baseline_mae,
        report.mae_alert,
        report.psi_score,
        report.ks_statistic,
        report.ks_p_value,
    )

    return report
