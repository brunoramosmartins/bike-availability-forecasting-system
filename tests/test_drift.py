"""Tests for src.monitoring.drift — pure drift detection functions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.monitoring.drift import (
    DriftReport,
    FeatureDriftResult,
    analyze_drift,
    check_mae_alert,
    compute_drift_score,
    compute_feature_drift,
    compute_ks_test,
    compute_psi,
    rolling_mae,
)

# ---------------------------------------------------------------------------
# rolling_mae
# ---------------------------------------------------------------------------


class TestRollingMAE:
    """Tests for rolling_mae()."""

    def test_returns_series(self) -> None:
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rolling_mae(y_true, y_pred, window=2)
        assert isinstance(result, pd.Series)

    def test_perfect_predictions_zero_mae(self) -> None:
        y = pd.Series(np.arange(100, dtype=float))
        result = rolling_mae(y, y, window=10)
        # After the first window, all values should be 0
        assert result.dropna().sum() == 0.0

    def test_window_size_respected(self) -> None:
        y_true = pd.Series(np.arange(20, dtype=float))
        y_pred = pd.Series(np.arange(20, dtype=float) + 1)
        result = rolling_mae(y_true, y_pred, window=5)
        # First 4 values should be NaN (window-1)
        assert result.isna().sum() == 4

    def test_constant_error(self) -> None:
        y_true = pd.Series(np.zeros(50))
        y_pred = pd.Series(np.ones(50))  # constant error of 1.0
        result = rolling_mae(y_true, y_pred, window=10)
        non_nan = result.dropna()
        np.testing.assert_allclose(non_nan.values, 1.0)


# ---------------------------------------------------------------------------
# compute_psi
# ---------------------------------------------------------------------------


class TestComputePSI:
    """Tests for compute_psi()."""

    def test_identical_distributions_near_zero(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)
        psi = compute_psi(data, data)
        assert psi < 0.05

    def test_shifted_distribution_high_psi(self) -> None:
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(5, 1, 1000)  # big shift
        psi = compute_psi(ref, cur)
        assert psi > 0.2

    def test_returns_nonnegative(self) -> None:
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(0.5, 1.5, 500)
        psi = compute_psi(ref, cur)
        assert psi >= 0

    def test_returns_float(self) -> None:
        ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cur = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        psi = compute_psi(ref, cur, n_bins=3)
        assert isinstance(psi, float)


# ---------------------------------------------------------------------------
# compute_ks_test
# ---------------------------------------------------------------------------


class TestComputeKS:
    """Tests for compute_ks_test()."""

    def test_identical_arrays_no_drift(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 500)
        result = compute_ks_test(data, data)
        assert result.drifted is False
        assert result.p_value > 0.05

    def test_shifted_arrays_drift_detected(self) -> None:
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(3, 1, 500)
        result = compute_ks_test(ref, cur)
        assert result.drifted is True
        assert result.p_value < 0.05

    def test_result_fields(self) -> None:
        ref = np.array([1.0, 2.0, 3.0])
        cur = np.array([1.5, 2.5, 3.5])
        result = compute_ks_test(ref, cur)
        assert hasattr(result, "statistic")
        assert hasattr(result, "p_value")
        assert hasattr(result, "drifted")
        assert isinstance(result.statistic, float)
        assert isinstance(result.p_value, float)


# ---------------------------------------------------------------------------
# check_mae_alert
# ---------------------------------------------------------------------------


class TestCheckMAEAlert:
    """Tests for check_mae_alert()."""

    def test_no_alert_below_threshold(self) -> None:
        assert check_mae_alert(0.25, 0.30, threshold=0.20) is False

    def test_alert_above_threshold(self) -> None:
        # 0.40 > 0.30 * 1.20 = 0.36
        assert check_mae_alert(0.40, 0.30, threshold=0.20) is True

    def test_exact_threshold_no_alert(self) -> None:
        # 0.36 is exactly 0.30 * 1.20 — not strictly greater
        assert check_mae_alert(0.36, 0.30, threshold=0.20) is False

    def test_zero_baseline(self) -> None:
        assert check_mae_alert(0.01, 0.0) is True

    def test_zero_both(self) -> None:
        assert check_mae_alert(0.0, 0.0) is False


# ---------------------------------------------------------------------------
# analyze_drift
# ---------------------------------------------------------------------------


class TestAnalyzeDrift:
    """Tests for analyze_drift()."""

    def _make_predictions_df(self, n: int = 200) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        actual = rng.normal(10, 3, n)
        predicted = actual + rng.normal(0, 0.5, n)
        return pd.DataFrame(
            {
                "predicted_value": predicted,
                "actual_value": actual,
            }
        )

    def test_returns_drift_report(self) -> None:
        df = self._make_predictions_df()
        report = analyze_drift(df, baseline_mae=0.5, model_name="lgbm")
        assert isinstance(report, DriftReport)

    def test_returns_none_insufficient_data(self) -> None:
        df = self._make_predictions_df(n=10)
        report = analyze_drift(df, baseline_mae=0.5, model_name="lgbm")
        assert report is None

    def test_report_fields(self) -> None:
        df = self._make_predictions_df()
        report = analyze_drift(df, baseline_mae=0.5, model_name="lgbm")
        assert report is not None
        assert report.model_name == "lgbm"
        assert report.baseline_mae == 0.5
        assert report.n_predictions == 200
        assert isinstance(report.current_mae, float)
        assert isinstance(report.psi_score, float)

    def test_to_dict(self) -> None:
        df = self._make_predictions_df()
        report = analyze_drift(df, baseline_mae=0.5, model_name="lgbm")
        assert report is not None
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "model_name" in d
        assert "mae_alert" in d
        assert "psi_score" in d

    def test_no_alert_when_mae_is_good(self) -> None:
        df = self._make_predictions_df()
        # Set a high baseline so current MAE should be lower
        report = analyze_drift(df, baseline_mae=10.0, model_name="lgbm")
        assert report is not None
        assert report.mae_alert is False


# ---------------------------------------------------------------------------
# compute_feature_drift
# ---------------------------------------------------------------------------


class TestComputeFeatureDrift:
    """Tests for compute_feature_drift()."""

    def test_returns_list_of_results(self) -> None:
        rng = np.random.default_rng(42)
        ref = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(5, 1, 200)})
        cur = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(5, 1, 200)})
        results = compute_feature_drift(ref, cur, ["a", "b"])
        assert len(results) == 2
        assert all(isinstance(r, FeatureDriftResult) for r in results)

    def test_sorted_by_psi_descending(self) -> None:
        rng = np.random.default_rng(42)
        ref = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(5, 1, 200)})
        # Shift feature "b" significantly
        cur = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(50, 1, 200)})
        results = compute_feature_drift(ref, cur, ["a", "b"])
        assert results[0].psi >= results[1].psi

    def test_detects_drift_on_shifted_feature(self) -> None:
        rng = np.random.default_rng(42)
        ref = pd.DataFrame({"x": rng.normal(0, 1, 500)})
        cur = pd.DataFrame({"x": rng.normal(10, 1, 500)})
        results = compute_feature_drift(ref, cur, ["x"])
        assert results[0].drifted is True

    def test_no_drift_on_same_distribution(self) -> None:
        rng = np.random.default_rng(42)
        ref = pd.DataFrame({"x": rng.normal(0, 1, 500)})
        cur = pd.DataFrame({"x": rng.normal(0, 1, 500)})
        results = compute_feature_drift(ref, cur, ["x"])
        assert results[0].drifted is False

    def test_empty_when_insufficient_data(self) -> None:
        ref = pd.DataFrame({"x": [1.0]})
        cur = pd.DataFrame({"x": [2.0]})
        results = compute_feature_drift(ref, cur, ["x"])
        assert results == []


# ---------------------------------------------------------------------------
# compute_drift_score
# ---------------------------------------------------------------------------


class TestComputeDriftScore:
    """Tests for compute_drift_score()."""

    def test_all_drifted(self) -> None:
        results = [
            FeatureDriftResult("a", 0.3, 0.5, 0.001, True),
            FeatureDriftResult("b", 0.4, 0.6, 0.001, True),
        ]
        assert compute_drift_score(results) == 1.0

    def test_none_drifted(self) -> None:
        results = [
            FeatureDriftResult("a", 0.01, 0.1, 0.5, False),
            FeatureDriftResult("b", 0.02, 0.1, 0.6, False),
        ]
        assert compute_drift_score(results) == 0.0

    def test_partial_drift(self) -> None:
        results = [
            FeatureDriftResult("a", 0.3, 0.5, 0.001, True),
            FeatureDriftResult("b", 0.01, 0.1, 0.5, False),
        ]
        assert compute_drift_score(results) == 0.5

    def test_empty_returns_zero(self) -> None:
        assert compute_drift_score([]) == 0.0
