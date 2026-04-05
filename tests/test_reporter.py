"""Tests for src.monitoring.reporter — Evidently AI report generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.monitoring.reporter import (
    generate_all_reports,
    generate_data_drift_report,
    generate_model_performance_report,
)


def _make_feature_df(n: int = 200) -> pd.DataFrame:
    """Create a synthetic feature DataFrame."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "num_bikes_available": rng.integers(0, 30, n).astype(float),
            "num_docks_available": rng.integers(0, 30, n).astype(float),
            "bikes_lag_1": rng.integers(0, 30, n).astype(float),
            "hour": rng.integers(0, 24, n),
            "weekday": rng.integers(0, 7, n),
        }
    )


def _make_performance_df(n: int = 200) -> pd.DataFrame:
    """Create a synthetic DataFrame with target and prediction columns."""
    rng = np.random.default_rng(42)
    target = rng.normal(10, 3, n)
    return pd.DataFrame(
        {
            "target": target,
            "prediction": target + rng.normal(0, 0.5, n),
        }
    )


class TestGenerateDataDriftReport:
    """Tests for generate_data_drift_report()."""

    def test_creates_html_file(self, tmp_path: Path) -> None:
        ref = _make_feature_df()
        cur = _make_feature_df()
        output = tmp_path / "drift.html"

        result = generate_data_drift_report(ref, cur, output)

        assert result == output
        assert output.exists()
        assert output.stat().st_size > 0

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        ref = _make_feature_df()
        cur = _make_feature_df()
        output = tmp_path / "subdir" / "drift.html"

        generate_data_drift_report(ref, cur, output)

        assert output.exists()


class TestGenerateModelPerformanceReport:
    """Tests for generate_model_performance_report()."""

    def test_creates_html_file(self, tmp_path: Path) -> None:
        ref = _make_performance_df()
        cur = _make_performance_df()
        output = tmp_path / "perf.html"

        result = generate_model_performance_report(ref, cur, output)

        assert result == output
        assert output.exists()
        assert output.stat().st_size > 0


class TestGenerateAllReports:
    """Tests for generate_all_reports()."""

    def test_generates_two_reports(self, tmp_path: Path) -> None:
        ref = _make_feature_df()
        ref["target"] = ref["num_bikes_available"]
        ref["prediction"] = ref["num_bikes_available"] + 0.5

        cur = _make_feature_df()
        cur["target"] = cur["num_bikes_available"]
        cur["prediction"] = cur["num_bikes_available"] + 0.3

        feature_cols = ["num_bikes_available", "num_docks_available", "hour"]

        paths = generate_all_reports(ref, cur, tmp_path, feature_cols=feature_cols)

        assert len(paths) == 2
        for p in paths:
            assert p.exists()
            assert p.suffix == ".html"
