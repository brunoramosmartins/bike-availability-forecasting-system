"""Tests for src.monitoring.__main__ — CLI entry point."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from src.monitoring.__main__ import run
from src.monitoring.drift import DriftReport


class TestRun:
    """Tests for run()."""

    @patch("src.monitoring.__main__.get_connection")
    @patch("src.monitoring.__main__.load_predictions")
    @patch("src.monitoring.__main__.load_baseline_metrics")
    def test_exits_zero_no_predictions(
        self,
        mock_metrics: MagicMock,
        mock_load: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        mock_metrics.return_value = {"lgbm": {"mae": 0.15, "rmse": 0.5, "r2": 0.98}}
        mock_load.return_value = pd.DataFrame(
            columns=["station_id", "target_time", "predicted_value", "actual_value"]
        )
        mock_conn.return_value.__enter__ = MagicMock()
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)

        code = run(["--model", "lgbm"])
        assert code == 0

    @patch("src.monitoring.__main__.analyze_drift")
    @patch("src.monitoring.__main__.get_connection")
    @patch("src.monitoring.__main__.load_predictions")
    @patch("src.monitoring.__main__.load_baseline_metrics")
    def test_exits_zero_no_alert(
        self,
        mock_metrics: MagicMock,
        mock_load: MagicMock,
        mock_conn: MagicMock,
        mock_analyze: MagicMock,
    ) -> None:
        mock_metrics.return_value = {"lgbm": {"mae": 0.15, "rmse": 0.5, "r2": 0.98}}
        mock_load.return_value = pd.DataFrame(
            {
                "station_id": ["1"] * 100,
                "target_time": pd.date_range("2026-04-01", periods=100, freq="15min"),
                "predicted_value": [5.0] * 100,
                "actual_value": [5.1] * 100,
            }
        )
        mock_conn.return_value.__enter__ = MagicMock()
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)
        mock_analyze.return_value = DriftReport(
            model_name="lgbm",
            current_mae=0.14,
            baseline_mae=0.15,
            mae_alert=False,
            psi_score=0.01,
            ks_statistic=0.05,
            ks_p_value=0.8,
            distribution_drift=False,
            n_predictions=100,
        )

        code = run(["--model", "lgbm"])
        assert code == 0

    @patch("src.monitoring.__main__.analyze_drift")
    @patch("src.monitoring.__main__.get_connection")
    @patch("src.monitoring.__main__.load_predictions")
    @patch("src.monitoring.__main__.load_baseline_metrics")
    def test_exits_one_on_alert(
        self,
        mock_metrics: MagicMock,
        mock_load: MagicMock,
        mock_conn: MagicMock,
        mock_analyze: MagicMock,
    ) -> None:
        mock_metrics.return_value = {"lgbm": {"mae": 0.15, "rmse": 0.5, "r2": 0.98}}
        mock_load.return_value = pd.DataFrame(
            {
                "station_id": ["1"] * 100,
                "target_time": pd.date_range("2026-04-01", periods=100, freq="15min"),
                "predicted_value": [5.0] * 100,
                "actual_value": [5.1] * 100,
            }
        )
        mock_conn.return_value.__enter__ = MagicMock()
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)
        mock_analyze.return_value = DriftReport(
            model_name="lgbm",
            current_mae=0.50,
            baseline_mae=0.15,
            mae_alert=True,
            psi_score=0.3,
            ks_statistic=0.5,
            ks_p_value=0.001,
            distribution_drift=True,
            n_predictions=100,
        )

        code = run(["--model", "lgbm"])
        assert code == 1

    @patch("src.monitoring.__main__.load_baseline_metrics")
    def test_exits_one_unknown_model(
        self,
        mock_metrics: MagicMock,
    ) -> None:
        mock_metrics.return_value = {"lgbm": {"mae": 0.15}}

        code = run(["--model", "unknown_model"])
        assert code == 1
