"""CLI entry point for model monitoring and drift detection.

Usage::

    python -m src.monitoring
    python -m src.monitoring --model lgbm --report --json --since 7

Loads predictions from the database, computes drift metrics, and
optionally generates Evidently HTML reports.

Exit code 1 when MAE alert fires (>20% degradation from baseline).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

from src.monitoring.drift import analyze_drift
from src.monitoring.store import load_baseline_metrics, load_predictions
from src.storage.connection import get_connection


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


REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"


def run(argv: list[str] | None = None) -> int:
    """Run drift analysis and optionally generate reports.

    Returns
    -------
    int
        Exit code: 0 = OK, 1 = MAE alert triggered.
    """
    parser = argparse.ArgumentParser(
        description="Monitor model performance and detect drift.",
    )
    parser.add_argument(
        "--model",
        default="lgbm",
        help="Model name to analyze (default: lgbm).",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate Evidently HTML reports in reports/.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Print drift analysis as JSON to stdout.",
    )
    parser.add_argument(
        "--since",
        type=int,
        default=7,
        help="Number of days to look back (default: 7).",
    )
    args = parser.parse_args(argv)

    logger = logging.getLogger("src.monitoring")
    start = time.monotonic()

    # ------------------------------------------------------------------
    # 1. Load baseline metrics
    # ------------------------------------------------------------------
    baseline_metrics = load_baseline_metrics()
    if args.model not in baseline_metrics:
        logger.error(
            "Model '%s' not found in metrics.json. Available: %s",
            args.model,
            list(baseline_metrics.keys()),
        )
        return 1

    baseline_mae = baseline_metrics[args.model]["mae"]
    logger.info("Baseline MAE for %s: %.4f", args.model, baseline_mae)

    # ------------------------------------------------------------------
    # 2. Load predictions from database
    # ------------------------------------------------------------------
    since_dt = datetime.now(timezone.utc) - timedelta(days=args.since)

    with get_connection() as conn:
        predictions_df = load_predictions(conn, args.model, since=since_dt)

    if predictions_df.empty:
        logger.warning(
            "No predictions found for model=%s in the last %d days. "
            "Store predictions first using save_predictions().",
            args.model,
            args.since,
        )
        return 0

    # ------------------------------------------------------------------
    # 3. Analyze drift
    # ------------------------------------------------------------------
    report = analyze_drift(predictions_df, baseline_mae, args.model)

    if report is None:
        logger.warning("Insufficient data for drift analysis.")
        return 0

    # ------------------------------------------------------------------
    # 4. Generate Evidently reports (optional)
    # ------------------------------------------------------------------
    if args.report:
        from src.dataset.features import FEATURE_COLS

        try:
            import pandas as pd

            from src.monitoring.reporter import generate_all_reports

            train_path = (
                Path(__file__).resolve().parent.parent.parent
                / "data"
                / "processed"
                / "train.parquet"
            )
            if train_path.exists():
                train_df = pd.read_parquet(train_path)
                # Build reference and current with target/prediction columns
                ref_data = train_df[FEATURE_COLS].copy()
                ref_data["target"] = train_df["y"]
                ref_data["prediction"] = train_df["y"]  # placeholder

                cur_data = predictions_df[["predicted_value", "actual_value"]].copy()
                cur_data.columns = ["prediction", "target"]

                generate_all_reports(
                    ref_data,
                    cur_data,
                    REPORTS_DIR,
                    feature_cols=None,
                )
            else:
                logger.warning("train.parquet not found — skipping Evidently reports.")
        except Exception:
            logger.exception("Failed to generate Evidently reports")

    # ------------------------------------------------------------------
    # 5. Output
    # ------------------------------------------------------------------
    if args.output_json:
        print(json.dumps(report.to_dict(), indent=2))

    elapsed = time.monotonic() - start
    logger.info("Monitoring pipeline complete in %.2fs", elapsed)

    if report.mae_alert:
        logger.warning(
            "MAE ALERT: current=%.4f exceeds baseline=%.4f by >20%%",
            report.current_mae,
            report.baseline_mae,
        )
        return 1

    return 0


def main() -> None:
    """Entry point with error handling and exit-code management."""
    load_dotenv()
    _configure_logging()
    logger = logging.getLogger("src.monitoring")

    try:
        code = run()
    except Exception:
        logger.exception("Unexpected error in monitoring pipeline")
        sys.exit(1)

    sys.exit(code)


if __name__ == "__main__":
    main()
