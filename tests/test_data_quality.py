"""Tests for src.storage.data_quality."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.storage.data_quality import (
    CheckResult,
    all_passed,
    main,
    results_to_json,
    run_checks,
)


def _conn_with_counts(counts: list[int]) -> MagicMock:
    conn = MagicMock()
    cursor = MagicMock()
    call_iter = iter(counts)

    def fetchone() -> tuple[int]:
        return (next(call_iter),)

    cursor.fetchone = fetchone
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn


class TestRunChecks:
    """Tests for run_checks()."""

    def test_all_pass_when_zero(self) -> None:
        conn = _conn_with_counts([0, 0, 0, 0])
        results = run_checks(conn)
        assert len(results) == 4
        assert all(r.passed for r in results)

    def test_fails_when_any_positive(self) -> None:
        conn = _conn_with_counts([0, 2, 0, 0])
        results = run_checks(conn)
        assert results[1].name == "negative_availability"
        assert results[1].violation_count == 2
        assert not results[1].passed
        assert not all_passed(results)


class TestResultsToJson:
    """Tests for results_to_json()."""

    def test_serializes_structure(self) -> None:
        results = [
            CheckResult(name="a", violation_count=0, passed=True),
            CheckResult(name="b", violation_count=1, passed=False),
        ]
        text = results_to_json(results)
        assert '"overall_passed": false' in text
        assert "b" in text


class TestMainCli:
    """Tests for main() CLI entry point."""

    @patch("src.storage.connection.get_connection")
    @patch("dotenv.load_dotenv")
    def test_exits_zero_when_all_pass(
        self,
        mock_load_dotenv: MagicMock,
        mock_get_connection: MagicMock,
    ) -> None:
        conn = _conn_with_counts([0, 0, 0, 0])
        mock_get_connection.return_value.__enter__ = MagicMock(return_value=conn)
        mock_get_connection.return_value.__exit__ = MagicMock(return_value=False)

        with patch("src.storage.data_quality.logging.basicConfig"):
            code = main([])

        assert code == 0

    @patch("src.storage.connection.get_connection")
    @patch("dotenv.load_dotenv")
    def test_exits_one_when_check_fails(
        self,
        mock_load_dotenv: MagicMock,
        mock_get_connection: MagicMock,
    ) -> None:
        conn = _conn_with_counts([1, 0, 0, 0])
        mock_get_connection.return_value.__enter__ = MagicMock(return_value=conn)
        mock_get_connection.return_value.__exit__ = MagicMock(return_value=False)

        with patch("src.storage.data_quality.logging.basicConfig"):
            code = main([])

        assert code == 1

    @patch("src.storage.connection.get_connection")
    @patch("dotenv.load_dotenv")
    def test_json_flag_prints_line(
        self,
        mock_load_dotenv: MagicMock,
        mock_get_connection: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        conn = _conn_with_counts([0, 0, 0, 0])
        mock_get_connection.return_value.__enter__ = MagicMock(return_value=conn)
        mock_get_connection.return_value.__exit__ = MagicMock(return_value=False)

        with patch("src.storage.data_quality.logging.basicConfig"):
            main(["--json"])

        captured = capsys.readouterr()
        assert "overall_passed" in captured.out
