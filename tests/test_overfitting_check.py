"""Unit tests for overfitting gap helpers."""

import pytest

from src.evaluation.overfitting_check import (
    format_gap_report,
    metric_gap_pct,
    passes_overfitting_check,
)


def test_metric_gap_pct_absolute():
    assert metric_gap_pct(0.92, 0.78) == pytest.approx(14.0)


def test_passes_when_gap_at_threshold():
    assert passes_overfitting_check(0.83, 0.78, max_gap_pct=5.0) is True
    assert passes_overfitting_check(0.84, 0.78, max_gap_pct=5.0) is False


def test_format_gap_report_keys():
    report = format_gap_report("f1_toxic", 0.82, 0.78)
    assert report["metric"] == "f1_toxic"
    assert report["gap_pct"] == pytest.approx(4.0)
    assert report["passes"] is True
