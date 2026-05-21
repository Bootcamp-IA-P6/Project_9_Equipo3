"""Overfitting gap checks (AGENTS.md: |Train - Test| < 5%)."""

from __future__ import annotations

MAX_GAP_PCT = 5.0


def metric_gap_pct(train_value: float, test_value: float) -> float:
    """Absolute gap in percentage points (e.g. 0.82 vs 0.78 → 4.0)."""
    return abs((train_value - test_value) * 100.0)


def passes_overfitting_check(
    train_value: float,
    test_value: float,
    max_gap_pct: float = MAX_GAP_PCT,
) -> bool:
    return metric_gap_pct(train_value, test_value) <= max_gap_pct


def format_gap_report(
    metric_name: str,
    train_value: float,
    test_value: float,
) -> dict[str, float | bool | str]:
    gap = metric_gap_pct(train_value, test_value)
    return {
        "metric": metric_name,
        "train": round(train_value, 4),
        "test": round(test_value, 4),
        "gap_pct": round(gap, 2),
        "passes": gap <= MAX_GAP_PCT,
    }
