"""Notebook-friendly overfitting evaluation tables (AGENTS.md gap rule)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.evaluation.overfitting_check import MAX_GAP_PCT


def metrics_to_eval_rows(metrics: dict[str, Any], model_name: str) -> list[dict[str, Any]]:
    """Turn a standard ``metrics`` block into evaluation table rows."""
    rows: list[dict[str, Any]] = []
    for key in ("accuracy", "f1_toxic"):
        block = metrics.get(key)
        if not block:
            continue
        rows.append(
            {
                "model": model_name,
                "metric": key,
                "train": block["train"],
                "test": block["test"],
                "gap_pp": block["gap_pct"],
                "pass": "PASS" if block.get("passes") else "FAIL",
            }
        )
    return rows


def report_to_evaluation_df(
    report: dict[str, Any],
    *,
    model_name: str | None = None,
    include_all_models: bool = True,
) -> pd.DataFrame:
    """Build evaluation DataFrame from a pipeline report JSON object."""
    name = model_name or report.get("best_model") or report.get("variant") or "selected"
    rows = metrics_to_eval_rows(report["metrics"], name)

    if include_all_models:
        for entry in report.get("all_models", []):
            entry_name = entry.get("name", "model")
            if "metrics" in entry:
                rows.extend(metrics_to_eval_rows(entry["metrics"], entry_name))
            else:
                rows.append(
                    {
                        "model": entry_name,
                        "metric": "f1_toxic (summary)",
                        "train": None,
                        "test": entry.get("test_f1_toxic"),
                        "gap_pp": entry.get("f1_gap_pct"),
                        "pass": "PASS" if entry.get("passes_overfitting") else "FAIL",
                    }
                )

    return pd.DataFrame(rows)


def experiments_to_evaluation_df(experiments: dict[str, Any]) -> pd.DataFrame:
    """Build evaluation table from Plan C v2 ``experiments`` dict."""
    rows: list[dict[str, Any]] = []
    for variant, exp in experiments.items():
        rows.extend(metrics_to_eval_rows(exp["metrics"], variant))
    return pd.DataFrame(rows)


def overall_pass_label(report: dict[str, Any]) -> str:
    return "PASS" if report.get("passes_overfitting") else "FAIL"


def print_evaluation_banner(phase_label: str) -> None:
    print("=" * 72)
    print(f"{phase_label} — OVERFITTING EVALUATION")
    print(f"Rule: |train − test| < {MAX_GAP_PCT:g} percentage points (accuracy & F1 toxic)")
    print("=" * 72)
