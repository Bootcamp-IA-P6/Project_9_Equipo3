"""
Integrated markdown report for stable production runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_integrated_report(metrics: dict, path: Path) -> None:
    """Write human-readable report with scores and gaps."""
    run_id = metrics.get("run_id", "unknown")
    lines = [
        f"# Stable Production Run — {run_id}",
        "",
        "## Targets",
        "- Test F1 (weighted) > 0.80",
        "- |Train F1 − Test/Val F1| < 5 pp (0.05)",
        "",
        "## Holdout test (final models)",
        "",
        "| Model | F1 (test) | F1 (train) | Train−Test gap (pp) | ROC-AUC | Gap OK |",
        "|-------|-----------|------------|---------------------|---------|--------|",
    ]

    for key, label in (
        ("distilbert", "DistilBERT"),
        ("logistic_regression", "LR-TFIDF"),
        ("ensemble", "Hybrid ensemble"),
    ):
        m = metrics.get(key)
        if not m:
            continue
        gap_pp = m.get("train_test_gap_pp", m.get("train_test_gap", 0) * 100)
        gap_ok = "✅" if gap_pp < 5 else "⚠️"
        lines.append(
            f"| {label} | {m.get('f1_weighted', '—')} | {m.get('f1_train', '—')} | "
            f"{gap_pp} | {m.get('roc_auc', '—')} | {gap_ok} |"
        )

    lr_gap = metrics.get("lr_gap_search")
    if lr_gap:
        lines.extend(
            [
                "",
                f"**LR gap search:** C={lr_gap.get('C')}, max_features={lr_gap.get('max_features')}, "
                f"min_df={lr_gap.get('min_df')} — gap {lr_gap.get('train_test_gap_pp')} pp "
                f"({'OK' if lr_gap.get('gap_ok') else 'above target'})",
            ]
        )

    lines.extend(["", "## Stratified 5-fold CV (train+val pool)", ""])

    for cv_key, title in (
        ("cv_logistic_regression", "LR-TFIDF"),
        ("cv_distilbert", "DistilBERT"),
    ):
        cv = metrics.get(cv_key)
        if not cv:
            continue
        stable = "✅" if cv.get("stable_across_folds") else "⚠️"
        lines.extend(
            [
                f"### {title} {stable}",
                "",
                f"- F1 mean ± std: **{cv.get('f1_mean')} ± {cv.get('f1_std')}** "
                f"(min {cv.get('f1_min')}, max {cv.get('f1_max')})",
                f"- Fold gap mean: {cv.get('gap_mean')} (max {cv.get('gap_max')})",
                f"- ROC-AUC mean: {cv.get('roc_auc_mean')}",
                "",
            ]
        )

    lines.extend(
        [
            "## Augmentation",
            "",
            f"- Enabled: {metrics.get('augmentation', {}).get('enabled', True)}",
            f"- Train size after aug: {metrics.get('augmentation', {}).get('train_size_after', '—')}",
            "",
            "## Artifacts",
            "",
            f"- JSON: `reports/stable/stable_run_{run_id}.json`",
            f"- CSV: `reports/stable/stable_summary_{run_id}.csv`",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
