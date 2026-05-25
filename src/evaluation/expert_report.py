"""
Integrated markdown report for Phase 5 expert runs.
"""

from __future__ import annotations

from pathlib import Path


def write_expert_report(metrics: dict, path: Path) -> None:
    run_id = metrics.get("run_id", "unknown")
    lines = [
        f"# Phase 5 Expert Adaptation — {run_id}",
        "",
        "## Targets",
        "- Test **F1-toxic** > 0.75",
        "- |Train F1-toxic − Test F1-toxic| < 5 pp (0.05)",
        "",
        "## Holdout test (tuned thresholds on validation)",
        "",
        "| Model | F1-toxic (test) | F1-toxic (train) | Toxic gap (pp) | Threshold | Gap OK |",
        "|-------|-------------------|--------------------|----------------|-----------|--------|",
    ]

    for key, label in (
        ("transformer", "Toxic-BERT"),
        ("logistic_regression", "LR-TFIDF (250 feat)"),
        ("ensemble", "Hybrid 0.7/0.3"),
    ):
        m = metrics.get(key)
        if not m:
            continue
        gap_pp = m.get(
            "train_test_gap_toxic_pp",
            m.get("train_test_gap_toxic", 0) * 100,
        )
        gap_ok = "✅" if m.get("gap_toxic_ok", gap_pp < 5) else "⚠️"
        lines.append(
            f"| {label} | {m.get('f1_toxic', '—')} | {m.get('f1_toxic_train', '—')} | "
            f"{gap_pp} | {m.get('threshold', '—')} | {gap_ok} |"
        )

    aug = metrics.get("augmentation", {})
    lines.extend(
        [
            "",
            "## Augmentation",
            f"- Pivot language: {aug.get('pivot_lang', '—')}",
            f"- Train size: {aug.get('train_size_before')} → {aug.get('train_size_after')} "
            f"(+{aug.get('added_samples', 0)})",
            "",
            "## Verdict",
        ]
    )

    verdicts = []
    for key, label in (
        ("transformer", "Toxic-BERT"),
        ("ensemble", "Hybrid"),
    ):
        m = metrics.get(key)
        if m and m.get("gap_toxic_ok"):
            verdicts.append(f"**{label}** toxic gap < 5 pp ✅")
        elif m:
            gap = m.get("train_test_gap_toxic_pp", "?")
            verdicts.append(f"**{label}** toxic gap {gap} pp ⚠️")

    lines.append(
        "; ".join(verdicts) if verdicts else "No model metrics recorded."
    )
    lines.extend(["", f"- JSON: `reports/expert/expert_run_{run_id}.json`", ""])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
