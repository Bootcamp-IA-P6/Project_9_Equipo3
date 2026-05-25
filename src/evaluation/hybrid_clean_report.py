"""Markdown report for Clean-Signal Dual-Input Hybrid runs."""

from __future__ import annotations

from pathlib import Path


def write_hybrid_clean_report(metrics: dict, path: Path) -> None:
    run_id = metrics.get("run_id", "unknown")
    target = float(metrics.get("target_f1_weighted", 0.80))
    ens = metrics.get("ensemble", {})
    f1w = ens.get("f1_weighted", 0)
    hit = "✅" if f1w >= target else "⚠️"

    lines = [
        f"# Clean-Signal Hybrid — {run_id}",
        "",
        f"## Target: integrated F1 (weighted) ≥ {target} — {hit} **{f1w}**",
        "",
        "| Model | F1 weighted (test) | F1 toxic (test) | Gap (pp) | Threshold |",
        "|-------|-------------------|-----------------|----------|-----------|",
    ]

    for key, label in (
        ("transformer", "Toxic-BERT (raw Text)"),
        ("logistic_regression", "LR (clean_text + meta)"),
        ("ensemble", "Dual hybrid"),
    ):
        m = metrics.get(key, {})
        if not m:
            continue
        gap_pp = m.get("train_test_gap_pp", m.get("train_test_gap", 0) * 100)
        lines.append(
            f"| {label} | {m.get('f1_weighted', '—')} | {m.get('f1_toxic', '—')} | "
            f"{gap_pp} | {m.get('threshold', '—')} |"
        )

    w = metrics.get("ensemble_weights", {})
    if w:
        lines.extend(
            [
                "",
                f"**Dynamic weights (val):** BERT={w.get('bert_weight')} LR={w.get('lr_weight')} "
                f"(val F1 weighted: BERT={w.get('bert_val_score')} LR={w.get('lr_val_score')})",
            ]
        )

    lines.extend(
        [
            "",
            f"- JSON: `reports/hybrid_clean/hybrid_clean_run_{run_id}.json`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
