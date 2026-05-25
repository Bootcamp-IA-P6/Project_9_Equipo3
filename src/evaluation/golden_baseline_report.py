"""Briefing-aligned report: Golden Baseline (gap <1%) + Final Hybrid (gap <5%, F1≥0.80)."""

from __future__ import annotations

from pathlib import Path


def write_golden_baseline_report(metrics: dict, path: Path) -> None:
    run_id = metrics.get("run_id", "unknown")
    target = float(metrics.get("target_f1_weighted", 0.80))
    max_gap_pp = float(metrics.get("max_train_test_gap_pp", 5.0))
    baseline_gap_target_pp = float(metrics.get("baseline_gap_target_pp", 1.0))

    base = metrics.get("golden_baseline", {})
    squeeze = metrics.get("performance_squeeze", {})
    hybrid = metrics.get("hybrid_safety_net", {})

    base_gap = base.get("train_test_gap_pp", 99)
    hybrid_gap = hybrid.get("train_test_gap_pp", 99)
    hybrid_f1 = hybrid.get("f1_weighted", 0)

    base_ok = base_gap < baseline_gap_target_pp
    hybrid_gap_ok = hybrid_gap < max_gap_pp
    hybrid_f1_ok = hybrid_f1 >= target
    hybrid_ok = hybrid_gap_ok and hybrid_f1_ok

    lines = [
        f"# Golden Baseline Strategy — {run_id}",
        "",
        "Two-step briefing alignment: **Esencial** frozen expert baseline, then **Experto** squeeze + hybrid.",
        "",
        "## Step 1 — Golden Baseline (Esencial)",
        "",
        f"| Metric | Value | Target |",
        f"|--------|-------|--------|",
        f"| F1 weighted (test) | **{base.get('f1_weighted', '—')}** | ~0.72 (pretrained expert) |",
        f"| Train–test gap (pp) | **{base_gap}** | < {baseline_gap_target_pp}% {'✅' if base_ok else '⚠️'} |",
        f"| Fine-tuning | None (all layers frozen) | — |",
        f"| Threshold | {base.get('threshold', '—')} | val-tuned |",
        "",
        "## Step 2 — Performance Squeeze (Experto)",
        "",
        f"| Metric | Value | Target |",
        f"|--------|-------|--------|",
        f"| F1 weighted (test) | **{squeeze.get('f1_weighted', '—')}** | ≥ {target} |",
        f"| Train–test gap (pp) | **{squeeze.get('train_test_gap_pp', '—')}** | ≤ 4.9% |",
        f"| R-Drop | {squeeze.get('rdrop_enabled', False)} | enabled |",
        f"| Layers trained | last {squeeze.get('freeze_mode', '—')} | 2 + head |",
        "",
        "## Step 3 — Hybrid Safety Net (Final)",
        "",
        f"| Metric | Value | Target |",
        f"|--------|-------|--------|",
        f"| F1 weighted (test) | **{hybrid_f1}** | ≥ {target} {'✅' if hybrid_f1_ok else '⚠️'} |",
        f"| Train–test gap (pp) | **{hybrid_gap}** | < {max_gap_pp}% {'✅' if hybrid_gap_ok else '⚠️'} |",
        f"| Weights | BERT {hybrid.get('bert_weight')} / LR {hybrid.get('lr_weight')} | anchor |",
        f"| LR regularization | C=0.001, max_features=200 | stability |",
        "",
        f"### Overall: {'✅ Briefing targets met' if base_ok and hybrid_ok else '⚠️ Review gaps / F1'}",
        "",
        f"- JSON: `reports/golden_baseline/golden_baseline_run_{run_id}.json`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
