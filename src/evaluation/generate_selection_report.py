"""Build overfitting-compliant model selection report (reports/model_selection_overfitting.json)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.evaluation.overfitting_check import MAX_GAP_PCT

ROOT = Path(__file__).resolve().parents[2]
COMPARISON_PATH = ROOT / "reports/phase5/comparison.json"
OUTPUT_PATH = ROOT / "reports/model_selection_overfitting.json"

REPORT_BY_PREFIX: list[tuple[str, Path]] = [
    ("phase4_", ROOT / "reports/phase4/phase4_optimization.json"),
    ("phase3_", ROOT / "reports/phase3/phase3_training.json"),
    ("plan_a_", ROOT / "reports/phase5/plan_a_report.json"),
    ("plan_c_distilbert_v1", ROOT / "reports/phase5/plan_c_report.json"),
    ("plan_c_v3", ROOT / "reports/phase5/plan_c_v3_report.json"),
]


def _load_json(path: Path) -> dict[str, Any] | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _enrich_from_detail(entry: dict[str, Any]) -> dict[str, Any]:
    """Merge train/gap fields from a pipeline report when available."""
    name = entry["name"]
    detail_path = None
    for prefix, path in REPORT_BY_PREFIX:
        if name.startswith(prefix) or name == prefix:
            detail_path = path
            break
    detail = _load_json(detail_path) if detail_path else None
    if not detail:
        v2 = _load_json(ROOT / "reports/phase5/plan_c_v2_report.json")
        if v2 and name.startswith("plan_c_v2_"):
            variant = name.replace("plan_c_v2_", "")
            exp = v2.get("experiments", {}).get(variant)
            if exp:
                detail = exp
    if not detail:
        return entry

    metrics = detail.get("metrics", {})
    if metrics:
        entry["train_accuracy"] = metrics.get("accuracy", {}).get("train")
        entry["train_f1_toxic"] = metrics.get("f1_toxic", {}).get("train")
        entry["accuracy_gap_pp"] = entry.get("accuracy_gap_pct") or metrics.get(
            "accuracy", {}
        ).get("gap_pct")
        entry["f1_gap_pp"] = entry.get("f1_gap_pct") or metrics.get("f1_toxic", {}).get(
            "gap_pct"
        )
        entry["accuracy_pass"] = metrics.get("accuracy", {}).get("passes")
        entry["f1_pass"] = metrics.get("f1_toxic", {}).get("passes")
    return entry


def build_selection_report() -> dict[str, Any]:
    comparison = _load_json(COMPARISON_PATH)
    if not comparison:
        raise FileNotFoundError(f"Missing {COMPARISON_PATH}")

    all_models: list[dict[str, Any]] = []
    for raw in comparison.get("models", []):
        row = {
            "name": raw["name"],
            "path": raw.get("path"),
            "test_accuracy": raw.get("test_accuracy"),
            "test_f1_toxic": raw.get("test_f1_toxic"),
            "passes_overfitting": bool(raw.get("passes_overfitting")),
            "api_default": raw.get("api_default", False),
            "accuracy_gap_pp": raw.get("accuracy_gap_pct"),
            "f1_gap_pp": raw.get("f1_gap_pct"),
            "status": raw.get("status"),
            "mlflow_run": raw.get("mlflow_run"),
        }
        all_models.append(_enrich_from_detail(row))

    compliant = [m for m in all_models if m["passes_overfitting"]]
    non_compliant = [m for m in all_models if not m["passes_overfitting"]]

    compliant_by_acc = sorted(
        compliant, key=lambda m: (m.get("test_accuracy") or 0), reverse=True
    )
    compliant_by_f1 = sorted(
        compliant, key=lambda m: (m.get("test_f1_toxic") or 0), reverse=True
    )
    all_by_acc = sorted(all_models, key=lambda m: (m.get("test_accuracy") or 0), reverse=True)

    best_compliant_acc = compliant_by_acc[0] if compliant_by_acc else None
    best_compliant_f1 = compliant_by_f1[0] if compliant_by_f1 else None
    highest_acc_any = all_by_acc[0] if all_by_acc else None

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overfitting_rule": f"|train - test| < {MAX_GAP_PCT:g} percentage points (accuracy AND f1_toxic)",
        "summary": {
            "best_compliant_test_accuracy": {
                "name": best_compliant_acc["name"] if best_compliant_acc else None,
                "test_accuracy": best_compliant_acc.get("test_accuracy") if best_compliant_acc else None,
                "test_f1_toxic": best_compliant_acc.get("test_f1_toxic") if best_compliant_acc else None,
                "path": best_compliant_acc.get("path") if best_compliant_acc else None,
                "recommendation": (
                    "Use as production default when Esencial overfitting rule is required."
                    if best_compliant_acc
                    else None
                ),
            },
            "best_compliant_test_f1_toxic": {
                "name": best_compliant_f1["name"] if best_compliant_f1 else None,
                "test_f1_toxic": best_compliant_f1.get("test_f1_toxic") if best_compliant_f1 else None,
                "test_accuracy": best_compliant_f1.get("test_accuracy") if best_compliant_f1 else None,
            },
            "highest_test_accuracy_any_model": {
                "name": highest_acc_any["name"] if highest_acc_any else None,
                "test_accuracy": highest_acc_any.get("test_accuracy") if highest_acc_any else None,
                "passes_overfitting": highest_acc_any.get("passes_overfitting") if highest_acc_any else None,
                "note": (
                    "Does not pass overfitting rule — not eligible for Esencial-compliant default."
                    if highest_acc_any and not highest_acc_any.get("passes_overfitting")
                    else None
                ),
            },
            "n_models_total": len(all_models),
            "n_compliant": len(compliant),
            "n_non_compliant": len(non_compliant),
        },
        "compliant_models_ranked_by_test_accuracy": compliant_by_acc,
        "compliant_models_ranked_by_test_f1": compliant_by_f1,
        "compliant_comparison_table": [
            {
                "rank": i + 1,
                "name": m["name"],
                "test_accuracy_pct": round((m.get("test_accuracy") or 0) * 100, 2),
                "test_f1_toxic_pct": round((m.get("test_f1_toxic") or 0) * 100, 2),
                "accuracy_gap_pp": m.get("accuracy_gap_pp"),
                "f1_gap_pp": m.get("f1_gap_pp"),
                "usable_for_moderation": m["name"] != "plan_c_v2_v2b",
                "path": m.get("path"),
            }
            for i, m in enumerate(compliant_by_acc)
        ],
        "all_models": all_models,
        "non_compliant_notable": sorted(
            non_compliant,
            key=lambda m: (m.get("test_accuracy") or 0),
            reverse=True,
        )[:3],
        "verdict": (
            f"{best_compliant_acc['name']} has the highest test accuracy ({best_compliant_acc['test_accuracy']:.1%}) "
            f"among models that pass the {MAX_GAP_PCT:g}% overfitting rule. "
            f"Recommended API default: {best_compliant_acc.get('path')}."
            if best_compliant_acc
            else "No model passes the overfitting rule."
        ),
    }


def write_selection_report() -> Path:
    report = build_selection_report()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return OUTPUT_PATH


if __name__ == "__main__":
    path = write_selection_report()
    print(path.read_text(encoding="utf-8"))
