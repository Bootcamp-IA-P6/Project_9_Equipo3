"""Build reports/phase5/comparison.json only from fresh pipeline report files."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.evaluation.overfitting_check import MAX_GAP_PCT

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports/phase5/comparison.json"


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _row_from_metrics_report(
    name: str,
    report: dict[str, Any],
    *,
    path_key: str = "model_path",
    api_default: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = report["metrics"]
    acc = metrics["accuracy"]
    f1 = metrics["f1_toxic"]
    row = {
        "name": name,
        "path": report.get(path_key) or report.get("model_dir"),
        "phase": report.get("phase") or report.get("plan"),
        "test_accuracy": acc["test"],
        "test_f1_toxic": f1["test"],
        "train_accuracy": acc["train"],
        "train_f1_toxic": f1["train"],
        "accuracy_gap_pct": acc["gap_pct"],
        "f1_gap_pct": f1["gap_pct"],
        "accuracy_pass": acc["passes"],
        "f1_pass": f1["passes"],
        "passes_overfitting": bool(
            report.get("passes_overfitting", metrics.get("passes_overfitting"))
        ),
        "api_default": api_default,
        "status": "PASSED" if report.get("passes_overfitting", metrics.get("passes_overfitting")) else "FAILED",
        "mlflow_run": report.get("mlflow_run"),
    }
    if extra:
        row.update(extra)
    return row


def build_comparison() -> dict[str, Any]:
    models: list[dict[str, Any]] = []
    missing: list[str] = []

    p3 = _load(ROOT / "reports/phase3/phase3_training.json")
    if p3:
        models.append(
            _row_from_metrics_report(
                f"phase3_{p3['best_model']}",
                {
                    **p3,
                    "metrics": p3["metrics"],
                    "passes_overfitting": p3["passes_overfitting"],
                },
            )
        )
    else:
        missing.append("phase3")

    p4 = _load(ROOT / "reports/phase4/phase4_optimization.json")
    if p4:
        models.append(
            _row_from_metrics_report(
                f"phase4_{p4['best_model']}",
                {
                    **p4,
                    "metrics": p4["metrics"],
                    "passes_overfitting": p4["passes_overfitting"],
                },
                api_default=False,
            )
        )
    else:
        missing.append("phase4")

    plan_a = _load(ROOT / "reports/phase5/plan_a_report.json")
    if plan_a:
        models.append(
            _row_from_metrics_report(
                f"plan_a_{plan_a.get('best_model', 'hybrid')}",
                plan_a,
                path_key="model_path",
            )
        )
    else:
        missing.append("plan_a")

    plan_c = _load(ROOT / "reports/phase5/plan_c_report.json")
    if plan_c:
        models.append(
            _row_from_metrics_report(
                "plan_c_distilbert_v1",
                plan_c,
                path_key="model_dir",
                extra={"base_model": plan_c.get("base_model")},
            )
        )
    else:
        missing.append("plan_c_v1")

    v2 = _load(ROOT / "reports/phase5/plan_c_v2_report.json")
    if v2:
        for variant, exp in v2.get("experiments", {}).items():
            models.append(
                _row_from_metrics_report(
                    f"plan_c_v2_{variant}",
                    exp,
                    path_key="model_dir",
                    extra={"status": exp.get("status")},
                )
            )
    else:
        missing.append("plan_c_v2")

    v3 = _load(ROOT / "reports/phase5/plan_c_v3_report.json")
    if v3:
        models.append(
            _row_from_metrics_report(
                "plan_c_v3",
                v3,
                path_key="model_dir",
                extra={"status": v3.get("status")},
            )
        )
    else:
        missing.append("plan_c_v3")

    compliant = [m for m in models if m["passes_overfitting"]]
    compliant.sort(key=lambda m: m["test_accuracy"], reverse=True)

    for m in models:
        m["api_default"] = False
    if compliant:
        compliant[0]["api_default"] = True
        api_default_name = compliant[0]["name"]
        api_default_path = compliant[0]["path"]
    else:
        api_default_name = None
        api_default_path = None

    all_by_acc = sorted(models, key=lambda m: m["test_accuracy"], reverse=True)
    highest_any = all_by_acc[0] if all_by_acc else None

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overfitting_rule": f"|train - test| < {MAX_GAP_PCT:g} percentage points (accuracy AND f1_toxic)",
        "split_note": (
            "Phase 3–4: data/processed/youtoxic_phase2.csv split column. "
            "Plan A/C/v2/v3: raw CSV stratified split (random_state=42, test_size=0.2)."
        ),
        "missing_reports": missing,
        "models": models,
        "compliant_ranked_by_test_accuracy": [
            {k: m[k] for k in ("name", "path", "test_accuracy", "test_f1_toxic", "accuracy_gap_pct", "f1_gap_pct")}
            for m in compliant
        ],
        "api_default": {
            "name": api_default_name,
            "path": api_default_path,
        },
        "highest_test_accuracy_any": {
            "name": highest_any["name"] if highest_any else None,
            "test_accuracy": highest_any["test_accuracy"] if highest_any else None,
            "passes_overfitting": highest_any["passes_overfitting"] if highest_any else None,
        },
        "recommendation": {
            "production_if_gap_required": (
                f"{api_default_name} ({api_default_path})"
                if api_default_name
                else "No compliant model — review phase4_optimize or relax rule"
            ),
            "highest_accuracy_overall": (
                f"{highest_any['name']} test_acc={highest_any['test_accuracy']:.1%} "
                f"passes_gap={highest_any['passes_overfitting']}"
                if highest_any
                else None
            ),
        },
    }


def write_comparison() -> Path:
    data = build_comparison()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return OUTPUT


if __name__ == "__main__":
    path = write_comparison()
    print(path.read_text(encoding="utf-8"))
