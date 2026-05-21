"""Run phases 1–5 pipelines and build final comparison (uv run python -m src.pipeline.run_all_phases)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run(module: str, *args: str) -> None:
    cmd = [sys.executable, "-m", module, *args]
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    steps = [
        ("src.pipeline.phase1_run_audit", []),
        ("src.pipeline.phase2_build_processed", []),
        ("src.pipeline.phase3_train_baseline", []),
        ("src.pipeline.phase4_optimize", []),
        ("src.pipeline.plan_a_train", []),
        ("src.pipeline.plan_c_train", []),
        ("src.pipeline.plan_c_train_v2", ["--variant", "all", "--no-v2c"]),
        ("src.pipeline.plan_c_train_v3", []),
        ("src.evaluation.build_comparison", []),
        ("src.evaluation.generate_selection_report", []),
    ]
    for module, args in steps:
        _run(module, *args)

    comparison = json.loads((ROOT / "reports/phase5/comparison.json").read_text())
    selection = json.loads((ROOT / "reports/model_selection_overfitting.json").read_text())
    print("\n=== DONE ===")
    print("comparison:", ROOT / "reports/phase5/comparison.json")
    print("selection:", ROOT / "reports/model_selection_overfitting.json")
    print("api_default:", comparison.get("api_default"))
    print("verdict:", selection.get("verdict"))


if __name__ == "__main__":
    main()
