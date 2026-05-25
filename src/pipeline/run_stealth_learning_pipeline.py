"""
Stealth Learning — focused unfreeze, SWA, micro-threshold, stable LR anchor.

    uv run python -m src.pipeline.run_stealth_learning_pipeline
    uv run python -m src.pipeline.run_stealth_learning_pipeline --skip-augmentation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.run_performance_push_pipeline import run_performance_push_pipeline
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stealth_learning_training.yaml"


def main():
    parser = argparse.ArgumentParser(description="Stealth Learning pipeline")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--skip-augmentation", action="store_true")
    args = parser.parse_args()
    run_performance_push_pipeline(
        config_path=Path(args.config) if args.config else DEFAULT_CONFIG,
        skip_augmentation=args.skip_augmentation,
    )


if __name__ == "__main__":
    main()
