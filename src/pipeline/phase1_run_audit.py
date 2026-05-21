"""Phase 1 audit summary without notebook (uv run python -m src.pipeline.phase1_run_audit)."""

from __future__ import annotations

import json
from pathlib import Path

from src.data.loader import load_processed_data
from src.evaluation.phase1_audit import (
    BINARY_TARGET,
    DEFAULT_MIN_TRAIN_POSITIVES,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    augmentation_recommendation,
    audit_structure,
    label_strategy_decision,
    multilabel_positive_counts,
    recommended_metrics,
    safe_vs_toxic_distribution,
    save_phase1_summary,
    sparse_label_report,
    stratified_label_split,
    text_length_stats,
)
from src.utils.execution_log import log_event

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/raw/youtoxic_english_1000.csv"
REPORT_PATH = ROOT / "reports/phase1/phase1_summary.json"
LOG_PATH = ROOT / "logs/execution.log"


def run_phase1_audit() -> dict:
    log_event("Phase 1 audit started", phase="1", log_path=LOG_PATH)
    df, label_cols = load_processed_data(str(DATA_PATH))
    structure = audit_structure(df)
    df_train, df_test, _, _ = stratified_label_split(df, label_cols)
    length_stats = text_length_stats(df)
    binary_dist = safe_vs_toxic_distribution(df)
    toxic_pct = float(binary_dist.loc[binary_dist["IsToxic"] == 1, "pct"].iloc[0])
    imbalance_ratio = binary_dist.attrs.get("imbalance_ratio")
    y_train = df_train[label_cols]
    label_counts = multilabel_positive_counts(y_train, label_cols)
    label_counts_full = multilabel_positive_counts(df[label_cols], label_cols)
    sparse_df = sparse_label_report(y_train, label_cols, min_positives=DEFAULT_MIN_TRAIN_POSITIVES)
    sparse_labels = sparse_df.loc[~sparse_df["viable_multilabel"], "label"].tolist()
    viable_labels = sparse_df.loc[sparse_df["viable_multilabel"], "label"].tolist()
    strategy = label_strategy_decision(viable_labels, sparse_labels)
    aug = augmentation_recommendation(len(df), toxic_pct, sparse_labels)
    metrics = recommended_metrics(imbalance_ratio or 1.0)

    summary = {
        "phase": 1,
        "dataset": DATA_PATH.name,
        "structure": structure,
        "train_test_split": {
            "test_size": DEFAULT_TEST_SIZE,
            "random_state": DEFAULT_RANDOM_STATE,
            "n_train": len(df_train),
            "n_test": len(df_test),
        },
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_text": int(df["Text"].duplicated().sum()),
        "text_length": length_stats.to_dict(),
        "safe_vs_toxic": binary_dist.to_dict(orient="records"),
        "imbalance_ratio": imbalance_ratio,
        "recommended_metrics": metrics,
        "multilabel_positives_full": label_counts_full.to_dict(),
        "multilabel_positives_train": label_counts.to_dict(),
        "sparse_labels_train": sparse_labels,
        "viable_multilabel_labels_train": viable_labels,
        "augmentation": aug,
        **strategy,
    }
    save_phase1_summary(summary, REPORT_PATH)
    log_event("Phase 1 audit completed", phase="1", log_path=LOG_PATH)
    return summary


if __name__ == "__main__":
    print(json.dumps(run_phase1_audit(), indent=2, default=str))
