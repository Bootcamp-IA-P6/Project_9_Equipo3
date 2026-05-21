"""Build Phase 2 processed dataset and vectorizers (CLI: uv run python -m src.pipeline.phase2_build_processed)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.loader import load_processed_data
from src.evaluation.phase1_audit import (
    BINARY_TARGET,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    stratified_label_split,
)
from src.features.phase2_text_pipeline import process_text
from src.features.phase2_vectorization import (
    fit_vectorizers,
    save_vectorizers,
    vectorizer_summary,
)
from src.utils.execution_log import log_event

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data/raw/youtoxic_english_1000.csv"
PROCESSED_CSV = ROOT / "data/processed/youtoxic_phase2.csv"
VECTORIZER_DIR = ROOT / "data/processed/vectorizers"
REPORT_PATH = ROOT / "reports/phase2/phase2_summary.json"
LOG_PATH = ROOT / "logs/execution.log"


def build_phase2_dataset() -> dict:
    """Run full Phase 2 pipeline and write artifacts."""
    log_event("Phase 2 build started", phase="2", log_path=LOG_PATH)

    df, label_cols = load_processed_data(str(RAW_PATH))
    _, _, y_train, y_test = stratified_label_split(
        df,
        label_cols,
        test_size=DEFAULT_TEST_SIZE,
        random_state=DEFAULT_RANDOM_STATE,
    )
    train_idx = y_train.index
    test_idx = y_test.index

    processed_rows = []
    for idx, row in df.iterrows():
        clean, lemmas = process_text(row["Text"])
        split = "train" if idx in train_idx else "test"
        processed_rows.append(
            {
                "Text": row["Text"],
                "CleanText": clean,
                "ProcessedText": lemmas,
                BINARY_TARGET: int(row[BINARY_TARGET]),
                "split": split,
            }
        )

    out_df = pd.DataFrame(processed_rows)
    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(PROCESSED_CSV, index=False)

    train_texts = out_df.loc[out_df["split"] == "train", "ProcessedText"].tolist()
    vec_bundle = fit_vectorizers(train_texts)
    paths = save_vectorizers(
        vec_bundle["tfidf_vectorizer"],
        vec_bundle["count_vectorizer"],
        VECTORIZER_DIR,
    )

    empty_processed = int((out_df["ProcessedText"].str.len() == 0).sum())
    summary = {
        "phase": 2,
        "source": RAW_PATH.name,
        "processed_csv": str(PROCESSED_CSV.relative_to(ROOT)),
        "n_rows": len(out_df),
        "n_train": int((out_df["split"] == "train").sum()),
        "n_test": int((out_df["split"] == "test").sum()),
        "target": BINARY_TARGET,
        "empty_processed_text": empty_processed,
        "vectorizers": {k: str(v.relative_to(ROOT)) for k, v in paths.items()},
        "vectorizer_stats": vectorizer_summary(
            vec_bundle["tfidf_vectorizer"],
            vec_bundle["count_vectorizer"],
        ),
        "tfidf_train_shape": list(vec_bundle["tfidf_shape"]),
        "count_train_shape": list(vec_bundle["count_shape"]),
        "augmentation_note": (
            "Deferred to training phase (Phase 3/4); synonym aug available in src.features.augmentation"
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log_event(f"Phase 2 processed CSV saved: {PROCESSED_CSV}", phase="2", log_path=LOG_PATH)
    log_event(f"Phase 2 vectorizers saved under {VECTORIZER_DIR}", phase="2", log_path=LOG_PATH)
    log_event("Phase 2 build completed", phase="2", log_path=LOG_PATH)
    return summary


if __name__ == "__main__":
    result = build_phase2_dataset()
    print(json.dumps(result, indent=2))
