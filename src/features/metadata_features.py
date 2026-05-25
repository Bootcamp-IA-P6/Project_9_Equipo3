"""
Numeric metadata features for hybrid LR (dual-input pipeline).
"""

from __future__ import annotations

import pandas as pd


DEFAULT_METADATA_COLUMNS = [
    "char_length",
    "word_count",
    "exclamation_ratio",
    "question_ratio",
    "caps_ratio",
]


def extract_metadata_features(
    df: pd.DataFrame,
    *,
    text_column: str = "Text",
    existing_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build 3–5 numeric features for LR fusion.

    Uses columns from ``comments_with_stats`` when present; otherwise computes from text.
    """
    text = df[text_column].fillna("").astype(str)
    out = pd.DataFrame(index=df.index)

    if existing_stats is not None:
        for col in ("char_length", "word_count", "n_labels"):
            if col in existing_stats.columns:
                out[col] = existing_stats[col].values

    if "char_length" not in out.columns:
        out["char_length"] = text.str.len()
    if "word_count" not in out.columns:
        out["word_count"] = text.str.split().str.len()
    if "n_labels" not in out.columns:
        label_cols = [c for c in df.columns if c.startswith("Is") and c != "IsToxic"]
        if label_cols:
            out["n_labels"] = df[label_cols].astype(int).sum(axis=1)
        elif "IsToxic" in df.columns:
            out["n_labels"] = df["IsToxic"].astype(int)
        else:
            out["n_labels"] = 0

    length = text.str.len().clip(lower=1)
    out["exclamation_ratio"] = text.str.count("!") / length
    out["question_ratio"] = text.str.count(r"\?") / length
    out["caps_ratio"] = text.apply(
        lambda s: sum(1 for c in s if c.isupper()) / max(len(s), 1)
    )

    return out[DEFAULT_METADATA_COLUMNS].astype(float)
