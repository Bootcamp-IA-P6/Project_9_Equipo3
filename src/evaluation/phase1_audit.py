"""Phase 1 data audit helpers (AGENTS.md — Data Collection & EDA)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

BINARY_TARGET = "IsToxic"
DEFAULT_MIN_TRAIN_POSITIVES = 5
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42


def audit_structure(df: pd.DataFrame) -> dict[str, Any]:
    """Dataset shape, dtypes, and memory footprint."""
    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
    }


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing counts and percentages."""
    missing = df.isna().sum()
    pct = (missing / len(df) * 100).round(2)
    return pd.DataFrame({"missing": missing, "pct": pct}).sort_values(
        "missing", ascending=False
    )


def text_length_stats(df: pd.DataFrame, text_col: str = "Text") -> pd.Series:
    """Character and word length summary for comment text."""
    char_len = df[text_col].astype(str).str.len()
    word_len = df[text_col].astype(str).str.split().str.len()
    return pd.Series(
        {
            "char_len_mean": char_len.mean(),
            "char_len_median": char_len.median(),
            "char_len_max": char_len.max(),
            "char_len_min": char_len.min(),
            "word_len_mean": word_len.mean(),
            "word_len_median": word_len.median(),
        }
    ).round(2)


def stratified_label_split(
    df: pd.DataFrame,
    label_cols: pd.Index | list[str],
    *,
    target_col: str = BINARY_TARGET,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split on the default binary target (Phase 3–aligned)."""
    cols = list(label_cols)
    y = df[cols]
    idx = df.index
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col],
    )
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]
    return df.loc[train_idx], df.loc[test_idx], y_train, y_test


def safe_vs_toxic_distribution(
    df: pd.DataFrame, target_col: str = BINARY_TARGET
) -> pd.DataFrame:
    """Safe (0) vs Toxic (1) counts and ratios for the default Esencial target."""
    counts = df[target_col].value_counts().sort_index()
    total = len(df)
    rows = []
    for label, count in counts.items():
        name = "Toxic" if label == 1 else "Safe"
        rows.append(
            {
                "class": name,
                "IsToxic": int(label),
                "count": int(count),
                "pct": round(100 * count / total, 2),
            }
        )
    out = pd.DataFrame(rows)
    if len(out) == 2:
        minority = out.loc[out["count"].idxmin(), "class"]
        ratio = out["count"].max() / max(out["count"].min(), 1)
        out.attrs["imbalance_ratio"] = round(ratio, 2)
        out.attrs["minority_class"] = minority
    return out


def multilabel_positive_counts(
    df: pd.DataFrame, label_cols: pd.Index | list[str]
) -> pd.Series:
    """Positive count per label column."""
    cols = list(label_cols)
    return df[cols].sum().sort_values(ascending=False).astype(int)


def sparse_label_report(
    y_subset: pd.DataFrame,
    label_cols: pd.Index | list[str],
    min_positives: int = DEFAULT_MIN_TRAIN_POSITIVES,
    *,
    subset_name: str = "train",
) -> pd.DataFrame:
    """Flag labels below minimum positive count on a given split (AGENTS: train ≥ N)."""
    counts = multilabel_positive_counts(y_subset, label_cols)
    rows = []
    for col, n in counts.items():
        rows.append(
            {
                "label": col,
                "subset": subset_name,
                "positives": int(n),
                "viable_multilabel": n >= min_positives,
            }
        )
    return pd.DataFrame(rows).sort_values("positives", ascending=False)


def label_cooccurrence(df: pd.DataFrame, label_cols: pd.Index | list[str]) -> pd.DataFrame:
    """Pairwise label co-occurrence counts (both labels = 1)."""
    cols = list(label_cols)
    mat = pd.DataFrame(0, index=cols, columns=cols, dtype=int)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if j < i:
                continue
            mat.loc[a, b] = int(((df[a] == 1) & (df[b] == 1)).sum())
            mat.loc[b, a] = mat.loc[a, b]
    return mat


def recommended_metrics(imbalance_ratio: float) -> list[str]:
    """Primary metrics when classes are imbalanced (AGENTS Phase 1)."""
    if imbalance_ratio >= 1.5:
        return ["F1-score (Toxic class)", "AUC-ROC", "Precision-Recall AUC"]
    return ["Accuracy", "F1-score (Toxic class)"]


def label_strategy_decision(
    viable_labels: list[str],
    sparse_labels: list[str],
    *,
    min_viable_for_medio: int = 3,
    max_sparse_for_medio: int = 8,
) -> dict[str, str]:
    """Esencial binary default; Medio+ multilabel only if train viability allows."""
    medio = (
        "Proceed with multilabel subset in Medio+"
        if len(viable_labels) >= min_viable_for_medio
        and len(sparse_labels) <= max_sparse_for_medio
        else "Stay on binary Safe vs Toxic until more data or labels"
    )
    return {
        "esencial_target": f"binary {BINARY_TARGET} (Safe vs Toxic)",
        "medio_plus_strategy": medio,
        "default_target": BINARY_TARGET,
    }


def augmentation_recommendation(
    n_rows: int,
    toxic_pct: float,
    sparse_labels: list[str],
) -> dict[str, Any]:
    """Phase 1 augmentation planning note."""
    needs_aug = toxic_pct < 40 or n_rows < 2000 or len(sparse_labels) > 3
    techniques = []
    if needs_aug:
        techniques = ["Synonym replacement (nlpaug)", "Back-translation (optional)"]
    return {
        "augmentation_recommended": needs_aug,
        "reason": (
            "Small dataset, class imbalance, and/or many sparse multilabel dimensions."
            if needs_aug
            else "Binary Safe/Toxic baseline is feasible without augmentation first."
        ),
        "techniques": techniques,
        "default_target": BINARY_TARGET,
    }


def save_phase1_summary(
    summary: dict[str, Any],
    path: str | Path = "reports/phase1/phase1_summary.json",
) -> Path:
    """Persist audit summary for downstream phases."""
    import json

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    return out
