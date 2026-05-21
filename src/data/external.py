"""Loads outside hate-speech datasets and maps them to our Text / IsToxic format.

Everything here feeds the training set only. The test set always stays pure
YouToxic, so the reported scores still reflect our real domain (YouTube comments).
"""
from pathlib import Path

import pandas as pd

from src.utils.config import project_root

TEXT_COLUMN = "Text"
TARGET_COLUMN = "IsToxic"

# Jigsaw tags six kinds of toxicity. We call a comment toxic if any tag is set.
_JIGSAW_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def _resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_root() / p


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Keep just Text and IsToxic, drop blank texts and repeated ones."""
    out = df[[TEXT_COLUMN, TARGET_COLUMN]].copy()
    out[TEXT_COLUMN] = out[TEXT_COLUMN].astype(str).str.strip()
    out[TARGET_COLUMN] = out[TARGET_COLUMN].astype(int)
    out = out[out[TEXT_COLUMN].str.len() > 0]
    out = out.drop_duplicates(subset=[TEXT_COLUMN])
    return out.reset_index(drop=True)


def load_davidson(path) -> pd.DataFrame:
    """Davidson tweets. The 'class' column is 0 hate, 1 offensive, 2 neither.

    Hate and offensive both count as toxic; neither counts as not toxic.
    """
    df = pd.read_csv(_resolve(path))
    mapped = pd.DataFrame({
        TEXT_COLUMN: df["tweet"],
        TARGET_COLUMN: (df["class"] != 2).astype(int),
    })
    return _tidy(mapped)


def load_jigsaw(path) -> pd.DataFrame:
    """Jigsaw Wikipedia comments. Toxic if any of the six toxicity tags is set."""
    df = pd.read_csv(_resolve(path))
    mapped = pd.DataFrame({
        TEXT_COLUMN: df["comment_text"],
        TARGET_COLUMN: df[_JIGSAW_LABELS].max(axis=1).astype(int),
    })
    return _tidy(mapped)


def sample_balanced(df: pd.DataFrame, n: int | None, random_state: int) -> pd.DataFrame:
    """Take up to n rows, split as evenly as possible between the two classes.

    This keeps the extra data from leaning too far toward one class. If n is None
    or bigger than the dataset, the whole thing comes back untouched.
    """
    if n is None or n >= len(df):
        return df
    half = n // 2
    parts = []
    for label in (0, 1):
        rows = df[df[TARGET_COLUMN] == label]
        parts.append(rows.sample(n=min(half, len(rows)), random_state=random_state))
    return pd.concat(parts, ignore_index=True)


_LOADERS = {"davidson": load_davidson, "jigsaw": load_jigsaw}


def load_external_training_data(config: dict) -> pd.DataFrame:
    """Read every external dataset switched on in the config and stack them.

    A dataset whose file is missing is skipped with a message, so training still
    runs if, say, Jigsaw has not been downloaded from Kaggle yet.
    """
    settings = config.get("external_data") or {}
    empty = pd.DataFrame(columns=[TEXT_COLUMN, TARGET_COLUMN])
    if not settings.get("enabled", False):
        return empty

    random_state = settings.get("random_state", 42)
    frames = []
    for name, loader in _LOADERS.items():
        opts = settings.get(name)
        if not opts:
            continue
        path = _resolve(opts["path"])
        if not path.exists():
            print(f"  [external] {name}: not found at {path}, skipping it.")
            continue
        data = sample_balanced(loader(path), opts.get("sample_size"), random_state)
        print(f"  [external] {name}: added {len(data)} rows ({data[TARGET_COLUMN].mean():.0%} toxic).")
        frames.append(data)

    if not frames:
        return empty
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset=[TEXT_COLUMN])
    return combined.reset_index(drop=True)
