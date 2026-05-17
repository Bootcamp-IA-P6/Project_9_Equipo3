from pathlib import Path

import pandas as pd

from src.utils.config import project_root

ID_COLUMNS = ("CommentId", "VideoId")
TEXT_COLUMN = "Text"
TARGET_COLUMN = "IsToxic"


def load_dataset(raw_path: str | Path) -> pd.DataFrame:
    path = Path(raw_path)
    if not path.is_absolute():
        path = project_root() / path

    return pd.read_csv(path, encoding="utf-8-sig")


def without_ids(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in ID_COLUMNS if c in df.columns]
    return df.drop(columns=drop)


def label_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("Is")]


def auxiliary_labels(df: pd.DataFrame) -> list[str]:
    return [c for c in label_columns(df) if c != TARGET_COLUMN]


def prepare_xy(
    df: pd.DataFrame,
    text_column: str = TEXT_COLUMN,
    label_column: str = TARGET_COLUMN,
) -> tuple[pd.Series, pd.Series]:
    work = df[[text_column, label_column]].copy()
    work[text_column] = work[text_column].astype(str).str.strip()
    work = work[work[text_column].str.len() > 0]
    work = work.drop_duplicates(subset=[text_column])

    y = work[label_column].map(_to_binary).astype(int)
    x = work[text_column]
    return x, y


def _to_binary(value) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().upper()
    return int(text in {"TRUE", "1", "YES"})
