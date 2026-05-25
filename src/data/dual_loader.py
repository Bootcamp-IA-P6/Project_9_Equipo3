"""
Dual-track data: raw ``Text`` + preprocessed ``clean_text`` (+ optional stats merge).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loader import load_raw_data
from src.features.metadata_features import extract_metadata_features
from src.features.text_preprocessor import TextPreprocessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _resolve_processed_paths(
    processed_preprocessed: str | Path,
    processed_stats: str | Path,
    project_root: Path | None,
) -> tuple[Path, Path]:
    root = project_root or Path.cwd()
    pre = Path(processed_preprocessed)
    stats = Path(processed_stats)
    if not pre.is_absolute():
        pre = root / pre
    if not stats.is_absolute():
        stats = root / stats
    return pre, stats


def load_dual_track_data(
    raw_path: str | Path,
    *,
    processed_preprocessed: str | Path = "data/processed/v2/comments_preprocessed.csv",
    processed_stats: str | Path = "data/processed/v2/comments_with_stats.csv",
    target: str = "IsToxic",
    text_column: str = "Text",
    id_column: str = "CommentId",
    features_config: str | Path = "configs/features.yaml",
    write_preprocessed_if_missing: bool = True,
    project_root: Path | None = None,
) -> pd.DataFrame:
    """
    Load raw CSV and attach ``clean_text`` + metadata features.

    Priority for ``clean_text``:
    1. ``comments_preprocessed.csv`` if it exists
    2. Merge from stats file if it contains ``clean_text``
    3. Run ``TextPreprocessor`` on ``Text`` (and optionally cache to preprocessed path)
    """
    raw_path = Path(raw_path)
    root = project_root or raw_path.resolve().parent.parent.parent
    pre_path, stats_path = _resolve_processed_paths(
        processed_preprocessed, processed_stats, root
    )
    feat_path = Path(features_config)
    if not feat_path.is_absolute():
        feat_path = root / feat_path

    df = load_raw_data(raw_path)
    if id_column not in df.columns:
        df[id_column] = range(len(df))

    clean_text: pd.Series | None = None

    if pre_path.exists():
        logger.info(f"Loading preprocessed text: {pre_path}")
        pre = pd.read_csv(pre_path)
        if "clean_text" not in pre.columns:
            raise ValueError(f"{pre_path} missing clean_text column")
        merge_cols = [id_column, "clean_text"]
        if id_column in pre.columns:
            df = df.merge(pre[merge_cols], on=id_column, how="left", suffixes=("", "_pre"))
        else:
            df = df.merge(
                pre[[text_column, "clean_text"]].drop_duplicates(text_column),
                on=text_column,
                how="left",
            )

    if stats_path.exists():
        logger.info(f"Merging stats: {stats_path}")
        stats = pd.read_csv(stats_path)
        key = id_column if id_column in stats.columns and id_column in df.columns else text_column
        stat_cols = [c for c in ("char_length", "word_count", "n_labels", "clean_text") if c in stats.columns]
        if stat_cols:
            df = df.merge(stats[[key] + stat_cols], on=key, how="left", suffixes=("", "_stats"))
        meta = extract_metadata_features(df, text_column=text_column)
    else:
        logger.warning(f"Stats file not found: {stats_path} — computing metadata from Text")
        meta = extract_metadata_features(df, text_column=text_column)

    for col in meta.columns:
        df[col] = meta[col].values

    if "clean_text" not in df.columns or df["clean_text"].isna().all():
        logger.info("Generating clean_text via TextPreprocessor")
        preprocessor = TextPreprocessor(config_path=str(feat_path))
        df["clean_text"] = preprocessor.transform(df[text_column])
        df["clean_text"] = df["clean_text"].where(
            df["clean_text"].astype(str).str.strip() != "",
            df[text_column],
        )
        if write_preprocessed_if_missing:
            pre_path.parent.mkdir(parents=True, exist_ok=True)
            export_cols = [id_column, text_column, "clean_text", target]
            export_cols = [c for c in export_cols if c in df.columns]
            df[export_cols].to_csv(pre_path, index=False)
            logger.info(f"Cached preprocessed CSV: {pre_path}")

    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    empty = (df["clean_text"].str.strip() == "").sum()
    if empty:
        logger.warning(f"{empty} empty clean_text rows — falling back to raw Text")
        mask = df["clean_text"].str.strip() == ""
        df.loc[mask, "clean_text"] = df.loc[mask, text_column]

    logger.info(
        f"Dual-track ready — rows={len(df)} | clean_text non-empty="
        f"{(df['clean_text'].str.strip() != '').sum()}"
    )
    return df
