"""Toxic-only back-translation augmentation for Plan C v2 (optional)."""

from __future__ import annotations

import logging
import time
from typing import Iterable

import pandas as pd

from src.evaluation.phase1_audit import BINARY_TARGET

logger = logging.getLogger(__name__)

# English -> Spanish -> English (lightweight, no Marian weights)
_DEFAULT_HOPS = ("es",)


def augment_toxic_backtranslation(
    train_df: pd.DataFrame,
    *,
    text_col: str = "Text",
    target_col: str = BINARY_TARGET,
    intermediate_langs: Iterable[str] = _DEFAULT_HOPS,
    max_samples: int | None = None,
    sleep_seconds: float = 0.05,
) -> pd.DataFrame:
    """
    Duplicate toxic training rows with back-translated text.

    Only rows with ``target_col == 1`` are augmented. Original rows are kept.
    """
    try:
        from deep_translator import GoogleTranslator
    except ImportError as exc:
        raise ImportError(
            "Install deep-translator for Plan C back-translation: uv add deep-translator"
        ) from exc

    toxic = train_df[train_df[target_col] == 1].copy()
    if toxic.empty:
        logger.warning("No toxic rows to augment.")
        return train_df.copy()

    if max_samples is not None:
        toxic = toxic.head(max_samples)

    langs = list(intermediate_langs)
    augmented_rows: list[pd.Series] = []

    for _, row in toxic.iterrows():
        text = str(row[text_col]).strip()
        if not text:
            continue
        try:
            current = text
            for lang in langs:
                current = GoogleTranslator(source="auto", target=lang).translate(current)
                time.sleep(sleep_seconds)
            restored = GoogleTranslator(source="auto", target="en").translate(current)
            time.sleep(sleep_seconds)
            if not restored or restored == text:
                continue
            new_row = row.copy()
            new_row[text_col] = restored
            augmented_rows.append(new_row)
        except Exception as err:
            logger.debug("Back-translation skipped for row: %s", err)
            continue

    if not augmented_rows:
        logger.warning("Back-translation produced no new rows; returning original train set.")
        return train_df.copy()

    out = pd.concat([train_df, pd.DataFrame(augmented_rows)], ignore_index=True)
    logger.info(
        "Back-translation: %s -> %s training rows (+%s synthetic toxic)",
        len(train_df),
        len(out),
        len(augmented_rows),
    )
    return out
