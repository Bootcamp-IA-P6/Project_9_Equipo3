"""
Toxic-only back-translation augmentation with cosine deduplication.

Augments only the positive (toxic) class in the training set via EN→ES→EN,
then drops synthetic samples too similar to the original training corpus.
"""

from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.logger import get_logger

logger = get_logger(__name__)


def toxic_back_translation(
    texts: Iterable[str],
    labels: Iterable[int | bool],
    *,
    source_lang: str = "en",
    pivot_lang: str = "es",
    min_words: int = 3,
    max_words: int = 60,
    rate_limit_every: int = 50,
    rate_limit_sleep_sec: float = 1.0,
    seed: int = 42,
) -> tuple[list[str], list[int]]:
    """
    Back-translate toxic samples only (label == 1).

    Returns parallel lists of augmented texts and labels (all toxic).
    """
    try:
        from deep_translator import GoogleTranslator
    except ImportError as e:
        raise ImportError(
            "Install augmentation deps: uv sync --extra train"
        ) from e

    import random

    random.seed(seed)
    to_pivot = GoogleTranslator(source=source_lang, target=pivot_lang)
    to_source = GoogleTranslator(source=pivot_lang, target=source_lang)

    aug_texts: list[str] = []
    aug_labels: list[int] = []

    pairs = [
        (str(t).strip(), int(bool(l)))
        for t, l in zip(texts, labels, strict=False)
        if int(bool(l)) == 1
    ]
    logger.info(f"Back-translation: {len(pairs)} toxic samples")

    for i, (text, label) in enumerate(pairs):
        words = text.split()
        if len(words) < min_words:
            continue
        try:
            short = " ".join(words[:max_words])
            pivot = to_pivot.translate(short)
            back = to_source.translate(pivot)
            if back and back.strip() and back.strip() != short.strip():
                aug_texts.append(back.strip())
                aug_labels.append(label)
        except Exception as exc:
            logger.warning(f"Back-translation failed at index {i}: {exc}")
            continue

        if rate_limit_every > 0 and i > 0 and i % rate_limit_every == 0:
            time.sleep(rate_limit_sleep_sec)

    logger.info(f"Back-translation produced {len(aug_texts)} samples")
    return aug_texts, aug_labels


def back_translate_texts(
    texts: Iterable[str],
    *,
    source_lang: str = "en",
    pivot_lang: str = "de",
    max_words: int = 60,
    rate_limit_every: int = 50,
    rate_limit_sleep_sec: float = 1.0,
    fallback_to_original: bool = True,
) -> list[str]:
    """
    Back-translate every text (EN→pivot→EN) for test-time augmentation.

    On failure, returns the original string when ``fallback_to_original`` is True.
    """
    try:
        from deep_translator import GoogleTranslator
    except ImportError as e:
        raise ImportError(
            "Install augmentation deps: uv sync --extra train"
        ) from e

    to_pivot = GoogleTranslator(source=source_lang, target=pivot_lang)
    to_source = GoogleTranslator(source=pivot_lang, target=source_lang)

    out: list[str] = []
    for i, raw in enumerate(texts):
        text = str(raw).strip()
        if not text:
            out.append(text)
            continue
        words = text.split()
        short = " ".join(words[:max_words])
        try:
            pivot = to_pivot.translate(short)
            back = to_source.translate(pivot)
            out.append(back.strip() if back and back.strip() else text)
        except Exception as exc:
            logger.warning(f"TTA back-translation failed at index {i}: {exc}")
            out.append(text if fallback_to_original else short)

        if rate_limit_every > 0 and i > 0 and i % rate_limit_every == 0:
            time.sleep(rate_limit_sleep_sec)

    return out


def deduplicate_by_cosine(
    synthetic_texts: list[str],
    synthetic_labels: list[int],
    reference_texts: list[str],
    *,
    threshold: float = 0.95,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> tuple[list[str], list[int]]:
    """
    Remove synthetic samples with max cosine similarity > threshold vs reference.
    """
    if not synthetic_texts:
        return [], []

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "Install augmentation deps: uv sync --extra train"
        ) from e

    model = SentenceTransformer(embedding_model)
    ref_emb = model.encode(reference_texts, show_progress_bar=False, convert_to_numpy=True)
    syn_emb = model.encode(synthetic_texts, show_progress_bar=False, convert_to_numpy=True)

    sims = cosine_similarity(syn_emb, ref_emb)
    max_sim = sims.max(axis=1)

    kept_texts: list[str] = []
    kept_labels: list[int] = []
    dropped = 0
    for text, label, sim in zip(synthetic_texts, synthetic_labels, max_sim, strict=False):
        if sim <= threshold:
            kept_texts.append(text)
            kept_labels.append(label)
        else:
            dropped += 1

    logger.info(
        f"Dedup: kept {len(kept_texts)}/{len(synthetic_texts)} "
        f"(dropped {dropped} with cosine > {threshold})"
    )
    return kept_texts, kept_labels


def augment_toxic_train(
    X_train: pd.Series,
    y_train: pd.Series,
    cfg: dict,
    *,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """
    Append toxic-only back-translated samples to training data (with dedup).
    """
    aug_cfg = cfg.get("augmentation", {})
    if not aug_cfg.get("enabled", True):
        return X_train, y_train

    syn_texts, syn_labels = toxic_back_translation(
        X_train.tolist(),
        y_train.tolist(),
        source_lang=aug_cfg.get("source_lang", "en"),
        pivot_lang=aug_cfg.get("pivot_lang", "es"),
        min_words=aug_cfg.get("min_words", 3),
        max_words=aug_cfg.get("max_words", 60),
        rate_limit_every=aug_cfg.get("rate_limit_every", 50),
        rate_limit_sleep_sec=aug_cfg.get("rate_limit_sleep_sec", 1.0),
        seed=seed,
    )

    dedup_cfg = aug_cfg.get("dedup", {})
    if dedup_cfg.get("enabled", True) and syn_texts:
        syn_texts, syn_labels = deduplicate_by_cosine(
            syn_texts,
            syn_labels,
            X_train.tolist(),
            threshold=float(dedup_cfg.get("cosine_threshold", 0.95)),
            embedding_model=dedup_cfg.get(
                "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
            ),
        )

    if not syn_texts:
        return X_train, y_train

    X_aug = pd.concat(
        [X_train, pd.Series(syn_texts, name=X_train.name)],
        ignore_index=True,
    )
    y_aug = pd.concat(
        [y_train, pd.Series(syn_labels, name=y_train.name)],
        ignore_index=True,
    )
    logger.info(f"Train size after augmentation: {len(X_aug)} (+{len(syn_texts)})")
    return X_aug, y_aug
