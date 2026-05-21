"""Phase 2 text cleaning and morphological processing (AGENTS.md)."""

from __future__ import annotations

import os
import re
import string
from functools import lru_cache
from pathlib import Path

import nltk

# Store NLTK data inside the repo (writable, reproducible across machines)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_NLTK_DATA_DIR = _PROJECT_ROOT / "nltk_data"
_NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
if str(_NLTK_DATA_DIR) not in nltk.data.path:
    nltk.data.path.insert(0, str(_NLTK_DATA_DIR))
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Emoji and symbols (supplement regex cleaning)
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_MENTION_PATTERN = re.compile(r"@\w+")
_NON_ALPHA_PATTERN = re.compile(r"[^a-z\s]+")


def ensure_nltk_data() -> None:
    """Download NLTK resources required for tokenization and lemmatization."""
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


@lru_cache(maxsize=1)
def _english_stopwords() -> set[str]:
    ensure_nltk_data()
    return set(stopwords.words("english"))


@lru_cache(maxsize=1)
def _lemmatizer() -> WordNetLemmatizer:
    ensure_nltk_data()
    return WordNetLemmatizer()


def clean_text_regex(text: str) -> str:
    """
    Regex cleaning: lowercase, URLs, mentions, emojis, punctuation, whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_PATTERN.sub("", text)
    text = _MENTION_PATTERN.sub("", text)
    text = _EMOJI_PATTERN.sub("", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = _NON_ALPHA_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text: str, *, remove_stopwords: bool = True) -> list[str]:
    """Tokenize and lemmatize; optionally drop English stopwords."""
    if not text:
        return []
    ensure_nltk_data()
    tokens = word_tokenize(text)
    stops = _english_stopwords() if remove_stopwords else set()
    lemmatizer = _lemmatizer()
    return [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t.isalpha() and (t not in stops)
    ]


def process_text(
    text: str,
    *,
    remove_stopwords: bool = True,
) -> tuple[str, str]:
    """
    Full Phase 2 pipeline for one comment.

    Returns:
        (clean_text_regex output, space-joined lemmatized tokens)
    """
    cleaned = clean_text_regex(text)
    tokens = tokenize_and_lemmatize(cleaned, remove_stopwords=remove_stopwords)
    return cleaned, " ".join(tokens)
