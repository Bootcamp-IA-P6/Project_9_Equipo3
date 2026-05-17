import os
import re
from pathlib import Path
from typing import Any

import nltk

_NLTK_DIR = Path(__file__).resolve().parents[2] / "data" / "nltk_data"
_NLTK_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NLTK_DATA", str(_NLTK_DIR))
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HTML_PATTERN = re.compile(r"<[^>]+>")
_MENTION_PATTERN = re.compile(r"@\w+")
_NON_ALPHA_PATTERN = re.compile(r"[^a-z0-9\s']")


def _ensure_nltk_data() -> None:
    for resource in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        try:
            if resource.startswith("punkt"):
                nltk.data.find(f"tokenizers/{resource}")
            else:
                nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        remove_html: bool = True,
        remove_stopwords: bool = True,
        use_lemmatization: bool = True,
    ):
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_html = remove_html
        self.remove_stopwords = remove_stopwords
        self.use_lemmatization = use_lemmatization
        self._stop_words: set[str] = set()
        self._lemmatizer: WordNetLemmatizer | None = None

    def fit(self, X, y=None):
        _ensure_nltk_data()
        if self.remove_stopwords:
            self._stop_words = set(stopwords.words("english"))
        if self.use_lemmatization:
            self._lemmatizer = WordNetLemmatizer()
        return self

    def transform(self, X):
        return [self._clean_text(text) for text in X]

    def _clean_text(self, text: str) -> str:
        value = str(text)
        if self.lowercase:
            value = value.lower()
        if self.remove_html:
            value = _HTML_PATTERN.sub(" ", value)
        if self.remove_urls:
            value = _URL_PATTERN.sub(" ", value)
        if self.remove_mentions:
            value = _MENTION_PATTERN.sub(" ", value)
        value = _NON_ALPHA_PATTERN.sub(" ", value)
        value = re.sub(r"\s+", " ", value).strip()

        tokens = word_tokenize(value)
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self._stop_words and len(t) > 1]
        if self.use_lemmatization and self._lemmatizer is not None:
            tokens = [self._lemmatizer.lemmatize(t) for t in tokens]

        return " ".join(tokens)


def build_preprocessor(config: dict[str, Any]) -> TextPreprocessor:
    return TextPreprocessor(**config.get("preprocessing", {}))
