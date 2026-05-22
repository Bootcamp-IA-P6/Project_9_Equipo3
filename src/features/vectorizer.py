"""
src/features/vectorizer.py

Vectorizador configurable desde YAML.
Traducción directa del notebook 03 a código de producción.

Decisión del proyecto: TF-IDF con ngram=(1,2) y max_features=5000.
Justificación:
    - Bigramas capturan contexto: 'black thug' es distinto a 'black' solo
    - max_features=5000 equilibra vocabulario vs overfitting (800 muestras train)
    - sublinear_tf=True evita que repetir una palabra infle artificialmente su peso

Uso:
    vec = Vectorizer()
    X_train_vec = vec.fit_transform(X_train_text)
    X_test_vec  = vec.transform(X_test_text)
"""

import yaml
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Vectorizer:
    """
    Wrapper sobre TfidfVectorizer / CountVectorizer.
    Parámetros controlados por configs/features.yaml.

    Regla crítica: fit() SOLO sobre train, transform() sobre train y test.
    Si se hace fit sobre todo el dataset antes del split → data leakage.
    """

    def __init__(self, config_path: str = "configs/features.yaml", method: str = None):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["vectorization"]

        self.method = method or cfg.get("method", "tfidf")
        c = cfg[self.method]

        if self.method == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features  = c["max_features"],
                ngram_range   = tuple(c["ngram_range"]),
                sublinear_tf  = c.get("sublinear_tf", True),
                min_df        = c.get("min_df", 3),
                analyzer      = "word",
                strip_accents = "unicode",
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features  = c["max_features"],
                ngram_range   = tuple(c["ngram_range"]),
                min_df        = c.get("min_df", 3),
                analyzer      = "word",
                strip_accents = "unicode",
            )

        logger.info(f"Vectorizer: {self.method} | max_features={c['max_features']} | ngram={c['ngram_range']}")

    def fit_transform(self, X_train):
        """Ajusta el vocabulario y transforma el train set."""
        logger.info("Vectorizando train set...")
        matrix = self.vectorizer.fit_transform(X_train)
        logger.info(f"  Shape: {matrix.shape} | Sparsidad: {1 - matrix.nnz/(matrix.shape[0]*matrix.shape[1]):.1%}")
        return matrix

    def transform(self, X):
        """Transforma sin ajustar (para test/producción)."""
        return self.vectorizer.transform(X)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()

    @property
    def vocabulary_size(self) -> int:
        return len(self.vectorizer.vocabulary_)