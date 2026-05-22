"""
src/features/text_preprocessor.py

Pipeline de preprocesamiento NLP.
Traducción directa del notebook 02 a código de producción.

Pasos:
    1. Lowercase
    2. Regex: URLs, @menciones, \\xa0, apostrofes, números
    3. spaCy: lematización (en_core_web_sm)
    4. NLTK: filtrado stopwords english + custom

Uso:
    preprocessor = TextPreprocessor()
    clean_series = preprocessor.transform(df["Text"])
    clean_text   = preprocessor.transform("texto crudo aqui")
"""

import re
import yaml
import nltk
import spacy
import pandas as pd
from pathlib import Path
from nltk.corpus import stopwords
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Descargar recursos NLTK si no existen
for resource in ["stopwords", "punkt"]:
    nltk.download(resource, quiet=True)


class TextPreprocessor:
    """
    Pipeline NLP para hate speech detection.
    Lee su configuración de configs/features.yaml.
    """

    # Stopwords custom: palabras frecuentes sin valor discriminante
    # en el dominio YouTube. No son stopwords generales.
    CUSTOM_STOPWORDS = {
        "youtube", "video", "watch", "like", "comment",
        "channel", "click", "subscribe", "link",
    }

    def __init__(self, config_path: str = "configs/features.yaml"):
        # Cargar config
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["preprocessing"]
        self.cfg = cfg

        # Stopwords: NLTK + custom
        self.stop_words = set(stopwords.words("english")) | self.CUSTOM_STOPWORDS
        self.min_len = cfg.get("min_token_length", 2)

        # Cargar modelo spaCy
        # disable=["parser","ner"] → solo usamos el lemmatizer, más rápido
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        logger.info(f"TextPreprocessor iniciado — spaCy {self.nlp.meta['version']}")

    # ── Pasos individuales ────────────────────────────────────────────────────

    def _lowercase(self, text: str) -> str:
        """Paso 1: minúsculas. 'BLACK' y 'black' son la misma feature."""
        return str(text).lower()

    def _clean_regex(self, text: str) -> str:
        """
        Paso 2: elimina ruido estructural con regex.
        Orden importante: primero lo más específico, luego lo general.
        """
        text = re.sub(r"http\S+|www\.\S+", "", text)   # URLs
        text = re.sub(r"@\w+", "", text)                # @menciones
        text = re.sub(r"[\n\t\r]", " ", text)           # saltos de línea
        text = re.sub(r"[^\x00-\x7F]+", " ", text)      # \xa0, emojis
        text = re.sub(r"'", "", text)                   # apóstrofes
        text = re.sub(r"\b\d+\b", "", text)             # números solos
        text = re.sub(r"\s+", " ", text)                # espacios múltiples
        return text.strip()

    def _lemmatize(self, text: str) -> str:
        """
        Paso 3+4: lematización con spaCy + filtrado de stopwords con NLTK.

        Por qué spaCy para lematizar:
            Entiende gramática: 'running'→'run', 'cops'→'cop'
            Un stemmer de NLTK simplemente corta: 'running'→'runn'

        Por qué NLTK para stopwords:
            Lista curada de 179 palabras funcionales.
            Más fácil de personalizar que la lista interna de spaCy.

        DECISIÓN del EDA: NO eliminar 'black','white','police','cop'
            → Aparecen en ambas clases con contexto distinto.
              El modelo necesita verlas para aprender por bigrams.
        """
        doc = self.nlp(text)
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_punct
            and not token.is_space
            and len(token.text) >= self.min_len
            and token.lemma_ not in self.stop_words
        ]
        return " ".join(tokens)

    def _transform_one(self, text: str) -> str:
        text = self._lowercase(text)
        text = self._clean_regex(text)
        text = self._lemmatize(text)
        return text

    # ── Interfaz pública ──────────────────────────────────────────────────────

    def transform(self, data) -> str | pd.Series:
        """
        Preprocesa un texto o una Serie completa.

        Args:
            data: str o pd.Series con textos crudos.

        Returns:
            str o pd.Series con textos limpios y lematizados.
        """
        if isinstance(data, pd.Series):
            logger.info(f"Preprocesando {len(data)} textos...")
            result = data.apply(self._transform_one)
            empty  = (result == "").sum()
            if empty > 0:
                logger.warning(f"  {empty} textos quedaron vacíos tras limpieza")
            return result
        return self._transform_one(data)