"""Tests del pipeline de preprocesamiento de texto."""

import re

import pytest

from src.features.text_preprocessor import TextPreprocessor


@pytest.fixture(scope="module")
def preprocessor(features_config: str) -> TextPreprocessor:
    return TextPreprocessor(config_path=features_config)


def test_empty_text_returns_empty_string(preprocessor: TextPreprocessor):
    assert preprocessor.transform("") == ""
    assert preprocessor.transform("   ") == ""


def test_url_text_removes_urls(preprocessor: TextPreprocessor):
    raw = "Visit https://example.com/path and www.test.org now"
    clean = preprocessor.transform(raw)

    assert "http" not in clean
    assert "www." not in clean
    assert "example.com" not in clean
    assert re.search(r"https?://", clean) is None


def test_normal_text_lowercase_and_lemmatized(preprocessor: TextPreprocessor):
    raw = "The runners are running quickly"
    clean = preprocessor.transform(raw)

    assert isinstance(clean, str)
    assert clean == clean.lower()
    assert clean != ""
    assert "run" in clean.split()
