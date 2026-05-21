import sys
from pathlib import Path

# Let the test import "src.*" from the project root
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.features.preprocess import TextPreprocessor


def test_text_preprocessor_lowercase():
    preprocessor = TextPreprocessor(lowercase=True, remove_stopwords=False, use_lemmatization=False)
    preprocessor.fit([""])
    assert preprocessor.transform(["HELLO WORLD"])[0] == "hello world"


def test_text_preprocessor_remove_html():
    preprocessor = TextPreprocessor(remove_html=True, remove_stopwords=False, use_lemmatization=False)
    preprocessor.fit([""])
    assert preprocessor.transform(["<p>Hello</p> <div>World</div>"])[0] == "hello world"


def test_text_preprocessor_remove_urls():
    preprocessor = TextPreprocessor(remove_urls=True, remove_stopwords=False, use_lemmatization=False)
    preprocessor.fit([""])
    assert preprocessor.transform(["Check this https://google.com out"])[0] == "check this out"


def test_text_preprocessor_remove_mentions():
    preprocessor = TextPreprocessor(remove_mentions=True, remove_stopwords=False, use_lemmatization=False)
    preprocessor.fit([""])
    assert preprocessor.transform(["Hello @youtube user"])[0] == "hello user"


def test_text_preprocessor_stopwords_and_lemmatization():
    preprocessor = TextPreprocessor(
        lowercase=True, 
        remove_stopwords=True, 
        use_lemmatization=True
    )
    preprocessor.fit([""])
    # "the" should be dropped as a stopword, and "cats" should shrink to "cat"
    processed = preprocessor.transform(["the cats are running fast"])
    assert "the" not in processed[0]
    assert "cats" not in processed[0]
    assert "cat" in processed[0]
