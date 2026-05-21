"""Unit tests for Phase 2 preprocessing."""

from src.features.phase2_text_pipeline import clean_text_regex, process_text


def test_clean_text_removes_url_and_mention():
    raw = "Hello https://evil.com @user"
    clean = clean_text_regex(raw)
    assert "https" not in clean
    assert "@user" not in clean


def test_process_text_returns_tokens():
    clean, processed = process_text("Thanks for the helpful video")
    assert isinstance(clean, str)
    assert len(processed.split()) >= 1
