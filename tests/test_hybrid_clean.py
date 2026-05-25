"""Tests for dual-input hybrid helpers."""

import numpy as np
import pandas as pd
import pytest

from src.features.metadata_features import extract_metadata_features
from src.models.hybrid_ensemble import compute_performance_weights


def test_extract_metadata_features():
    df = pd.DataFrame({"Text": ["Hello!! WORLD?", "ok"], "IsToxic": [1, 0]})
    meta = extract_metadata_features(df)
    assert list(meta.columns) == [
        "char_length",
        "word_count",
        "exclamation_ratio",
        "question_ratio",
        "caps_ratio",
    ]
    assert meta.loc[0, "exclamation_ratio"] > 0


def test_compute_performance_weights_bounds():
    y = np.array([0, 1, 1, 0])
    b = np.array([0.2, 0.8, 0.7, 0.3])
    l = np.array([0.3, 0.6, 0.75, 0.25])
    bw, lw, info = compute_performance_weights(
        b, l, y, min_lr_weight=0.15, max_lr_weight=0.45
    )
    assert pytest.approx(bw + lw, rel=1e-5) == 1.0
    assert 0.15 <= lw <= 0.45
    assert "bert_val_score" in info


def test_load_dual_track_data_generates_clean_text(tmp_path, monkeypatch):
    pytest.importorskip("spacy")
    raw = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "CommentId": [1, 2],
            "Text": ["Visit http://x.com now!!!", "nice video"],
            "IsToxic": [1, 0],
        }
    ).to_csv(raw, index=False)

    from src.data import dual_loader as dl

    monkeypatch.setattr(
        dl.TextPreprocessor,
        "__init__",
        lambda self, config_path=None: None,
    )
    monkeypatch.setattr(
        dl.TextPreprocessor,
        "transform",
        lambda self, s: s.str.lower(),
    )

    out = dl.load_dual_track_data(
        raw,
        processed_preprocessed=tmp_path / "pre.csv",
        processed_stats=tmp_path / "missing_stats.csv",
        write_preprocessed_if_missing=True,
        project_root=tmp_path,
    )
    assert "clean_text" in out.columns
    assert (out["char_length"] > 0).all()
