"""
Logistic regression on TF-IDF(clean_text) + scaled metadata features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from src.features.metadata_features import DEFAULT_METADATA_COLUMNS
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MetadataLRModel:
    """TF-IDF on clean text + numeric metadata → logistic regression."""

    def __init__(
        self,
        lr_cfg: dict,
        tfidf_cfg: dict,
        *,
        metadata_columns: list[str] | None = None,
        C: float | None = None,
    ):
        self.metadata_columns = metadata_columns or list(DEFAULT_METADATA_COLUMNS)
        ngram = tuple(tfidf_cfg.get("ngram_range", [1, 2]))
        self.tfidf = TfidfVectorizer(
            max_features=int(tfidf_cfg.get("max_features", 5000)),
            ngram_range=ngram,
            sublinear_tf=bool(tfidf_cfg.get("sublinear_tf", True)),
            min_df=int(tfidf_cfg.get("min_df", 3)),
            analyzer="word",
            strip_accents="unicode",
        )
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(
            C=float(C if C is not None else lr_cfg.get("C", 0.05)),
            max_iter=int(lr_cfg.get("max_iter", 2000)),
            class_weight=lr_cfg.get("class_weight", "balanced"),
            solver=lr_cfg.get("solver", "lbfgs"),
            random_state=42,
        )
        self.is_fitted = False

    @property
    def C(self) -> float:
        return float(self.clf.C)

    def _meta_matrix(self, meta: pd.DataFrame) -> np.ndarray:
        cols = [c for c in self.metadata_columns if c in meta.columns]
        return meta[cols].astype(float).values

    def _features(self, X_clean: pd.Series, meta: pd.DataFrame, *, fit: bool) -> np.ndarray:
        if fit:
            X_t = self.tfidf.fit_transform(X_clean.astype(str))
            X_m = self.scaler.fit_transform(self._meta_matrix(meta))
        else:
            X_t = self.tfidf.transform(X_clean.astype(str))
            X_m = self.scaler.transform(self._meta_matrix(meta))
        return hstack([X_t, X_m])

    def fit(
        self,
        X_clean: pd.Series,
        meta: pd.DataFrame,
        y,
    ) -> "MetadataLRModel":
        X = self._features(X_clean, meta, fit=True)
        self.clf.fit(X, y)
        self.is_fitted = True
        logger.info(
            f"Metadata LR trained — C={self.C} | "
            f"tfidf_dim={len(self.tfidf.vocabulary_)} | meta_dim={len(self.metadata_columns)}"
        )
        return self

    def predict_proba(self, X_clean: pd.Series, meta: pd.DataFrame) -> np.ndarray:
        X = self._features(X_clean, meta, fit=False)
        return self.clf.predict_proba(X)

    def predict(self, X_clean: pd.Series, meta: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X_clean, meta).argmax(axis=1)

    def train_test_gap(
        self,
        X_train_clean,
        meta_train,
        y_train,
        X_test_clean,
        meta_test,
        y_test,
    ) -> tuple[float, float, float]:
        preds_train = self.predict(X_train_clean, meta_train)
        preds_test = self.predict(X_test_clean, meta_test)
        y_tr = np.asarray(y_train).astype(int)
        y_te = np.asarray(y_test).astype(int)
        f1_train = float(f1_score(y_tr, preds_train, average="weighted", zero_division=0))
        f1_test = float(f1_score(y_te, preds_test, average="weighted", zero_division=0))
        return f1_train, f1_test, abs(f1_train - f1_test)

    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "tfidf": self.tfidf,
                "scaler": self.scaler,
                "clf": self.clf,
                "metadata_columns": self.metadata_columns,
            },
            path,
        )
        logger.info(f"Metadata LR saved: {path}")

    @classmethod
    def load(cls, path: str | Path) -> "MetadataLRModel":
        import joblib

        blob = joblib.load(path)
        inst = cls.__new__(cls)
        inst.tfidf = blob["tfidf"]
        inst.scaler = blob["scaler"]
        inst.clf = blob["clf"]
        inst.metadata_columns = blob["metadata_columns"]
        inst.is_fitted = True
        return inst


def fit_metadata_lr_with_gap_control(
    X_train_clean,
    meta_train,
    y_train,
    X_test_clean,
    meta_test,
    y_test,
    lr_cfg: dict,
    tfidf_cfg: dict,
    *,
    max_gap: float = 0.05,
    X_train_gap_clean=None,
    meta_train_gap=None,
    y_train_gap=None,
) -> tuple[MetadataLRModel, dict]:
    gap_cfg = lr_cfg.get("gap_search", {})
    X_gap = X_train_gap_clean if X_train_gap_clean is not None else X_train_clean
    meta_gap = meta_train_gap if meta_train_gap is not None else meta_train
    y_gap = y_train_gap if y_train_gap is not None else y_train

    grid = (
        gap_cfg.get("param_grid")
        if gap_cfg.get("enabled", True)
        else [{"C": float(lr_cfg.get("C", 0.05)), **tfidf_cfg}]
    )

    best: MetadataLRModel | None = None
    best_meta: dict = {}
    best_gap = float("inf")

    for params in grid:
        merged = {**tfidf_cfg, **{k: v for k, v in params.items() if k != "C"}}
        c = float(params.get("C", lr_cfg.get("C", 0.05)))
        model = MetadataLRModel(lr_cfg, merged, C=c)
        model.fit(X_train_clean, meta_train, y_train)
        f1_train, f1_test, gap = model.train_test_gap(
            X_gap, meta_gap, y_gap, X_test_clean, meta_test, y_test
        )
        logger.info(
            f"Metadata LR gap — C={c} max_features={merged.get('max_features')} "
            f"train_f1={f1_train:.4f} test_f1={f1_test:.4f} gap={gap:.4f}"
        )
        meta = {
            "C": c,
            "max_features": int(merged.get("max_features", 5000)),
            "min_df": int(merged.get("min_df", 3)),
            "f1_train": round(f1_train, 4),
            "f1_test": round(f1_test, 4),
            "train_test_gap": round(gap, 4),
            "train_test_gap_pp": round(gap * 100, 2),
            "gap_ok": gap < max_gap,
        }
        if gap < best_gap:
            best, best_meta = model, meta
            best_gap = gap
        if gap < max_gap:
            break

    return best, best_meta  # type: ignore[return-value]
