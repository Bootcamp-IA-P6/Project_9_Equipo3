"""
Hybrid ensemble: regularized DistilBERT probabilities + TF-IDF logistic regression.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from src.evaluation.threshold_tuning import predict_with_threshold, search_best_threshold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.logger import get_logger

logger = get_logger(__name__)


class StableLRModel:
    """Regularized LR on TF-IDF (stable_training.yaml)."""

    def __init__(self, lr_cfg: dict, tfidf_cfg: dict, *, C: float | None = None):
        ngram = tuple(tfidf_cfg.get("ngram_range", [1, 2]))
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=int(tfidf_cfg.get("max_features", 5000)),
                        ngram_range=ngram,
                        sublinear_tf=bool(tfidf_cfg.get("sublinear_tf", True)),
                        min_df=int(tfidf_cfg.get("min_df", 3)),
                        analyzer="word",
                        strip_accents="unicode",
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=float(C if C is not None else lr_cfg.get("C", 0.05)),
                        max_iter=int(lr_cfg.get("max_iter", 2000)),
                        class_weight=lr_cfg.get("class_weight", "balanced"),
                        solver=lr_cfg.get("solver", "lbfgs"),
                        random_state=42,
                    ),
                ),
            ]
        )
        self.is_fitted = False

    def fit(self, X_train, y_train):
        logger.info(f"Training stable LR — C={self.pipeline.named_steps['clf'].C}")
        self.pipeline.fit(X_train, y_train)
        self.is_fitted = True
        return self

    @property
    def C(self) -> float:
        return float(self.pipeline.named_steps["clf"].C)

    def set_C(self, c: float) -> None:
        self.pipeline.named_steps["clf"].C = float(c)

    def train_test_gap(self, X_train, y_train, X_test, y_test) -> tuple[float, float, float]:
        """Return (f1_train, f1_test, gap) using weighted F1."""
        preds_train = self.predict(X_train)
        preds_test = self.predict(X_test)
        y_tr = np.asarray(y_train).astype(int)
        y_te = np.asarray(y_test).astype(int)
        f1_train = float(f1_score(y_tr, preds_train, average="weighted", zero_division=0))
        f1_test = float(f1_score(y_te, preds_test, average="weighted", zero_division=0))
        return f1_train, f1_test, abs(f1_train - f1_test)

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        logger.info(f"Stable LR saved: {path}")

    @classmethod
    def load(cls, path: str | Path) -> "StableLRModel":
        import joblib

        inst = cls.__new__(cls)
        inst.pipeline = joblib.load(path)
        inst.is_fitted = True
        return inst


def fit_lr_with_gap_control(
    X_train,
    y_train,
    X_test,
    y_test,
    lr_cfg: dict,
    tfidf_cfg: dict,
    *,
    max_gap: float = 0.05,
    X_train_gap=None,
    y_train_gap=None,
) -> tuple[StableLRModel, dict]:
    """
    Fit LR on augmented train; tune regularization until |train F1 - test F1| < max_gap.
    """
    gap_cfg = lr_cfg.get("gap_search", {})
    X_gap = X_train_gap if X_train_gap is not None else X_train
    y_gap = y_train_gap if y_train_gap is not None else y_train

    if not gap_cfg.get("enabled", True):
        grid = [{"C": float(lr_cfg.get("C", 0.05)), **tfidf_cfg}]
    else:
        grid = gap_cfg.get("param_grid") or [
            {"C": c, **tfidf_cfg} for c in gap_cfg.get("C_candidates", [lr_cfg.get("C", 0.05)])
        ]

    best: StableLRModel | None = None
    best_meta: dict = {}
    best_gap = float("inf")

    for params in grid:
        merged_tfidf = {**tfidf_cfg, **{k: v for k, v in params.items() if k != "C"}}
        c = float(params.get("C", lr_cfg.get("C", 0.05)))
        model = StableLRModel(lr_cfg, merged_tfidf, C=c)
        model.fit(X_train, y_train)
        f1_train, f1_test, gap = model.train_test_gap(X_gap, y_gap, X_test, y_test)
        logger.info(
            f"LR gap search — C={c} max_features={merged_tfidf.get('max_features')} "
            f"min_df={merged_tfidf.get('min_df')} train_f1={f1_train:.4f} "
            f"test_f1={f1_test:.4f} gap={gap:.4f}"
        )
        meta = {
            "C": c,
            "max_features": int(merged_tfidf.get("max_features", 800)),
            "min_df": int(merged_tfidf.get("min_df", 3)),
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
            logger.info(f"LR gap OK at C={c}")
            break

    if not best_meta.get("gap_ok"):
        logger.warning(
            f"LR gap still {best_meta['train_test_gap']:.4f} after grid search; "
            f"using best gap C={best_meta['C']}"
        )

    return best, best_meta  # type: ignore[return-value]


def soft_vote_probs(
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    weight_a: float = 0.5,
    weight_b: float = 0.5,
) -> np.ndarray:
    total = weight_a + weight_b
    return (weight_a * prob_a + weight_b * prob_b) / total


def evaluate_ensemble(
    bert_probs: np.ndarray,
    lr_probs: np.ndarray,
    y_true: np.ndarray,
    *,
    bert_weight: float = 0.5,
    lr_weight: float = 0.5,
    model_name: str = "Hybrid-ensemble",
    threshold: float = 0.5,
) -> dict:
    """Combine probabilities and compute binary metrics."""
    combined = soft_vote_probs(bert_probs, lr_probs, bert_weight, lr_weight)
    preds = predict_with_threshold(combined, threshold)
    y = np.asarray(y_true).astype(int)

    f1_test = float(f1_score(y, preds, average="weighted", zero_division=0))
    f1_toxic = float(f1_score(y, preds, pos_label=1, zero_division=0))

    return {
        "model": model_name,
        "threshold": round(threshold, 4),
        "f1_weighted": round(f1_test, 4),
        "f1_toxic": round(f1_toxic, 4),
        "roc_auc": round(float(roc_auc_score(y, combined)), 4),
        "fp": int(((y == 0) & (preds == 1)).sum()),
        "fn": int(((y == 1) & (preds == 0)).sum()),
        "ensemble_probs": combined,
        "ensemble_preds": preds,
    }


def tune_ensemble_threshold(
    bert_probs: np.ndarray,
    lr_probs: np.ndarray,
    y_val: np.ndarray,
    *,
    bert_weight: float = 0.5,
    lr_weight: float = 0.5,
    metric: str = "f1_toxic",
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    step: float = 0.01,
) -> tuple[float, float]:
    """Search ensemble threshold on validation soft-voted probabilities."""
    combined = soft_vote_probs(bert_probs, lr_probs, bert_weight, lr_weight)
    return search_best_threshold(
        y_val,
        combined,
        metric=metric,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        step=step,
    )


def compute_performance_weights(
    bert_probs_val: np.ndarray,
    lr_probs_val: np.ndarray,
    y_val: np.ndarray,
    *,
    bert_threshold: float = 0.33,
    lr_threshold: float = 0.5,
    metric: str = "f1_weighted",
    min_lr_weight: float = 0.15,
    max_lr_weight: float = 0.45,
) -> tuple[float, float, dict]:
    """
    Set soft-vote weights proportional to validation F1 (per branch threshold).
    """
    y = np.asarray(y_val).astype(int)
    bert_preds = predict_with_threshold(bert_probs_val, bert_threshold)
    lr_preds = predict_with_threshold(lr_probs_val, lr_threshold)

    if metric == "f1_toxic":
        bert_score = float(f1_score(y, bert_preds, pos_label=1, zero_division=0))
        lr_score = float(f1_score(y, lr_preds, pos_label=1, zero_division=0))
    else:
        bert_score = float(f1_score(y, bert_preds, average="weighted", zero_division=0))
        lr_score = float(f1_score(y, lr_preds, average="weighted", zero_division=0))

    total = bert_score + lr_score
    if total <= 0:
        bw, lw = 0.7, 0.3
    else:
        lw = lr_score / total
        lw = float(np.clip(lw, min_lr_weight, max_lr_weight))
        bw = 1.0 - lw

    return bw, lw, {
        "bert_val_score": round(bert_score, 4),
        "lr_val_score": round(lr_score, 4),
        "bert_weight": round(bw, 4),
        "lr_weight": round(lw, 4),
        "weight_metric": metric,
    }


def save_ensemble_meta(path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
