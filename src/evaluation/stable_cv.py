"""
Stratified K-fold cross-validation for the stable production pipeline.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.utils.logger import get_logger

logger = get_logger(__name__)


def stratified_kfold_cv(
    X: pd.Series,
    y: pd.Series,
    *,
    n_splits: int,
    random_state: int,
    fit_predict_fn: Callable[[pd.Series, pd.Series, pd.Series, pd.Series], dict[str, float]],
) -> dict[str, Any]:
    """
    Run stratified K-fold CV with a caller-provided fit/eval hook.

    ``fit_predict_fn`` receives (X_tr, y_tr, X_val, y_val) and returns
    per-fold metrics including at least ``f1_weighted``.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics: list[dict[str, float]] = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr = X.iloc[tr_idx].reset_index(drop=True)
        y_tr = y.iloc[tr_idx].reset_index(drop=True)
        X_val = X.iloc[val_idx].reset_index(drop=True)
        y_val = y.iloc[val_idx].reset_index(drop=True)

        m = fit_predict_fn(X_tr, y_tr, X_val, y_val)
        m["fold"] = fold
        fold_metrics.append(m)
        logger.info(
            f"CV fold {fold + 1}/{n_splits} — F1={m['f1_weighted']:.4f} "
            f"gap={m.get('train_val_gap', 0):.4f}"
        )

    f1s = [m["f1_weighted"] for m in fold_metrics]
    gaps = [m.get("train_val_gap", 0.0) for m in fold_metrics]
    rocs = [m.get("roc_auc", np.nan) for m in fold_metrics]

    summary = {
        "n_splits": n_splits,
        "f1_mean": round(float(np.mean(f1s)), 4),
        "f1_std": round(float(np.std(f1s)), 4),
        "f1_min": round(float(np.min(f1s)), 4),
        "f1_max": round(float(np.max(f1s)), 4),
        "gap_mean": round(float(np.mean(gaps)), 4),
        "gap_std": round(float(np.std(gaps)), 4),
        "gap_max": round(float(np.max(gaps)), 4),
        "roc_auc_mean": round(float(np.nanmean(rocs)), 4),
        "folds": fold_metrics,
    }
    stable = summary["f1_std"] < 0.05 and summary["gap_max"] < 0.05
    summary["stable_across_folds"] = stable
    return summary


def evaluate_lr_fold(
    lr_model_factory,
    X_tr: pd.Series,
    y_tr: pd.Series,
    X_val: pd.Series,
    y_val: pd.Series,
    *,
    augment_fn=None,
    cfg: dict | None = None,
    seed: int = 42,
) -> dict[str, float]:
    """Fit LR on (optionally augmented) fold train; score fold val."""
    if augment_fn and cfg is not None:
        X_fit, y_fit = augment_fn(X_tr, y_tr, cfg, seed=seed)
    else:
        X_fit, y_fit = X_tr, y_tr

    model = lr_model_factory()
    model.fit(X_fit, y_fit)

    y_val_arr = y_val.astype(int).values
    preds_val = model.predict(X_val)
    preds_train = model.predict(X_fit)
    probs_val = model.predict_proba(X_val)[:, 1]

    f1_val = float(f1_score(y_val_arr, preds_val, average="weighted", zero_division=0))
    f1_train = float(
        f1_score(y_fit.astype(int), preds_train, average="weighted", zero_division=0)
    )
    gap = abs(f1_train - f1_val)

    return {
        "f1_weighted": f1_val,
        "f1_train": f1_train,
        "train_val_gap": round(gap, 4),
        "train_val_gap_pp": round(gap * 100, 2),
        "roc_auc": round(float(roc_auc_score(y_val_arr, probs_val)), 4),
    }
