import json
import os
import sys
import yaml
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Make sure "src.*" imports work no matter where this script is run from
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.load_data import load_dataset, prepare_xy
from src.data.external import load_external_training_data
from src.features.preprocess import build_preprocessor
from src.features.augmentation import augment_dataset
from src.evaluation.metrics import compute_metrics
from src.utils.config import load_config, project_root

# Optuna is noisy by default, so quiet it down to keep our own prints readable
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial, X_train_full, y_train_full, external, config):
    # How TF-IDF turns text into numbers. Wide ranges so the model can actually learn.
    max_features = trial.suggest_int("max_features", 1000, 10000, step=500)
    ngram_choice = trial.suggest_categorical("ngram_range", ["unigram", "bigram"])
    ngram_range = (1, 1) if ngram_choice == "unigram" else (1, 2)
    min_df = trial.suggest_int("min_df", 1, 5)
    max_df = trial.suggest_float("max_df", 0.85, 1.0)
    
    # How much synthetic toxic text to add to the training data
    augment_multiplier = trial.suggest_float("augment_multiplier", 0.0, 0.2, step=0.1)

    # The model is always a soft-voting ensemble of three classifiers.
    # Optuna does not choose whether to use them, only how to tune each one.
    lr_C = trial.suggest_float("lr_C", 0.01, 100.0, log=True)
    nb_alpha = trial.suggest_float("nb_alpha", 0.1, 2.0, log=True)
    rf_n_estimators = trial.suggest_int("rf_n_estimators", 50, 300, step=50)
    rf_max_depth = trial.suggest_int("rf_max_depth", 5, 30)

    # Measure this trial with 3 fold cross validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=config["data"]["random_state"])
    val_f1s = []
    train_f1s = []

    for train_idx, val_idx in skf.split(X_train_full, y_train_full):
        X_tr, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_tr, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

        # The model trains on this YouToxic fold plus every external dataset.
        # The validation fold and the train score below stay pure YouToxic.
        if len(external):
            X_fit = pd.concat([X_tr, external["Text"]], ignore_index=True)
            y_fit = pd.concat([y_tr, external["IsToxic"]], ignore_index=True)
        else:
            X_fit, y_fit = X_tr, y_tr

        # Only the training data gets synthetic rows; the validation fold stays real
        if augment_multiplier > 0:
            X_tr_aug, y_tr_aug = augment_dataset(
                X_fit, y_fit,
                label_to_augment=1,
                multiplier=augment_multiplier,
                random_state=config["data"]["random_state"]
            )
        else:
            X_tr_aug, y_tr_aug = X_fit, y_fit

        # Build a fresh pipeline for this fold
        prep = build_preprocessor(config)
        
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df
        )

        # The three base models of the ensemble, each with this trial's params
        clf_lr = LogisticRegression(
            C=lr_C,
            class_weight="balanced",
            max_iter=1000,
            random_state=config["data"]["random_state"]
        )
        clf_nb = MultinomialNB(alpha=nb_alpha)
        clf_rf = RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            class_weight="balanced",
            random_state=config["data"]["random_state"]
        )

        # Soft voting averages the three probability outputs into one prediction
        classifier = VotingClassifier(
            estimators=[("lr", clf_lr), ("nb", clf_nb), ("rf", clf_rf)],
            voting="soft",
        )

        fold_pipeline = Pipeline([
            ("preprocess", prep),
            ("tfidf", vectorizer),
            ("clf", classifier)
        ])

        fold_pipeline.fit(X_tr_aug, y_tr_aug)

        # Predict on the real training fold, not the augmented one
        y_tr_pred = fold_pipeline.predict(X_tr)
        y_val_pred = fold_pipeline.predict(X_val)

        train_metrics = compute_metrics(y_tr, y_tr_pred)
        val_metrics = compute_metrics(y_val, y_val_pred)

        train_f1s.append(train_metrics["f1"])
        val_f1s.append(val_metrics["f1"])

    mean_train_f1 = np.mean(train_f1s)
    mean_val_f1 = np.mean(val_f1s)
    gap = abs(mean_train_f1 - mean_val_f1)

    # Optimise for validation F1, but keep the train/val gap small so the model
    # passes the 5% overfitting limit. We start penalising at 3% (the holdout gap
    # tends to run a bit higher than this CV estimate) and weight it heavily, so
    # Optuna prefers params that generalise instead of memorising the training set.
    score = mean_val_f1 - max(0.0, gap - 0.03) * 3.0
    return float(score)


def run_optimization(n_trials: int = 50) -> dict:
    config = load_config()
    root = project_root()

    df = load_dataset(config["data"]["raw_path"])
    X, y = prepare_xy(df, config["data"]["text_column"], config["data"]["label_column"])

    # Same split as train.py, so the test set is never seen during tuning
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y
    )

    # Load the external datasets once and reuse them across every trial and fold
    external = load_external_training_data(config)
    if len(external):
        print(f"External training data: {len(external)} rows from outside datasets.")

    print(f"Running Optuna study with {n_trials} trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train_full, y_train_full, external, config), n_trials=n_trials)

    print("\n--- Optuna Optimization Complete ---")
    print(f"Best Trial Score: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Write the winning params where train.py will look for them
    best_params = study.best_params
    best_params_path = root / "configs" / "best_params.yaml"
    best_params_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(best_params_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(best_params, f, default_flow_style=False)
        
    print(f"\nSaved best parameters to {best_params_path}")
    return best_params


if __name__ == "__main__":
    run_optimization()
