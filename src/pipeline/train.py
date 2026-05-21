import json
import sys
import yaml
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Make sure "src.*" imports work no matter where this script is run from
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.load_data import load_dataset, prepare_xy
from src.data.external import load_external_training_data
from src.features.augmentation import augment_dataset
from src.features.preprocess import build_preprocessor
from src.evaluation.metrics import (
    compute_metrics,
    confusion_matrix_list,
    metrics_gap,
    passes_gap_constraint,
)
from src.utils.config import load_config, project_root


def build_model_pipeline(config: dict, best_params: dict | None = None) -> Pipeline:
    prep = build_preprocessor(config)
    
    # If Optuna already found good params, build the ensemble; otherwise a plain model
    if best_params:
        print("Building soft-voting ensemble (LR + NB + RF) with Optuna hyperparameters...")
        
        ngram_choice = best_params.get("ngram_range", "unigram")
        ngram_range = (1, 1) if ngram_choice == "unigram" else (1, 2)
        
        vectorizer = TfidfVectorizer(
            max_features=best_params["max_features"],
            ngram_range=ngram_range,
            min_df=best_params["min_df"],
            max_df=best_params["max_df"]
        )

        # The three base models of the ensemble, with the Optuna-tuned params
        clf_lr = LogisticRegression(
            C=best_params["lr_C"],
            class_weight="balanced",
            max_iter=1000,
            random_state=config["data"]["random_state"]
        )
        clf_nb = MultinomialNB(alpha=best_params["nb_alpha"])
        clf_rf = RandomForestClassifier(
            n_estimators=best_params["rf_n_estimators"],
            max_depth=best_params["rf_max_depth"],
            class_weight="balanced",
            random_state=config["data"]["random_state"]
        )

        # Soft voting averages the three probability outputs into one prediction
        classifier = VotingClassifier(
            estimators=[("lr", clf_lr), ("nb", clf_nb), ("rf", clf_rf)],
            voting="soft",
        )
    else:
        print("Building baseline Logistic Regression pipeline...")
        feat = config["features"]
        ngram_range = tuple(feat["ngram_range"])

        vectorizer = TfidfVectorizer(
            max_features=feat["max_features"],
            ngram_range=ngram_range,
            min_df=feat["min_df"],
            max_df=feat["max_df"],
        )

        model_cfg = config["model"]
        classifier = LogisticRegression(
            C=model_cfg["C"],
            class_weight=model_cfg.get("class_weight"),
            max_iter=model_cfg["max_iter"],
            random_state=config["data"]["random_state"],
        )

    return Pipeline(
        [
            ("preprocess", prep),
            ("tfidf", vectorizer),
            ("clf", classifier),
        ]
    )


def train(config_path: str | None = None) -> dict:
    config = load_config(config_path)
    root = project_root()

    df = load_dataset(config["data"]["raw_path"])
    x, y = prepare_xy(df, config["data"]["text_column"], config["data"]["label_column"])

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )

    # Pull in outside hate-speech datasets (Davidson, Jigsaw). They join the
    # training set only; the test set above stays pure YouToxic, so the scores
    # keep reflecting our real domain of YouTube comments.
    external = load_external_training_data(config)
    n_external = len(external)
    if n_external:
        x_fit_base = pd.concat([x_train, external["Text"]], ignore_index=True)
        y_fit_base = pd.concat([y_train, external["IsToxic"]], ignore_index=True)
        print(f"Training data: {len(x_train)} YouToxic + {n_external} external = {len(x_fit_base)} rows.")
    else:
        x_fit_base, y_fit_base = x_train, y_train

    # Reuse the tuned params if a previous Optuna run saved them here
    best_params_path = root / "configs" / "best_params.yaml"
    best_params = None
    if best_params_path.exists():
        with open(best_params_path, "r", encoding="utf-8") as f:
            best_params = yaml.safe_load(f)

    # Add synthetic toxic examples to the training data, if the tuned params asked for it
    if best_params and best_params.get("augment_multiplier", 0.0) > 0:
        multiplier = best_params["augment_multiplier"]
        print(f"Applying data augmentation on training set (multiplier={multiplier})...")
        x_train_fit, y_train_fit = augment_dataset(
            x_fit_base, y_fit_base,
            label_to_augment=1,
            multiplier=multiplier,
            random_state=config["data"]["random_state"]
        )
        print(f"Dataset augmented: training samples went from {len(x_fit_base)} to {len(x_train_fit)}.")
    else:
        x_train_fit, y_train_fit = x_fit_base, y_fit_base

    pipeline = build_model_pipeline(config, best_params)
    pipeline.fit(x_train_fit, y_train_fit)

    # Train score uses the YouToxic rows only, so the train/test gap stays a fair comparison
    y_train_pred = pipeline.predict(x_train)
    y_test_pred = pipeline.predict(x_test)

    train_metrics = compute_metrics(y_train, y_train_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)
    gaps = metrics_gap(train_metrics, test_metrics)
    max_gap = config["model"]["max_train_test_gap"]
    gap_ok = passes_gap_constraint(train_metrics, test_metrics, max_gap)

    report = {
        "n_samples": int(len(x)),
        "n_train_original": int(len(x_train)),
        "n_external": int(n_external),
        "n_train_fit": int(len(x_train_fit)),
        "n_test": int(len(x_test)),
        "positive_rate": float(y.mean()),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "gaps": gaps,
        "gap_constraint": max_gap,
        "gap_ok": gap_ok,
        "confusion_matrix_test": confusion_matrix_list(y_test, y_test_pred),
        "is_ensemble": isinstance(pipeline.named_steps["clf"], VotingClassifier),
        "config": best_params if best_params else {
            "C": config["model"]["C"],
            "max_features": config["features"]["max_features"],
            "ngram_range": config["features"]["ngram_range"],
        },
    }

    model_path = root / config["paths"]["model_file"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    metrics_path = root / config["paths"]["metrics_file"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _print_summary(report, model_path)
    return report


def _print_summary(report: dict, model_path: Path) -> None:
    print(f"Model saved: {model_path}")
    print(f"YouToxic train: {report.get('n_train_original', report['n_train_fit'])} | external: {report.get('n_external', 0)} | fit total: {report['n_train_fit']} | test: {report['n_test']}")
    print(f"Positive rate: {report['positive_rate']:.2%}")
    print(f"Is Ensemble Model: {report['is_ensemble']}")
    print("Train Metrics:", report["train_metrics"])
    print("Test Metrics: ", report["test_metrics"])
    print("Gaps:         ", report["gaps"])
    status = "PASS" if report["gap_ok"] else "FAIL"
    print(f"Gap constraint (< {report['gap_constraint']:.0%}): {status}")


def main() -> None:
    root = project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    train()


if __name__ == "__main__":
    main()
