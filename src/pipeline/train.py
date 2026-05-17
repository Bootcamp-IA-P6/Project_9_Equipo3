import json
import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.load_data import load_dataset, prepare_xy
from src.evaluation.metrics import (
    compute_metrics,
    confusion_matrix_list,
    metrics_gap,
    passes_gap_constraint,
)
from src.features.preprocess import build_preprocessor
from src.utils.config import load_config, project_root


def build_model_pipeline(config: dict) -> Pipeline:
    prep = build_preprocessor(config)
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

    pipeline = build_model_pipeline(config)
    pipeline.fit(x_train, y_train)

    y_train_pred = pipeline.predict(x_train)
    y_test_pred = pipeline.predict(x_test)

    train_metrics = compute_metrics(y_train, y_train_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)
    gaps = metrics_gap(train_metrics, test_metrics)
    max_gap = config["model"]["max_train_test_gap"]
    gap_ok = passes_gap_constraint(train_metrics, test_metrics, max_gap)

    report = {
        "n_samples": int(len(x)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "positive_rate": float(y.mean()),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "gaps": gaps,
        "gap_constraint": max_gap,
        "gap_ok": gap_ok,
        "confusion_matrix_test": confusion_matrix_list(y_test, y_test_pred),
        "config": {
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
    print(f"Samples: {report['n_samples']} (train {report['n_train']}, test {report['n_test']})")
    print(f"Positive rate: {report['positive_rate']:.2%}")
    print("Train:", report["train_metrics"])
    print("Test: ", report["test_metrics"])
    print("Gaps: ", report["gaps"])
    status = "PASS" if report["gap_ok"] else "FAIL"
    print(f"Gap constraint (< {report['gap_constraint']:.0%}): {status}")


def main() -> None:
    root = project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    train()


if __name__ == "__main__":
    main()
