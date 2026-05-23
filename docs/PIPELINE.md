# Training pipeline

Entry point: [`src/pipeline/run_pipeline.py`](../src/pipeline/run_pipeline.py)

## Command

```bash
python -m src.pipeline.run_pipeline --model lr
```

| Flag | Choices | Default |
|------|---------|---------|
| `--model` | `lr`, `rf`, `xgboost` | `lr` |

Run from the repository root so `configs/` and `data/raw/` resolve correctly.

## Phases

1. **Load data** — `load_raw_data()` reads `configs/pipeline.yaml` → `data/raw/youtoxic_english_1000.csv`
2. **Split** — stratified train/test (`test_size`, `random_state` in YAML)
3. **Preprocess** — `TextPreprocessor` (lowercase, regex cleanup, spaCy lemmas, NLTK stopwords)
4. **Train** — `build_model(model_type)` fits TF-IDF + classifier pipeline
5. **Cross-validation** — 5-fold stratified CV, F1 weighted + ROC-AUC
6. **Evaluate** — `Evaluator.evaluate_and_report()` on test set
7. **Save** — `models/experiments/{model}/{model}_pipeline_{timestamp}.joblib`
8. **MLflow** — metrics and sklearn pipeline under `mlruns/`
9. **Reports** — append row to `reports/summary.csv`; PNGs in `reports/pipeline/{model}/`

## Configuration

| File | Keys (examples) |
|------|-----------------|
| `configs/pipeline.yaml` | `target_binary: IsToxic`, `test_size: 0.2`, `cv_folds: 5` |
| `configs/features.yaml` | TF-IDF `max_features`, `ngram_range`, preprocessing flags |
| `configs/models.yaml` | LR `C`, RF `n_estimators`, etc. |
| `configs/best_params.yaml` | Optuna winner for LR (overrides defaults when training LR) |

## Outputs

| Path | Content |
|------|---------|
| `reports/summary.csv` | All runs — model comparison table |
| `reports/pipeline/lr/cm_lr.png` | Confusion matrix |
| `reports/pipeline/lr/roc_lr.png` | ROC curve |
| `reports/pipeline/lr/errors_lr.csv` | False positives / negatives |
| `reports/pipeline/lr/exp_*.json` | Full metrics per run |
| `models/experiments/lr/*.joblib` | Serialized pipeline |

## Evaluator API

[`src/evaluation/evaluator.py`](../src/evaluation/evaluator.py):

```python
from src.evaluation.evaluator import Evaluator

evaluator = Evaluator(output_dir="reports/pipeline/lr")
metrics = evaluator.evaluate_and_report(
    model, X_test, y_test, model_name="LR",
    X_train=X_train, y_train=y_train, cv_results=cv_results,
    summary_path="reports/summary.csv",
)
```

Metrics include: `f1_weighted`, `f1_toxic`, `roc_auc`, `fp`, `fn`, `cv_test_gap_pp`, `train_test_gap_pp`, plus paths to plots.

## Production model

Inference uses `models/final_model.joblib` (loaded by `ModelService`). After a successful pipeline run, copy or export the best experiment artifact to `final_model.joblib` if you want to update production.
