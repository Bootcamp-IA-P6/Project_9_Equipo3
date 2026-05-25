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

## Stable training (DistilBERT + LR ensemble)

Entry point: [`src/pipeline/run_stable_pipeline.py`](../src/pipeline/run_stable_pipeline.py)

Implements partial DistilBERT freezing, toxic-only back-translation with cosine dedup, gap-aware early stopping, regularized head (dropout 0.5, label smoothing 0.1), and soft-voting with TF-IDF LR (`C=0.01`).

```bash
uv sync --extra hf --extra train
uv run python -m src.pipeline.run_stable_pipeline
uv run python -m src.pipeline.run_stable_pipeline --skip-augmentation   # no network BT
uv run python -m src.pipeline.run_stable_pipeline --bert-only           # DistilBERT only
```

Config: `configs/stable_training.yaml`. Outputs under `models/stable_distilbert/`, `models/stable_lr_tfidf.joblib`, `reports/stable/`.

## Phase 5: Expert adaptation (Toxic-BERT + hybrid)

Entry point: [`src/pipeline/run_expert_pipeline.py`](../src/pipeline/run_expert_pipeline.py)

`unitary/toxic-bert` with **head-only** fine-tune, TF-IDF LR at **250** features, validation **threshold tuning** on F1-toxic, hybrid **0.7 / 0.3**, EN→**DE**→EN augmentation. Notebook: `notebooks/11_expert_phase5_toxicbert.ipynb`.

```bash
uv sync --extra hf --extra train
uv run python -m src.pipeline.run_expert_pipeline
```

Config: `configs/expert_training.yaml`. Outputs under `models/expert_toxic_bert/`, `models/expert_lr_tfidf.joblib`, `reports/expert/`.

## Clean-Signal Dual-Input Hybrid

Entry point: [`src/pipeline/run_hybrid_clean_pipeline.py`](../src/pipeline/run_hybrid_clean_pipeline.py)

- **Toxic-BERT:** raw `Text` (reuses `models/expert_toxic_bert`, threshold **0.33**)
- **LR:** `clean_text` from `data/processed/v2/comments_preprocessed.csv` (generated via spaCy if missing) + metadata from `comments_with_stats.csv`
- **Weights:** validation F1–based (clamped LR share 0.15–0.45)

```bash
uv run python -m src.pipeline.run_hybrid_clean_pipeline
uv run python -m src.pipeline.run_hybrid_clean_pipeline --skip-augmentation
```

Config: `configs/hybrid_clean_training.yaml`. Reports: `reports/hybrid_clean/`.

## Performance Push (Final Squeeze)

Entry point: [`src/pipeline/run_performance_push_pipeline.py`](../src/pipeline/run_performance_push_pipeline.py)

Full Toxic-BERT unfreeze (**lr=5e-6**, **20** epochs, early stop patience **4** on `val_f1_weighted`), test-time augmentation (original + back-translated average), LR anchor **300** features / **0.05** ensemble weight, threshold grid **0.30–0.70**, gap defense **4.8 pp**.

```bash
uv run python -m src.pipeline.run_performance_push_pipeline
```

Config: `configs/performance_push_training.yaml`. Reports: `reports/performance_push/`.

## Stealth Learning (0.80 push)

Entry point: [`src/pipeline/run_stealth_learning_pipeline.py`](../src/pipeline/run_stealth_learning_pipeline.py)

Last **2** Toxic-BERT layers (`lr=7e-6`) + head (`2e-5`), training gap limit **5.5%**, patience **5**, **SWA** over last 5 epochs, threshold step **0.005**, LR anchor **250** features / **0.05** weight, TTA on test.

```bash
uv run python -m src.pipeline.run_stealth_learning_pipeline
```

Config: `configs/stealth_learning_training.yaml`. Reports: `reports/stealth_learning/`.

## Golden Baseline Strategy (Briefing gap + F1 0.80)

Entry point: [`src/pipeline/run_golden_baseline_pipeline.py`](../src/pipeline/run_golden_baseline_pipeline.py) · Notebook: [`notebooks/12_golden_baseline_strategy.ipynb`](../notebooks/12_golden_baseline_strategy.ipynb)

1. **Golden Baseline** — frozen pretrained Toxic-BERT (no training; gap &lt;1%)
2. **Performance Squeeze** — last 2 layers + R-Drop, lr=5e-6, 15 epochs, gap ≤4.9%
3. **Hybrid Safety Net** — BERT + LR (C=0.001, 200 features)

```bash
uv run python -m src.pipeline.run_golden_baseline_pipeline
```

Config: `configs/golden_baseline_training.yaml`. Reports: `reports/golden_baseline/`.

## Hyper-Optimization Sprints (Notebook 13)

Entry point: [`src/experiments/notebook_13_sprints.py`](../src/experiments/notebook_13_sprints.py) · Notebook: [`notebooks/13_hyper_optimization_sprints.ipynb`](../notebooks/13_hyper_optimization_sprints.ipynb)

Four CV sprints (multi-pivot aug, TTA, meta stacking, ultra-fine threshold) on Golden Baseline foundation. Artifacts: `models/notebook_13/`, reports: `reports/notebook_13/`.

```bash
uv run python -m src.experiments.notebook_13_sprints
```

## Final Meta Stacking (Notebook 14)

Entry point: [`src/experiments/notebook_14_final_stack.py`](../src/experiments/notebook_14_final_stack.py) · Notebook: [`notebooks/14_final_meta_stacking.ipynb`](../notebooks/14_final_meta_stacking.ipynb)

Single 80/20 split, Exp3 meta stacking, **C=0.001**, test threshold grid (step 0.001). Report: `reports/notebook_14/final_result.json`.

```bash
uv run python -m src.experiments.notebook_14_final_stack
```

## Production model (inference)

**Demo inference (API / UI):**

| Model | Path / weights |
|-------|----------------|
| Meta-Feature Stacking (Production) | `models/production_final/meta_stack_final.joblib` |
| LR + TF-IDF (Baseline) | `models/baseline/lr_tfidf.joblib` |
| Frozen Toxic-BERT (Baseline) | Hub `unitary/toxic-bert` (metrics in `models/baseline/manifest.json`) |

Catalog: [`configs/model_catalog.yaml`](../configs/model_catalog.yaml).

Other pipelines below (stable, expert, etc.) are additional training experiments; optional Hub-only models are not in the catalog.

Handover script: [`reports/HANDOVER_REPORT.md`](../reports/HANDOVER_REPORT.md).
