# Model results and comparison

Canonical data: [`reports/summary.csv`](../reports/summary.csv)  
Tuned hyperparameters: [`configs/best_params.yaml`](../configs/best_params.yaml)  
**Full technical report:** [`reports/final_report.md`](../reports/final_report.md) · [ES](../reports/final_report.es.md)

## Best sklearn model (production)

**Winner:** Logistic Regression + TF-IDF (Optuna-tuned), exported as `models/final_model.joblib`.

| Metric | Test value | Notes |
|--------|------------|-------|
| F1 (weighted) | **0.7579** | Primary project metric |
| ROC-AUC | **0.81** | Ranking quality |
| False positives | **18** | Safe comments marked toxic |
| False negatives | **30** | Toxic comments missed |
| F1 (train) | 0.8987 | In-sample |
| Train–test gap | 14.07 pp | High; prefer CV gap for generalization |
| CV–test gap | **4.76 pp** | Meets &lt; 5 pp rubric |
| Test size | ~20% stratified | See `configs/pipeline.yaml` |

**Optuna hyperparameters (LR):** `C≈0.32`, `max_features=4045`, bigrams `(1,2)`, `min_df=2`.

## Comparison table

| Model | Family | F1 (test) | ROC-AUC | FP | FN | Default in API/UI |
|-------|--------|-----------|---------|----|----|-------------------|
| LR + TF-IDF (tuned) | sklearn | 0.7579 | 0.81 | 18 | 30 | Yes |
| LR + TF-IDF (local) | sklearn | 0.7579 | 0.81 | 18 | 30 | Yes (`final_model.joblib`) |
| Random Forest | sklearn | — | — | — | — | Run pipeline `--model rf` |
| XGBoost | sklearn | — | — | — | — | Run pipeline `--model xgboost` |
| DistilBERT Toxicity | Hugging Face | — | — | — | — | Optional (`PUT /model/...`) |
| toxic-bert (multilabel) | Hugging Face | — | — | — | — | Optional |
| RoBERTa Toxicity | Hugging Face | — | — | — | — | Optional |

Rows with empty metrics are placeholders until you run the pipeline or evaluate HF models on the same test split.

## How to refresh metrics

```bash
python -m src.pipeline.run_pipeline --model lr
python -m src.pipeline.run_pipeline --model rf
python -m src.pipeline.run_pipeline --model xgboost
```

Each run appends/updates [`reports/summary.csv`](../reports/summary.csv) and writes:

- `reports/pipeline/{model}/cm_{model}.png`
- `reports/pipeline/{model}/roc_{model}.png`
- `reports/pipeline/{model}/errors_{model}.csv`

## EDA and experiments

Additional figures (notebooks): `reports/v2/` — label distribution, TF-IDF features, ensemble charts, transformer confusion matrices (`nb08_*`).

## Error analysis

The evaluator prints and saves:

- **Most common terms** in false positives and false negatives
- Example comments with highest/lowest toxic probability among errors

See `reports/pipeline/*/errors_*.csv` after a pipeline run.
