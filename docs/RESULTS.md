# Model results and comparison

**Demo catalog:** [`configs/model_catalog.yaml`](../configs/model_catalog.yaml) · Baseline metrics: [`models/baseline/manifest.json`](../models/baseline/manifest.json)

| Model | F1 (test, weighted) | Train–test gap | Default in UI |
|-------|---------------------|----------------|---------------|
| LR + TF-IDF (Baseline) | 0.758 | 4.76 pp | No |
| Frozen Toxic-BERT (Baseline) | 0.790 | 0.16 pp | No |
| **Meta-Feature Stacking (Production)** | **0.805** | **2.54 pp** | **Yes** |

**Handover:** [`reports/HANDOVER_REPORT.md`](../reports/HANDOVER_REPORT.md) · **Production JSON:** [`reports/notebook_14/final_result.json`](../reports/notebook_14/final_result.json) · **Golden baseline:** [`reports/golden_baseline/`](../reports/golden_baseline/)

## Baselines

**LR + TF-IDF** — Notebooks 01–03, artifact `models/baseline/lr_tfidf.joblib`, tuning in [`configs/best_params.yaml`](../configs/best_params.yaml).

**Frozen Toxic-BERT** — Notebook 12, `unitary/toxic-bert` inference-only; see golden baseline reports and `manifest.json` → `frozen_toxic_bert`.

## Production

```bash
uv run python -m src.experiments.notebook_14_final_stack
```

Requires `uv sync --extra hf`.

## Other experiments

Historical table: [`reports/summary.csv`](../reports/summary.csv). RF/XGBoost pipelines and `reports/v2/` figures are teammate or archived work — not in the demo model catalog.
