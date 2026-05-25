# Resultados y comparativa

**Catálogo demo:** [`configs/model_catalog.yaml`](../configs/model_catalog.yaml) · Baselines: [`models/baseline/manifest.json`](../models/baseline/manifest.json)

| Modelo | F1 (test, ponderado) | Brecha train–test | Por defecto |
|--------|----------------------|-------------------|-------------|
| LR + TF-IDF (Baseline) | 0,758 | 4,76 pp | No |
| Frozen Toxic-BERT (Baseline) | 0,790 | 0,16 pp | No |
| **Meta-Feature Stacking (Production)** | **0,805** | **2,54 pp** | **Sí** |

**Guion:** [`reports/HANDOVER_REPORT.md`](../reports/HANDOVER_REPORT.md)

## Baselines

- **LR + TF-IDF:** `models/baseline/lr_tfidf.joblib`
- **Frozen Toxic-BERT:** Hub `unitary/toxic-bert`, informes en `reports/golden_baseline/`

## Producción

```bash
uv run python -m src.experiments.notebook_14_final_stack
```

Requiere `uv sync --extra hf`.
