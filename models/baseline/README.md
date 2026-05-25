# Baseline models

| Entry in `manifest.json` | UI name | On disk |
|--------------------------|---------|---------|
| `lr_tfidf` | LR + TF-IDF (Baseline) | `lr_tfidf.joblib` |
| `frozen_toxic_bert` | Frozen Toxic-BERT (Baseline) | Hugging Face `unitary/toxic-bert` at runtime |

Reports for frozen BERT: `reports/golden_baseline/`. Production model: `../production_final/`.
