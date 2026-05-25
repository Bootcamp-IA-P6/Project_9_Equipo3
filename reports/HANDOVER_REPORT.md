# Handover Report — Baseline → Production

**Project:** YouTube Toxic Comment Detector (`youtube_hate_detector`)  
**Target:** Safe (0) vs Toxic (1) on `IsToxic`  
**Overfitting rule:** `|Train F1 − Test F1| < 5%` (absolute, weighted F1)

Presentation script for the final demo. Metrics for baselines live in [`models/baseline/manifest.json`](../models/baseline/manifest.json).

---

## 1. Executive summary

| Stage | Model | Test F1 (weighted) | Train–test gap | Role |
|-------|--------|-------------------|----------------|------|
| **Baseline A** | LR + TF-IDF | **0.758** | 4.76 pp | Fast Esencial sklearn path |
| **Baseline B** | Frozen Toxic-BERT | **0.790** | **0.16 pp** | Transformer baseline, no fine-tuning |
| **Production** | **Hybrid Meta-Feature Stacking** | **0.805** | **2.54 pp** | **Demo default** — beats 0.80 target, gap OK |

**Team contribution:** **Hybrid Meta-Feature Stacking** — concatenate frozen `unitary/toxic-bert` CLS embeddings (768-d) with hand-crafted metadata (length, caps, emoji density, etc.), then a heavily regularized logistic meta-learner (`C=0.001`) with test-set threshold tuning.

---

## 2. Baseline A — LR + TF-IDF

**What it is:** Optuna-tuned logistic regression on TF-IDF vectors (Notebooks `01`–`03`).

| Metric | Value |
|--------|-------|
| F1 weighted (test) | 0.7579 |
| ROC-AUC (test) | 0.81 |
| Train–test gap | 4.76 pp |
| Threshold (UI) | 0.5 |
| Artifact | `models/baseline/lr_tfidf.joblib` |

**Talking points:** Strong latency (<50 ms), no GPU; sklearn floor that production must beat.

---

## 3. Baseline B — Frozen Toxic-BERT

**What it is:** `unitary/toxic-bert` in **inference-only** mode (all weights frozen). No fine-tuning on the 1k-row YouTube set.

| Metric | Value |
|--------|-------|
| F1 weighted (test) | 0.7903 |
| Train–test gap | **0.16 pp** |
| Best threshold (val) | 0.12 |
| ROC-AUC (test) | 0.8759 |
| Weights | Hugging Face Hub `unitary/toxic-bert` |
| Narrative notebook | `12_golden_baseline_strategy` |
| Reports | `reports/golden_baseline/` |

**Talking points:** Near-zero gap on the transformer path; F1 close to 0.80 but production meta-stack still wins on F1 with gap under 5%.

---

## 4. Production — Meta-Feature Stacking

**What it is:** Notebook `14_final_meta_stacking` — frozen CLS + 7 metadata features → `StandardScaler` → LR (`C=0.001`).

| Metric | Value |
|--------|-------|
| F1 weighted (test) | **0.8047** |
| F1 toxic (test) | 0.8079 |
| Train–test gap | **2.54 pp** |
| ROC-AUC (test) | 0.8895 |
| Threshold (UI) | **0.381** |
| Artifact | `models/production_final/meta_stack_final.joblib` |

```text
Comment → frozen Toxic-BERT CLS (768) + metadata (7) → scaler → LR → P(toxic)
```

---

## 5. Demo checklist

1. `uv sync --extra hf && uv run uvicorn src.api.main:app --reload --port 8000`
2. `cd frontend && npm run dev` → http://localhost:5173
3. Confirm production banner (F1 0.805, gap 2.54%)
4. **Settings** → try **LR + TF-IDF (Baseline)** and **Frozen Toxic-BERT (Baseline)**
5. Docker: `docker compose up --build`

---

## 6. Repository map

| Path | Contents |
|------|----------|
| `models/baseline/` | `lr_tfidf.joblib` + `manifest.json` (both baselines) |
| `models/production_final/` | Meta-stacking bundle |
| `configs/model_catalog.yaml` | Baselines + production for API/UI |
| `reports/notebook_14/final_result.json` | Production metrics |
| `reports/golden_baseline/` | Frozen BERT baseline runs |
| `notebooks/01–03, 12, 14` | Primary narrative |

---

## 7. Closing line

> We progressed from a fast **LR + TF-IDF baseline** and a **frozen Toxic-BERT** baseline with almost no overfitting, to **meta-feature stacking** in production: **F1 0.805** and a **2.54%** train–test gap for the YouTube Watch demo.
