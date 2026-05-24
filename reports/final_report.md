# Technical Results Report — YouTube Toxic Comment Detector (SignalMod)

**Project:** Binary moderation assistant (Safe vs Toxic) for YouTube-style comments  
**Dataset:** `youtoxic_english_1000.csv` (1,000 rows)  
**Production model:** `models/final_model.joblib` — Logistic Regression + TF-IDF (Optuna-tuned)  
**Report date:** 2026-05-23  
**Related artifacts:** [`summary.csv`](summary.csv) · [`pipeline/lr/`](pipeline/lr/) · EDA [`v2/`](v2/)

---

## 1. Executive summary

We built an end-to-end NLP pipeline (preprocess → TF-IDF → classifier), a FastAPI inference service, and a Streamlit demo. The **selected production model** is **LR + TF-IDF** tuned with Optuna, reaching **F1 (weighted) = 0.7579** and **ROC-AUC = 0.81** on the held-out test split, with a **CV–test gap of 4.76 pp** (within the project’s &lt; 5 pp overfitting target). Transformer models are available optionally via the API catalog but were not adopted as the default due to latency, dependency weight, and lack of evaluation on the same project test split.

---

## 2. Decisions made

| Area | Decision | Rationale (from implementation) |
|------|----------|--------------------------------|
| **Task formulation** | Binary classification on `IsToxic` | Configured in `configs/pipeline.yaml`; multilabel columns exist but default mode is `binary`. |
| **Labels (UI/API)** | **Safe** / **Toxic** | User-facing copy in API (`is_toxic`) and Streamlit; raw CSV keeps `IsToxic`. |
| **Preprocessing** | Lowercase → regex cleanup → spaCy lemmas → NLTK + custom stopwords | `TextPreprocessor` in `src/features/text_preprocessor.py`. |
| **Sensitive tokens** | Keep *black*, *white*, *police*, *cop*, etc. | EDA decision documented in preprocessor: needed for context/bigrams, not removed as stopwords. |
| **Vectorization** | TF-IDF, unigrams + bigrams | Captures phrases like toxic bigrams; tuned `max_features=4045`, `min_df=2`. |
| **Baseline family** | LR (winner), RF, XGBoost available | `build_model()` in `src/models/baseline.py`; pipeline flag `--model`. |
| **Hyperparameter search** | Optuna → `configs/best_params.yaml` | Best LR stored; exported to `final_model.joblib`. |
| **Primary metric** | F1 weighted + ROC-AUC | `configs/models.yaml` → `evaluation.primary_metric`. |
| **Overfitting control** | \|CV F1 − test F1\| &lt; 5 pp | Reported as `cv_test_gap_pp` in `Evaluator`. |
| **Class imbalance** | `class_weight: balanced` (LR/RF) | `configs/models.yaml`. |
| **Serving** | `ModelService` + FastAPI + Streamlit | Local joblib default; HF models switchable via `PUT /model/{name}`. |
| **Deployment** | Docker Compose (`youtube_hate_detector`) | API :8000, Streamlit :8501. |
| **Experiment tracking** | MLflow (`mlruns/`) + `reports/summary.csv` | Pipeline phase 8–9. |

---

## 3. Dataset overview and limitations

### 3.1 Statistics (raw split source)

| Statistic | Value |
|-----------|-------|
| Total comments | 1,000 |
| Safe (`IsToxic = 0`) | 538 (53.8%) |
| Toxic (`IsToxic = 1`) | 462 (46.2%) |
| Mean comment length | ~186 characters |

Train/test: **80/20 stratified** (`test_size: 0.2`, `random_state: 42`) → **200 test samples** per pipeline run.

### 3.2 Limitations

1. **Small sample size (~1k rows)** — High variance in metrics; rare toxicity subtypes are underrepresented.
2. **Single language (English)** — Model and tokenizer choices assume English YouTube comments.
3. **Topic / event bias** — Many comments relate to specific news videos (e.g. Ferguson-era discourse); generalization to other channels is uncertain.
4. **Multilabel sparsity** — Columns such as `IsRacist`, `IsSexist`, `IsHomophobic` have very few positives; project stayed on binary `IsToxic` (see EDA plots in `reports/v2/05_multilabel_overlap.png`).
5. **Label noise** — Crowd/annotator subjectivity on edge cases (sarcasm, reclaimed slurs, political rhetoric).
6. **Preprocessing information loss** — Lemmatization and stopword removal can remove cues; empty texts are back-filled with raw text in the pipeline.
7. **Identity-related vocabulary** — Frequent tokens (*black*, *white*, *police*) appear in both errors and valid discussion, increasing false positives/negatives on racial/political content.

---

## 4. Metrics — all models

Canonical source: [`summary.csv`](summary.csv). Plots: `reports/pipeline/{model}/`, EDA: `reports/v2/`.

### 4.1 Production sklearn model (Optuna-tuned LR)

From `configs/best_params.yaml` — metrics on **held-out test** after tuning (reference for `final_model.joblib`):

| Metric | Value |
|--------|-------|
| F1 (weighted, test) | **0.7579** |
| F1 (train) | 0.8987 |
| ROC-AUC | **0.81** |
| False positives (FP) | 18 |
| False negatives (FN) | 30 |
| Train–test gap | 14.07 pp |
| **CV–test gap** | **4.76 pp** ✓ (&lt; 5 pp) |

**Tuned hyperparameters:** `C ≈ 0.32`, `max_features = 4045`, `ngram_range = (1,2)`, `min_df = 2`, `sublinear_tf = false`.

### 4.2 Pipeline re-run (LR, default config path)

Latest automated run (`reports/pipeline/lr/exp_20260523_163600_lr.json`) — same split/preprocess, documents reproducibility:

| Metric | Value |
|--------|-------|
| F1 (weighted, test) | 0.7387 |
| F1 (toxic, pos=1) | 0.7045 |
| ROC-AUC | 0.7838 |
| Accuracy | 0.74 |
| FP / FN | 22 / 30 |
| CV F1 mean ± std | 0.7193 ± 0.0382 |
| CV–test gap | 1.94 pp |
| Train–test gap | 15.97 pp |

The tuned artifact (**0.7579**) outperforms this default-config rerun; production uses the tuned weights in `final_model.joblib`.

### 4.3 Sklearn baselines not yet in summary table

| Model | Status | Action |
|-------|--------|--------|
| Random Forest (`rf`) | Not evaluated in `summary.csv` | `python -m src.pipeline.run_pipeline --model rf` |
| XGBoost (`xgboost`) | Not evaluated in `summary.csv` | `python -m src.pipeline.run_pipeline --model xgboost` |

### 4.4 Hugging Face models (API catalog)

Available in `ModelService` for live switching; **not benchmarked on the project test split** in `summary.csv`:

| Model | Type | Catalog F1 note | Production default |
|-------|------|-----------------|--------------------|
| LR + TF-IDF (local) | joblib | 0.76 (aligned with tuning) | **Yes** |
| DistilBERT Toxicity | HF remote | ~0.85 (external) | No |
| toxic-bert (multilabel) | HF remote | ~0.88 (external, Jigsaw) | No |
| RoBERTa Toxicity | HF remote | ~0.87 (external) | No |
| Fine-tuned local HF | HF local | To be evaluated | No |

Notebook experiments (confusion matrices): `reports/v2/nb08_*`.

### 4.5 Comparison table (summary)

| Model | F1 (test) | ROC-AUC | FP | FN | CV–test gap (pp) | Default |
|-------|-----------|---------|----|----|------------------|---------|
| **LR + TF-IDF (tuned)** | **0.7579** | **0.81** | 18 | 30 | **4.76** | Yes |
| LR (pipeline rerun) | 0.7387 | 0.784 | 22 | 30 | 1.94 | — |
| RF | — | — | — | — | — | Run pipeline |
| XGBoost | — | — | — | — | — | Run pipeline |
| HF catalog models | — | — | — | — | — | Optional |

---

## 5. Selected model and justification

**Selected:** **Logistic Regression + TF-IDF** (`models/final_model.joblib`, Optuna-tuned).

**Why this model**

1. **Best project test performance** among trained sklearn artifacts (F1 0.7579, ROC-AUC 0.81).
2. **Meets generalization criterion** — CV–test gap 4.76 pp &lt; 5 pp (train–test gap is higher but in-sample; CV gap is the rubric metric).
3. **Operational fit** — Inference &lt; 50 ms, no GPU, small artifact (~100 KB joblib), runs in Docker without downloading large transformers.
4. **Interpretability** — TF-IDF coefficients inspectable (`reports/v2/11_lr_coeficientes.png`); helps moderation audits.
5. **Stable stack** — sklearn + spaCy + NLTK already used in training pipeline; same `ModelService` path for API and Streamlit.

**Why not transformers by default**

- Heavier dependencies (torch, transformers), slower cold start, higher memory.
- Catalog accuracy figures are **not** measured on `youtoxic_english_1000` test split in this repo.
- Acceptable as **optional** models via API for comparison demos.

---

## 6. Error analysis

Source: `reports/pipeline/lr/errors_lr.csv` and evaluator term counts (latest LR pipeline run, 52 errors on n=200 test).

### 6.1 Confusion summary (pipeline LR run)

| | Predicted Safe | Predicted Toxic |
|--|----------------|-----------------|
| **Actual Safe** | TN | **22 FP** |
| **Actual Toxic** | **30 FN** | TP |

Production tuned model: **18 FP / 30 FN** (fewer false positives).

### 6.2 Most common terms in errors

**False positives** (safe text marked toxic) — frequent lemmas:

`black(14)`, `white(9)`, `shoot(8)`, `would(9)`, `police(5)`, `cop(6)`, `people(7)`, `guy(7)`

→ Model often reacts to **identity and violence-related vocabulary in non-toxic political/news context**.

**False negatives** (toxic missed) — frequent lemmas:

`police(8)`, `black(6)`, `criminal(6)`, `kill(5)`, `make(6)`, `people(5)`

→ Missed toxicity when phrasing is **indirect, sarcastic, or embedded in long argumentative comments** (probabilities just below threshold).

### 6.3 Representative patterns

| Type | Pattern | Example risk |
|------|---------|----------------|
| FP | Political/racial discussion without insult | Comments about Ferguson, police, “black/white” discourse |
| FP | Profanity in non-directed context | e.g. mild expletives in traffic/frustration |
| FN | Long, indirect hate | Low model score despite toxic content in body |
| FN | Sarcasm / coded language | Hard for bag-of-words + linear model |

Artifacts: [`pipeline/lr/cm_lr.png`](pipeline/lr/cm_lr.png), [`pipeline/lr/roc_lr.png`](pipeline/lr/roc_lr.png).

---

## 7. Future improvements

| Priority | Improvement | Expected benefit |
|----------|-------------|----------------|
| High | **More labeled data** + domain-balanced sampling | Reduce FP on political text; improve recall |
| High | **Threshold tuning** on validation (per use case) | Trade FP vs FN for moderation policy |
| High | **Evaluate RF/XGBoost and fine-tuned DistilBERT** on same test split | Fair comparison in `summary.csv` |
| Medium | **Back-translation / synonym augmentation** | Mitigate small-data overfitting (EDA: `reports/v2/15_augmentation_*`) |
| Medium | **Fine-tune DistilBERT on project data** | Capture context better than TF-IDF |
| Medium | **Ensemble** LR + transformer vote | Notebook track: `reports/v2/12_ensemble_comparativa.png` |
| Medium | **Human-in-the-loop review queue** | Route borderline scores (e.g. 0.4–0.6) to moderators |
| Low | **Multilabel heads** for sparse sublabels | Only if minimum positive counts per label are met |
| Low | **React frontend** (if replacing Streamlit) | Production UX parity with YouTube |
| Low | **PostgreSQL logging** | Audit trail for predictions in production |

---

## 8. Reproducibility

```bash
# Retrain and refresh metrics
python -m src.pipeline.run_pipeline --model lr

# View comparison table
cat reports/summary.csv

# Run API + UI
docker compose up --build
```

---

## 9. References

| Document | Path |
|----------|------|
| Model comparison CSV | [`summary.csv`](summary.csv) |
| Results (EN) | [`../docs/RESULTS.md`](../docs/RESULTS.md) |
| Pipeline (EN) | [`../docs/PIPELINE.md`](../docs/PIPELINE.md) |
| API (EN) | [`../docs/API.md`](../docs/API.md) |
| Best hyperparameters | [`../configs/best_params.yaml`](../configs/best_params.yaml) |
