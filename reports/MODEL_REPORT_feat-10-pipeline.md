# Branch model report (`feat/10-pipeline`)

Metrics below come from **executed notebook outputs** in the v2 series. Two evaluation setups exist:

- **Classical (nb04–07):** stratified hold-out, **200 test** rows (~80/20 from ~1000 samples).
- **Transformers (nb08):** train/val/test split, **150 test** rows (~15%). Do not compare F1 across setups as if they were the same split.

**Artifacts on disk today:** `models/final_model.joblib` (LR tuned), `models/finetuned_hf/` (DistilBERT). `lr_baseline.joblib` and `best_ensemble.joblib` are referenced in notebooks but **not** present in the current `models/` folder.

---

## 1. Classical ML (TF-IDF pipelines)

| Model | Notebook | F1 test (weighted) | F1 train | Train–test gap | CV mean | CV–test gap | ROC-AUC (test) | Accuracy (test) | Artifact / notes |
|--------|----------|------------------|----------|----------------|---------|-------------|----------------|-----------------|------------------|
| **Logistic Regression baseline** | `04_baseline_v2.ipynb` | **0.7531** | 0.8623 | 10.91 pp | 0.7015 | 5.17 pp | 0.7737 (CV val) / **0.8075** (nb05 final on LR) | 0.76 | `models/lr_baseline.joblib` (nb04; missing on disk) |
| **"RF baseline"** (loaded in nb06) | `06_tuning_clean_v2.ipynb` | **0.7531** | 0.8623 | 10.91 pp | 0.7015 | 5.17 pp | — | — | Loaded from `models/best_ensemble.joblib` (same metrics as LR; file missing on disk) |
| **Complement NB** | `05_ensemble_v2.ipynb` | — (CV only) | 0.8970 | 22.07 pp (train–val) | **0.6762** | — | 0.7709 | — | Not saved as production |
| **Random Forest** (ensemble compare) | `05_ensemble_v2.ipynb` | — (CV only) | 0.8212 | 12.70 pp | **0.6942** | — | 0.7567 | — | Pipeline in notebook only |
| **XGBoost** | `05_ensemble_v2.ipynb` | — (CV only) | 0.7602 | 12.72 pp | **0.6331** | — | 0.6535 | — | Not saved |
| **LR ensemble winner** (CV winner → refit on train) | `05_ensemble_v2.ipynb` | **0.7531** | 0.8623 | 10.91 pp | 0.7015 (LR ref) | — | **0.8075** | 0.76 | `models/best_ensemble.joblib` (LR pipeline; missing on disk) |
| **LinearSVC** | `06_tuning_clean_v2.ipynb` | **0.7250** | 0.9649 | 23.99 pp | 0.6847 | 4.03 pp | — | — | Not saved |
| **LR tuned (Optuna)** | `06_tuning_clean_v2.ipynb` | **0.7579** | 0.8987 | 14.07 pp | 0.7104 | **4.76 pp** | — | — | **`models/final_model.joblib`** — best classical F1 on 200-test split |
| **RF tuned (Optuna)** | `06_tuning_clean_v2.ipynb` | **0.6924** | 0.8133 | 12.09 pp | 0.7139 | **2.15 pp** | — | — | Not saved separately |

**nb06 summary table (printed):** LR tuned marked winner for F1 test; RF tuned best **cv–test gap** (2.15 pp).

---

## 2. Data augmentation (same LR tuned backbone, nb07)

All rows use the tuned LR pipeline retrained on augmented train; **test unchanged**.

| Model | Notebook | F1 test | Δ vs no-aug | F1 train | Train–test gap | CV mean | CV–test gap | FP / FN |
|--------|----------|---------|-------------|----------|----------------|---------|-------------|---------|
| **LR tuned (no augmentation)** | `07_augmentation_clean_v2.ipynb` | **0.7579** | — | 0.8987 | 14.07 pp | 0.7104 | 4.76 pp | 18 / 30 |
| **LR + WordNet** | same | **0.6855** | −7.24 pp | 0.9307 | 24.52 pp | 0.7452 | 5.97 pp | 35 / 28 |
| **LR + EDA** | same | **0.7054** | −5.25 pp | 0.9401 | 23.47 pp | 0.7410 | 3.56 pp | 32 / 27 |
| **LR + back-translation** | same | **0.7271** | −3.08 pp | 0.9409 | 21.38 pp | 0.7736 | 4.65 pp | 20 / 34 |

**Conclusion in notebook:** augmentation **hurt** test F1 vs baseline; no new artifact promoted.

---

## 3. Transformers and zero-shot (nb08, 150-test split)

| Model | Notebook | Type | F1 weighted (test) | ROC-AUC | FP | FN | Saved artifact |
|--------|----------|------|-------------------|---------|----|----|----------------|
| **DistilBERT base (fine-tuned)** | `08_transformers_clean_v2.ipynb` | Fine-tuned | **0.7735** | **0.8561** | 22 | 12 | **`models/finetuned_hf/`** (+ `model_metadata.json`) |
| **DistilBERT toxic (martin-ha, fine-tuned)** | same | Fine-tuned | **0.7091** | 0.8077 | 15 | 28 | `models/nb08_toxic_distilbert` (path in notebook; not in current disk listing) |
| **RoBERTa Hate (cardiffnlp, fine-tuned)** | same | Fine-tuned | **0.7071** | 0.8456 | 24 | 20 | `models/nb08_roberta` (notebook path) |
| **LR + TF-IDF (nb06 ref on nb08 test)** | same | Baseline sklearn | **0.6944** | 0.7794 | 15 | 30 | Uses `final_model.joblib` |
| **Zero-shot DistilBERT toxic** | same | Zero-shot | **0.6670** | 0.8048 | 12 | 36 | — |
| **Zero-shot RoBERTa hate** | same | Zero-shot | **0.4320** | 0.7500 | 1 | 65 | — |

**nb08 winner:** DistilBERT base — **+7.91 pp** F1 vs LR reference on the **same** 150-sample test set. Test **accuracy** for DistilBERT in classification report: **0.77**.

`models/finetuned_hf/model_metadata.json`: `f1_weighted: 0.7735`, `roc_auc: 0.8561`.

---

## 4. Notebooks without a scored production model

| Notebook | Role |
|----------|------|
| `01_eda_v2.ipynb` | EDA only (~1000 rows, ~46% toxic) |
| `02_preprocessing_v2.ipynb` | Cleaning / lemmatization → `data/processed/` |
| `03_vectorization_v2.ipynb` | TF-IDF feature config |

---

## 5. Ranking (by reported F1 test / weighted)

| Rank | Model | Notebook | F1 | Notes |
|------|--------|----------|-----|--------|
| 1 | **DistilBERT base FT** | 08 | **0.7735** | 150-test split; on disk |
| 2 | **LR tuned (Optuna)** | 06 | **0.7579** | 200-test split; `final_model.joblib` |
| 3 | LR / ensemble LR | 04, 05, 06 | 0.7531 | Same score across several runs |
| 4 | LR + back-trans aug | 07 | 0.7271 | Worse than no aug |
| 5 | LinearSVC | 06 | 0.7250 | |
| 6 | DistilBERT toxic FT | 08 | 0.7091 | |
| 7 | RoBERTa hate FT | 08 | 0.7071 | |
| 8 | LR + EDA aug | 07 | 0.7054 | |
| 9 | LR ref on nb08 test | 08 | 0.6944 | Different split |
| 10 | RF tuned | 06 | 0.6924 | Best cv–test gap among tuned sklearn |
| 11 | LR + WordNet aug | 07 | 0.6855 | |
| 12 | Zero-shot toxic DistilBERT | 08 | 0.6670 | |
| 13 | Zero-shot RoBERTa hate | 08 | 0.4320 | |

### 5.1 AGENTS.md goal vs actual model on this branch

[AGENTS.md](../AGENTS.md) defines what “success” means for the project. This branch delivers strong notebook metrics but does not fully meet the Esencial bar or the team’s deployment stack. The gap is about **selection criteria**, **generalization rule**, and **delivery path**, not only raw F1.

#### What AGENTS.md asks for

| Topic | AGENTS.md expectation |
|--------|------------------------|
| **Business goal** | Automate **toxic** comment detection (`IsToxic` → Safe / Toxic); **practicality over precision**. |
| **Esencial model** | Lightweight classical ML (Naïve Bayes, logistic regression, random forest) on TF-IDF; stratified train/test. |
| **Overfitting (Esencial)** | `\|Train metric − Test metric\| < 5` **percentage points** on the held-out test set (Phase 3). |
| **Imbalance** | Prefer **F1** or **AUC-ROC** over accuracy when toxic class is minority (here ~46% toxic — less critical). |
| **UI / inference** | **FastAPI** + **React**; browser calls `/predict` only (not Streamlit). |
| **Tooling** | **`uv`** (`pyproject.toml`, `uv.lock`); config in `configs/`; modular `src/evaluation/` for metrics. |
| **Avanzado / Experto** | Optional DL (RNN/LSTM), then transformers + MLflow; Docker with React build; DB for predictions. |

#### What this branch actually ships

| Topic | Actual on `feat/10-pipeline` |
|--------|------------------------------|
| **Label** | Binary **IsToxic** — aligned with AGENTS default. |
| **Best test F1** | **DistilBERT base** fine-tuned — F1 weighted **0.7735** (`08_transformers_clean_v2`, 150-test split), artifact `models/finetuned_hf/`. |
| **Default in demo UI** | **LR tuned + TF-IDF** — F1 test **0.7579** (`06_tuning_clean_v2`, 200-test split), artifact `models/final_model.joblib`; first option in Streamlit. |
| **AGENTS train–test gap** | **No top-F1 model passes** the `< 5 pp` rule on **train vs test** F1 (weighted): LR tuned **14.07 pp**, LR/RF baselines **~10.9 pp**, DistilBERT **not logged** in `model_metadata.json`. |
| **Notebook rubric (nb06)** | Several models pass **cv–test gap** `< 5 pp` (e.g. LR tuned **4.76 pp**, RF tuned **2.15 pp**). That is **not** the same as AGENTS’s train–test rule. |
| **Esencial classifiers** | LR and RF covered; **Naïve Bayes** only in ensemble CV table (Complement NB), not as production baseline. |
| **UI / stack** | **Streamlit** app (`src/app/`) — briefing-style, not AGENTS **React + FastAPI**. |
| **Tooling** | Notebooks + partial configs; **`uv` / `pyproject.toml` not on branch**; no `src/evaluation/overfitting_check.py` wired to reports. |
| **Phase 5** | Transformers + local MLflow in nb08 — partial; **no** Docker+React deploy, **no** prediction DB. |

#### Side-by-side: goal vs what you would pick today

| If you optimize for… | AGENTS-aligned choice | What this branch optimizes for |
|----------------------|------------------------|--------------------------------|
| **Strict Esencial gap** (`train − test` F1) | None verified among leaders; **RF tuned** has lowest **cv–test** gap (2.15 pp) but **lower F1** (0.6924) | Notebooks highlight **LR tuned** (best classical F1, cv–test 4.76 pp) |
| **Maximum test F1** | Not specified as primary if gap fails | **DistilBERT** (0.7735) — saved as `finetuned_hf/` |
| **Practical demo (briefing)** | FastAPI + React | **Streamlit** + **LR** default (fast, local joblib) |
| **Practicality over precision** | Favor deployable, stable model | Mixed: best score is DL; default is classical |

#### Summary

- **Goal (AGENTS):** Ship a **generalizing** classical baseline under the **5 pp train–test** rule, exposed via **FastAPI + React**, with **`uv`** and evaluation helpers in `src/`.
- **Actual:** Strong **research trail** (v2 notebooks) with **best F1 = DistilBERT**, but **Esencial overfitting rule is not met** by the models with the highest F1; the **running UI defaults to LR tuned**, which also **fails** the AGENTS train–test gap. **Stack and packaging** lag AGENTS (Streamlit, no `uv`, no canonical overfitting report in code).
- **Implication:** For AGENTS compliance, either **re-train / regularize** until train–test gap `< 5 pp` (and document accuracy + F1), **promote DistilBERT only after** computing the same gap on hold-out, or **document an explicit exception** (practicality) while finishing React + FastAPI and `uv`.

**Practical best on branch (metrics only):** **DistilBERT fine-tuned** (`08_transformers_clean_v2` → `models/finetuned_hf/`).

**Best classical + saved for Streamlit default path:** **LR tuned** (`06_tuning_clean_v2` → `final_model.joblib`).

---

## 6. Streamlit vs this report

- **UI default:** first option **"LR + TF-IDF (local)"** → loads `final_model.joblib` when present.
- **Highest F1 in branch:** DistilBERT in **`models/finetuned_hf/`** (select **"DistilBERT (local HF)"** in the app).

---

## 7. Related files

- Partial machine-readable summary: `reports/final_comparison.json`
- Branch overview (if present): `reports/BRANCH_REPORT_feat-10-pipeline.md`

---

*Generated from executed v2 notebook outputs. Classical and transformer metrics use different test splits unless noted.*
