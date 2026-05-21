# Final model comparison (fresh run)

**Generated:** 2026-05-21 (pipeline orchestrator `uv run python -m src.pipeline.run_all_phases`)

**Overfitting rule (AGENTS.md):** `|train − test| < 5` percentage points on **both** test accuracy and F1 (toxic class).

**Canonical JSON:** [`phase5/comparison.json`](phase5/comparison.json) · [`model_selection_overfitting.json`](model_selection_overfitting.json)

---

## Winner (passes gap + highest test accuracy)

| Field | Value |
|--------|--------|
| **Name** | `phase4_random_forest_tuned` |
| **Artifact** | `models/phase4_best.joblib` |
| **Test accuracy** | **72.0%** |
| **Test F1 (toxic)** | **66.3%** |
| **Accuracy gap** | 3.12 pp ✓ |
| **F1 gap** | 4.25 pp ✓ |
| **API default** | Yes (`configs/config.yaml` → `model.path`) |

Phase 3 logistic has higher test scores (73.5% / 70.7% F1) but **fails** the gap rule (15.75 pp / 17.5 pp).

---

## All models (this run)

| Model | Test acc | Test F1 | Acc gap | F1 gap | Pass? |
|--------|----------|---------|---------|--------|-------|
| plan_c_distilbert_v1 | **79.0%** | **79.4%** | 14.62 | 13.74 | **No** |
| phase3_logistic_regression | 73.5% | 70.7% | 15.75 | 17.50 | No |
| plan_c_v2_v2a | 70.5% | 63.4% | 5.00 | 7.04 | No |
| **phase4_random_forest_tuned** | **72.0%** | **66.3%** | 3.12 | 4.25 | **Yes** |
| plan_a_optuna_logistic_plan_a | 65.5% | 60.6% | 3.96 | 4.60 | Yes |
| plan_c_v3 | 62.0% | 64.2% | 7.25 | 5.85 | No |
| plan_c_v2_v2b | 55.0% | 4.3% | 0.12 | 2.98 | Yes* |

\* v2b passes gaps but is not usable (near-random toxic detection).

---

## Compliant models only (ranked by test accuracy)

1. **phase4_random_forest_tuned** — 72.0% acc, 66.3% F1  
2. **plan_a_optuna_logistic_plan_a** — 65.5% acc, 60.6% F1  
3. plan_c_v2_v2b — degenerate (exclude from production)

---

## Highest accuracy overall (does not pass gap)

**plan_c_distilbert_v1** — 79.0% accuracy, 79.4% F1, gaps ~14 pp. Use only if you explicitly waive the Esencial overfitting rule.

---

## Data split note

- **Phase 3 & 4:** `data/processed/youtoxic_phase2.csv` (`split` column from Phase 2, `random_state=42`).  
- **Plan A / C / v2 / v3:** Raw CSV with the same stratified indices (`random_state=42`, 20% test). Features differ (processed text vs raw / transformer).

---

## Regenerate

```bash
uv run python -m src.pipeline.run_all_phases
uv run jupyter nbconvert --execute notebooks/phase4_optimization.ipynb --inplace
```
