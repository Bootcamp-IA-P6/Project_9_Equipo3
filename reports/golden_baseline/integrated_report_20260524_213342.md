# Golden Baseline Strategy — 20260524_213342

Two-step briefing alignment: **Esencial** frozen expert baseline, then **Experto** squeeze + hybrid.

## Step 1 — Golden Baseline (Esencial)

| Metric | Value | Target |
|--------|-------|--------|
| F1 weighted (test) | **0.7903** | ~0.72 (pretrained expert) |
| Train–test gap (pp) | **0.16** | < 1.0% ✅ |
| Fine-tuning | None (all layers frozen) | — |
| Threshold | 0.12 | val-tuned |

## Step 2 — Performance Squeeze (Experto)

| Metric | Value | Target |
|--------|-------|--------|
| F1 weighted (test) | **0.7588** | ≥ 0.8 |
| Train–test gap (pp) | **2.83** | ≤ 4.9% |
| R-Drop | True | enabled |
| Layers trained | last partial_last_2 | 2 + head |

## Step 3 — Hybrid Safety Net (Final)

| Metric | Value | Target |
|--------|-------|--------|
| F1 weighted (test) | **0.7479** | ≥ 0.8 ⚠️ |
| Train–test gap (pp) | **4.39** | < 5.0% ✅ |
| Weights | BERT 0.9 / LR 0.1 | anchor |
| LR regularization | C=0.001, max_features=200 | stability |

### Overall: ⚠️ Review gaps / F1

- JSON: `reports/golden_baseline/golden_baseline_run_20260524_213342.json`
