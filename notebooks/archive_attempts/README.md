# Archive — experimental notebooks

Notebooks **04–11** and **13** document iterative experiments (ensembles, tuning, augmentation, stable production runs, expert Toxic-BERT, hyper-optimization sprints). They are kept for reproducibility but are **not** part of the primary project narrative.

**Primary storyline** (parent `notebooks/` folder):

| Notebook | Focus |
|----------|--------|
| `01_eda_v2` | Data audit, Safe vs Toxic |
| `02_preprocessing_v2` | Cleaning pipeline |
| `03_vectorization_v2` | TF-IDF features |
| `12_golden_baseline_strategy` | Frozen BERT + golden baseline metrics |
| `14_final_meta_stacking` | **Production** hybrid meta-feature stacking |

Re-run production artifacts:

```bash
uv run python -m src.experiments.notebook_14_final_stack
```
