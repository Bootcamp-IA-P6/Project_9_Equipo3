# Production — Meta-Feature Stacking

| File | Description |
|------|-------------|
| `meta_stack_final.joblib` | Scaler + meta-learner bundle |
| `manifest.json` | Metrics from Notebook 14 |

Default model in API, UI, and Docker. Regenerate:

```bash
uv run python -m src.experiments.notebook_14_final_stack
```
