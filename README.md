# Project_9_Equipo3 — YouTube Toxic Comment Detector

Automated detection of **toxic** YouTube comments to support moderation. Repository folder: **`ai-nlp`**.

Full agent and execution rules: **[AGENTS.md](AGENTS.md)**.

---

## Python environment (`uv`)

From the repository root **`.`**:

```bash
uv sync
uv run jupyter notebook notebooks/phase1_data_collection_eda.ipynb
uv run pytest
```

Add dependencies with `uv add <package>` (dev: `uv add --dev <package>`). Do not use `pip install` for project deps. Secrets go in **`.env`** (see `.env.example`).

---

## Team compromise vs. briefing

**Team decisions in [AGENTS.md](AGENTS.md) take priority** over the Factoría F5 briefing when they differ.

| Topic | Briefing (typical) | This repo |
|-------|-------------------|-----------|
| Frontend | Streamlit | **React** (`frontend/`) |
| Backend | FastAPI | **FastAPI** (`src/api/`) |
| Secrets | Config files | **`.env`** (see `.env.example`) |
| Labels (UI) | Mixed wording | **Safe** / **Toxic** |
| Default model target | — | **`IsToxic`** (binary); multilabel optional after EDA |
| Python tooling | pip / conda | **`uv`** (`pyproject.toml`, `uv.lock`) |

---

## Architecture

Paths are relative to the repository root **`.`** (clone directory: **`ai-nlp`**).

```text
.
├── data/
│   ├── raw/                # Original dataset (not committed)
│   └── processed/          # Preprocessed data
├── frontend/               # React SPA (YouTube Watch Page UI)
├── notebooks/              # EDA and experiments
├── models/                 # Trained models (.joblib, etc.)
├── reports/                # Metrics per experiment
├── src/
│   ├── data/               # Load and scrape
│   ├── features/           # Preprocessing and vectorization
│   ├── models/             # Baseline, ensemble, deep learning
│   ├── evaluation/         # Metrics and error analysis
│   ├── pipeline/           # End-to-end pipelines
│   ├── api/                # FastAPI
│   └── utils/              # Logger, config loader
├── configs/                # Non-secret YAML config
├── tests/                  # Unit tests
└── logs/                   # Execution logs
```

**Stack:** FastAPI (API + model) + React (UI). React calls `/predict` and `/scrape` over HTTP.

---

## Labels

- **Default:** **Safe** vs **Toxic** using dataset column **`IsToxic`**.
- **Multilabel:** Review in Phase 1 EDA; extend only if each kept label has enough training positives and overfitting stays under control (see AGENTS.md).

---

## Configuration

| Kind | Where | Examples |
|------|--------|----------|
| Secrets | `.env` (git-ignored) | `YOUTUBE_API_KEY`, `DATABASE_URL` |
| Training / paths | `configs/config.yaml` | `MODEL_PATH`, `TRAIN_TEST_SIZE` |

Copy `.env.example` to `.env` before running locally.

---

## Docker and frontend build (required for deploy)

Do not deploy API-only without building the UI:

1. `cd frontend && npm ci && npm run build` → `frontend/dist/`
2. FastAPI serves `dist/` via static files; API routes unchanged.
3. **Docker:** multi-stage image — Node stage (build React) → Python stage (FastAPI + copied `dist/`).

Details and checklist: **[AGENTS.md — Docker & frontend build](AGENTS.md#docker--frontend-build-do-not-omit)**.

---

## Considerations

See **[AGENTS.md § Considerations](AGENTS.md#8-considerations)** for deployment pitfalls, multilabel UI changes, sparse labels on small data, and migration away from legacy Streamlit paths.

---

## Phase 3 — Run API + React UI

```bash
uv sync
uv run python -m src.pipeline.phase3_train_baseline
uv run uvicorn src.api.main:app --reload --port 8000
# other terminal:
cd frontend && npm install && npm run dev
```

Open http://localhost:5173 — comments call `POST /predict` via Vite proxy.

## Phase 5 — Plan A & C (higher accuracy attempts)

```bash
uv run python -m src.pipeline.plan_a_train      # TF-IDF + advanced features + MLflow
uv run python -m src.pipeline.plan_c_train      # DistilBERT v1 + MLflow (~2 min on CPU)
uv run python -m src.pipeline.plan_c_train_v2 --variant all  # regularized v2 matrix
uv run python -m src.pipeline.plan_c_train_v3              # v2a fine-tune: class weights + F1 stop
uv run python -m src.pipeline.run_all_phases               # rerun phases 1–5 + build comparison JSON
uv run mlflow ui --backend-store-uri mlruns      # compare runs (tag: status PASSED/FAILED)

**Final comparison (after full rerun):** [`reports/FINAL_MODEL_COMPARISON.md`](reports/FINAL_MODEL_COMPARISON.md) · [`reports/phase5/comparison.json`](reports/phase5/comparison.json) · [`reports/model_selection_overfitting.json`](reports/model_selection_overfitting.json)
```

| Model | Test acc | Test F1 (toxic) | Passes 5% gap? | MLflow run |
|-------|----------|-----------------|----------------|------------|
| Phase 4 RF | 73.5% | 68.6% | Yes | — |
| Plan A (hybrid+aug) | 67.5% | 61.5% | No | `plan_a_hybrid` |
| Plan C DistilBERT v1 | **78%** | **78.6%** | No | `plan_c_distilbert` |
| Plan C v2a (freeze 4) | 73.0% | 65.4% | No (F1 gap 5.31pp) | `plan_c_v2_v2a` |
| Plan C v2b (head-only) | 55.0% | 4.3% | Yes (degenerate) | `plan_c_v2_v2b` |
| **Plan C v3** (v2a + class weights) | 59.5% | 66.1% | **Yes** (F1 gap 2.89pp) | `plan_c_v3` |

Notebook: [`notebooks/plan_c_distilbert_v2.ipynb`](notebooks/plan_c_distilbert_v2.ipynb)

**MLflow:** No older notebook used MLflow; tracking starts in Phase 5 pipelines.  
**API default:** still `models/phase4_best.joblib` (balanced gap + F1 goal not met by Plan C v2).

## Phase 4 — Tuning & tests

```bash
uv run python -m src.pipeline.phase4_optimize
uv run pytest tests/
```

Production model: `models/phase4_best.joblib` (tuned Random Forest; passes 5% overfitting gap).  
YouTube scrape: set `YOUTUBE_API_KEY` in `.env`, then `POST /scrape` with video URL in `text`.

---

## Quick links

- [AGENTS.md](AGENTS.md) — roadmap, UI spec, quality standards
- [Briefing PDF](docs/project-ai-nlp.pdf) — official requirements (team compromise overrides where documented)
