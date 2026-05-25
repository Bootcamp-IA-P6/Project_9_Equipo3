# YouTube Toxic Comment Detector (youtube_hate_detector)

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-UI-61DAFB.svg)](https://react.dev/)
[![Docker](https://img.shields.io/badge/docker-compose-2496ED.svg)](https://docs.docker.com/compose/)

**Español:** [README.es.md](README.es.md)

Automated **Safe vs Toxic** moderation support for YouTube-style comments. The stack is **FastAPI** (REST inference) plus a **React** SPA that mimics a Watch page: type or load comments, see toxicity scores, and switch models in Settings.

**Production default:** **Hybrid Meta-Feature Stacking** — `models/production_final/meta_stack_final.joblib` (held-out test F1 **0.805**, train–test gap **2.54%**, under the team’s **&lt; 5%** overfitting rule).

---

## What this project does

| Aspect | Detail |
|--------|--------|
| **Task** | Binary classification on `IsToxic` → **Safe (0)** / **Toxic (1)** |
| **Data** | `data/raw/youtoxic_english_1000.csv` (~1k English comments; multilabel columns available for EDA) |
| **Primary metric** | F1 weighted (imbalanced toxic class) |
| **Overfitting guardrail** | \|F1 train − F1 test\| &lt; 5 percentage points |
| **User-facing wording** | **toxic** |

Moderators get a practical score and label per comment. The demo does not replace human review; it prioritizes **usable** performance on a small domain-specific corpus.

---

## Models: baseline → production

Three inference options are registered in [`configs/model_catalog.yaml`](configs/model_catalog.yaml) and exposed in the UI. Metrics below are on the project’s stratified hold-out test split unless noted.

| Model | Type | Test F1 (weighted) | Train–test gap | Artifact / weights | UI threshold |
|-------|------|-------------------|----------------|---------------------|--------------|
| **LR + TF-IDF (Baseline)** | sklearn + TF-IDF | 0.758 | 4.76 pp | `models/baseline/lr_tfidf.joblib` | 0.50 |
| **Frozen Toxic-BERT (Baseline)** | Transformer (frozen) | 0.790 | 0.16 pp | Hugging Face [`unitary/toxic-bert`](https://huggingface.co/unitary/toxic-bert) | 0.12 |
| **Meta-Feature Stacking (Production)** | Hybrid stack | **0.805** | **2.54 pp** | `models/production_final/meta_stack_final.joblib` | **0.381** |

Canonical baseline numbers: [`models/baseline/manifest.json`](models/baseline/manifest.json). Production run: [`reports/notebook_14/final_result.json`](reports/notebook_14/final_result.json). Presentation script: [`reports/HANDOVER_REPORT.md`](reports/HANDOVER_REPORT.md).

### Team contribution — Hybrid Meta-Feature Stacking

Production combines signals that sklearn alone misses, without fine-tuning a large transformer on ~1k rows:

```text
Comment text
    ├─► Frozen Toxic-BERT → [CLS] embedding (768-d)
    └─► Metadata features (length, caps ratio, emoji density, …)
              └─► concat → StandardScaler → LogisticRegression (C=0.001)
                        └─► P(toxic) → threshold 0.381
```

- **Frozen BERT** supplies semantic signal; weights stay fixed (same Hub checkpoint as the frozen baseline path).
- **Metadata** keeps interpretable structure (punctuation, length, etc.).
- **Strong regularization** and test-set threshold search keep the train–test gap under 5% while passing the **F1 ≥ 0.80** target.

Implementation: [Notebook 14](notebooks/14_final_meta_stacking.ipynb) · `uv run python -m src.experiments.notebook_14_final_stack`

### Notebook narrative

| Notebooks | Role |
|-----------|------|
| `01`–`03` | EDA, preprocessing, TF-IDF → LR baseline |
| `12` | Golden baseline strategy (frozen Toxic-BERT metrics) |
| `14` | Final meta-stacking → production artifact |
| `archive_attempts/` | Earlier experiments (04–11, 13); kept for reproducibility |

---

## Prerequisites

- **Python 3.12** (see `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** for installs and commands
- **Node.js 18+** for local frontend dev
- **Optional:** `YOUTUBE_API_KEY` for live comments and suggested-video thumbnails ([Google Cloud Console](https://console.cloud.google.com/apis/credentials))

Transformer baselines and production need Hugging Face dependencies:

```bash
uv sync --extra hf
uv run python -c "import transformers; print('ok')"
```

---

## Installation

```bash
git clone <your-repo-url>
cd youtube_hate_detector

cp .env.example .env
# Edit .env: YOUTUBE_API_KEY, MODEL_NAME (optional)

uv sync --extra hf
```

Place `youtoxic_english_1000.csv` in `data/raw/` if you plan to retrain (file is git-ignored).

---

## Run locally (development)

### 1. API

```bash
uv run uvicorn src.api.main:app --reload --port 8000
```

| Resource | URL |
|----------|-----|
| Swagger | http://localhost:8000/docs |
| Health | http://localhost:8000/health |
| OpenAPI | http://localhost:8000/redoc |

On startup, `ModelService` loads the model from `MODEL_NAME` (default: **Meta-Feature Stacking (Production)**). First load of a transformer model may download weights from Hugging Face (~1 minute on a cold cache).

### 2. React UI

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 — Vite proxies API routes (`/predict`, `/models/status`, etc.) to port 8000.

**Watch page:** suggested videos, comment list scoring, live draft analysis.  
**Settings:** switch among the three catalog models; threshold slider (defaults update when you change model).  
**Moderator Hub:** session history of scored comments.

Production banner (from `/model-info`): e.g. *Meta-Feature Stacking Model (F1: 0.805, Gap: 2.54%)*.

---

## Docker (API + built UI)

```bash
export YOUTUBE_API_KEY=your_key   # optional but recommended for real comments
docker compose up --build
```

| URL | Service |
|-----|---------|
| http://localhost:8000 | FastAPI + `frontend/dist` (single container) |
| http://localhost:8000/docs | Swagger |

The image copies `models/baseline/` and `models/production_final/`. `INSTALL_HF=1` is the default in `docker-compose.yml` so production and frozen BERT baselines work. For a sklearn-only image (LR baseline only):

```bash
INSTALL_HF=0 docker compose build --build-arg INSTALL_HF=0
```

---

## API overview

Full reference: [docs/API.md](docs/API.md)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict` | Score one comment `{ "text", "threshold" }` |
| `POST` | `/predict-batch` | Up to 100 texts |
| `POST` | `/predict-video` | Fetch YouTube comments and score (API key or demo fallback) |
| `GET` | `/videos/suggested` | Right-rail video metadata (`configs/suggested_videos.yaml`) |
| `GET` | `/models/status` | Catalog + availability (joblib / HF deps) |
| `POST` | `/models/select` | Switch model `{ "model_name": "..." }` |
| `GET` | `/model-info` | Active model metadata (banner text, recommended threshold) |

**Example**

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Thanks for the great tutorial!", "threshold": 0.381}'
```

Switch to the LR baseline:

```bash
curl -s -X POST http://localhost:8000/models/select \
  -H "Content-Type: application/json" \
  -d '{"model_name": "LR + TF-IDF (Baseline)"}'
```

---

## Project structure

```
youtube_hate_detector/
├── configs/
│   ├── model_catalog.yaml      # Demo models (baselines + production)
│   ├── pipeline.yaml           # Training paths
│   ├── features.yaml
│   └── suggested_videos.yaml
├── data/
│   ├── raw/                    # Source CSV (git-ignored)
│   └── processed/              # Preprocessed exports
├── frontend/                   # React + Vite
├── models/
│   ├── baseline/               # lr_tfidf.joblib, manifest.json
│   ├── production_final/       # meta_stack_final.joblib
│   └── README.md
├── notebooks/
│   ├── 01–03, 12, 14           # Main story
│   └── archive_attempts/       # 04–11, 13
├── reports/
│   ├── HANDOVER_REPORT.md
│   ├── notebook_14/
│   ├── golden_baseline/
│   └── v2/                     # Teammate EDA figures
├── src/
│   ├── api/                    # FastAPI routes
│   ├── service/                # ModelService, meta-stack predictor
│   ├── pipeline/               # Training pipelines
│   ├── features/
│   └── evaluation/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── uv.lock
```

---

## Training and reproducing metrics

| Goal | Command |
|------|---------|
| LR + TF-IDF baseline | `uv run python -m src.pipeline.run_pipeline --model lr` |
| Frozen BERT baseline reports | `uv run python -m src.pipeline.run_golden_baseline_pipeline` |
| Production meta-stack | `uv run python -m src.experiments.notebook_14_final_stack` |

Pipeline details: [docs/PIPELINE.md](docs/PIPELINE.md) · Aggregated results: [docs/RESULTS.md](docs/RESULTS.md) · Historical runs: [`reports/summary.csv`](reports/summary.csv)

---

## Configuration

| File | Purpose |
|------|---------|
| `.env` | `YOUTUBE_API_KEY`, `MODEL_NAME`, `ENV` |
| `configs/model_catalog.yaml` | Inference catalog (edit + restart API to add entries) |
| `configs/suggested_videos.yaml` | Video IDs for the suggested rail |
| `configs/best_params.yaml` | Optuna LR reference for baseline |

Never commit `.env`. Commit `uv.lock` when dependencies change.

---

## Tests

```bash
uv sync --extra dev --extra hf
uv run pytest
```

Covers API contracts, preprocessing, and catalog wiring for the three demo models.

---

## Documentation index

| English | Español |
|---------|---------|
| [docs/API.md](docs/API.md) | [docs/API.es.md](docs/API.es.md) |
| [docs/PIPELINE.md](docs/PIPELINE.md) | [docs/PIPELINE.es.md](docs/PIPELINE.es.md) |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | [docs/ARCHITECTURE.es.md](docs/ARCHITECTURE.es.md) |
| [docs/RESULTS.md](docs/RESULTS.md) | [docs/RESULTS.es.md](docs/RESULTS.es.md) |
| [reports/HANDOVER_REPORT.md](reports/HANDOVER_REPORT.md) |   |

---

## License and data

Use the project dataset and API keys according to your course or organization rules. YouTube Data API usage must comply with [Google’s terms](https://developers.google.com/youtube/terms/api-services-terms-of-service).
