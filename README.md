# YouTube Toxic Comment Detector (youtube_hate_detector)

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-UI-61DAFB.svg)](https://react.dev/)
[![Docker](https://img.shields.io/badge/docker-compose-2496ED.svg)](https://docs.docker.com/compose/)

**Español:** [README.es.md](README.es.md)

Automated **Safe vs Toxic** classification for YouTube-style comments. Production stack: **FastAPI** (REST) + **React** (YouTube Watch UI). Default model: **Logistic Regression + TF-IDF** (`models/final_model.joblib`).

---

## Clone and layout

```bash
git clone <your-repo-url>
cd youtube_hate_detector   # use this folder name locally (team convention)
```

```
youtube_hate_detector/
├── configs/              # pipeline, features, model_catalog, suggested_videos
├── frontend/             # React SPA (Vite)
├── models/               # final_model.joblib, experiments/
├── src/
│   ├── api/              # FastAPI routes
│   └── service/          # ModelService (inference)
├── pyproject.toml        # uv dependencies
├── uv.lock
└── docker-compose.yml
```

---

## How to use FastAPI

The API loads `ModelService` once at startup and serves JSON only (the React app is the UI).

```bash
cp .env.example .env
uv sync                    # baseline (LR model only)
uv sync --extra hf         # required for DistilBERT / toxic-bert / Fine-tuned HF models
uv run uvicorn src.api.main:app --reload --port 8000
```

Verify HF deps: `uv run python -c "import transformers; print('ok')"`.

| Resource | URL |
|----------|-----|
| Swagger | http://localhost:8000/docs |
| Health | http://localhost:8000/health |

**Main endpoints**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict` | Score one comment `{ "text", "threshold" }` |
| `POST` | `/predict-video` | Fetch YouTube comments + score `{ "url", "max_comments", "threshold" }` |
| `GET` | `/videos/suggested` | Metadata for right-rail videos (from `configs/suggested_videos.yaml`) |
| `GET` | `/models` | Available models |
| `GET` | `/models/status` | Per-model availability (HF deps, local weights) |
| `PUT` | `/model/{name}` | Switch active model (warmup-validated) |

Set `YOUTUBE_API_KEY` in `.env` for real comments and suggested-video thumbnails.

**Change models without UI changes:** edit [`configs/model_catalog.yaml`](configs/model_catalog.yaml), then restart the API or use Settings in the app.

---

## React UI (local dev)

```bash
# Terminal 1 — API
uv run uvicorn src.api.main:app --reload --port 8000

# Terminal 2 — frontend (proxies API)
cd frontend && npm install && npm run dev
```

Open http://localhost:5173 — Watch page with staged demo player, real suggested videos (click to load comments), English UI.

---

## Docker

```bash
export YOUTUBE_API_KEY=your_key   # optional but recommended
docker compose up --build         # LR model only (default)

# Hugging Face models (transformers + torch; larger image):
INSTALL_HF=1 docker compose build --build-arg INSTALL_HF=1
INSTALL_HF=1 docker compose up
```

| URL | Service |
|-----|---------|
| http://localhost:8000 | API + built React SPA |
| http://localhost:8000/docs | Swagger |

Container: `youtube_hate_detector-app`.

---

## Training (unchanged)

```bash
uv run python -m src.pipeline.run_pipeline --model lr
```

See [docs/PIPELINE.md](docs/PIPELINE.md).

---

## Configuration

| File | Purpose |
|------|---------|
| `.env` | Secrets (`YOUTUBE_API_KEY`, `MODEL_NAME`) |
| `configs/model_catalog.yaml` | Inference models for API/UI |
| `configs/suggested_videos.yaml` | YouTube IDs for the suggested rail |
| `configs/pipeline.yaml` | Training data paths |

---

## Tests

```bash
uv sync --extra dev --extra hf
uv run pytest
```

---

## Briefing vs team stack

| Topic | Briefing | This repo |
|-------|----------|-----------|
| UI | Streamlit | **React** |
| API | FastAPI | **FastAPI** |
| Package manager | varies | **`uv`** |

Legacy Streamlit (`src/app/`) has been removed.
