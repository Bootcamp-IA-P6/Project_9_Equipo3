# YouTube Toxic Comment Detector (SignalMod)

Binary **Safe vs Toxic** comment moderation assistant with a **FastAPI** backend and a **Streamlit** UI.

## Quick start (Docker)

No manual setup beyond Docker. The image bundles the default model (`models/final_model.joblib`), configs, and NLP assets (spaCy + NLTK).

```bash
docker compose up --build
```

| Service   | URL |
|-----------|-----|
| Streamlit UI | http://localhost:8501 |
| FastAPI      | http://localhost:8000 |
| API docs     | http://localhost:8000/docs |

Optional: set `YOUTUBE_API_KEY` for live comment scraping on `/predict-video`:

```bash
export YOUTUBE_API_KEY=your_key_here
docker compose up --build
```

Stop containers:

```bash
docker compose down
```

Docker image and containers use the project name `youtube_hate_detector` (e.g. `youtube_hate_detector-api`). If you previously built `ai-nlp-app:latest`, remove it once: `docker rmi ai-nlp-app:latest`.

## Architecture

```
youtube_hate_detector/
├── configs/           # YAML hyperparameters (non-secret)
├── data/
│   ├── raw/           # Original dataset (gitignored)
│   └── processed/
├── models/            # Serialized models (e.g. final_model.joblib)
├── src/
│   ├── api/           # FastAPI (REST)
│   ├── app/           # Streamlit UI
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── pipeline/
│   ├── service/       # ModelService (inference)
│   └── utils/
├── tests/
├── Dockerfile
└── docker-compose.yml
```

## Local development (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Terminal 1 — API
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Streamlit
streamlit run src/app/app.py --server.port 8501
```

Copy `env.example` to `.env` if you need a YouTube API key or custom `MODEL_NAME`.

## Tests

```bash
pytest tests/ -v
```
