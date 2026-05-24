# youtube_hate_detector — shared image for FastAPI + Streamlit services
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    NLTK_DATA=/app/nltk_data \
    MODEL_NAME="LR + TF-IDF (local)" \
    ENV=production

WORKDIR /app

# System deps for spaCy / sklearn wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only PyTorch keeps the image smaller; sufficient for the default local LR model
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm

# NLTK corpora used by TextPreprocessor
RUN python - <<'PY'
import nltk
for pkg in ("stopwords", "punkt"):
    nltk.download(pkg, download_dir="/app/nltk_data")
PY

COPY configs/ configs/
COPY src/ src/
COPY models/final_model.joblib models/final_model.joblib

# Default env template (overridden by docker-compose)
COPY env.example .env.example

EXPOSE 8000 8501
