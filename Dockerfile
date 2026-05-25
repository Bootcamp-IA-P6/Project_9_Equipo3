# youtube_hate_detector — multi-stage: React + FastAPI (uv)
FROM node:22-bookworm-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci 2>/dev/null || npm install
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm

ARG INSTALL_HF=0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    NLTK_DATA=/app/nltk_data \
    MODEL_NAME="Meta-Feature Stacking (Production)" \
    ENV=production \
    PORT=8000 \
    INSTALL_HF=${INSTALL_HF}

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock* README.md ./
RUN if [ "$INSTALL_HF" = "1" ]; then \
      uv sync --frozen --no-dev --extra hf 2>/dev/null || uv sync --no-dev --extra hf; \
    else \
      uv sync --frozen --no-dev 2>/dev/null || uv sync --no-dev; \
    fi

RUN uv run python -m spacy download en_core_web_sm \
    && uv run python - <<'PY'
import nltk
for pkg in ("stopwords", "punkt"):
    nltk.download(pkg, download_dir="/app/nltk_data")
PY

COPY configs/ configs/
COPY src/ src/
COPY models/baseline/ models/baseline/
COPY models/production_final/ models/production_final/
COPY --from=frontend-build /app/frontend/dist frontend/dist
COPY .env.example .env.example

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --retries=12 --start-period=60s \
  CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Shell form so ${PORT} is expanded at runtime (Render injects PORT; local defaults to 8000).
CMD uv run uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
