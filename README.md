# YouTube Toxic Comment Detector (Essential)

Binary classifier for YouTube comments: **input** `Text`, **target** `IsToxic`.

**Data schema:** drop `CommentId` and `VideoId` only. All `Is*` columns stay in the dataset for EDA and future work; only `IsToxic` is the training target in the Essential phase.

Dataset: [youtoxic_english_1000.csv](https://drive.google.com/file/d/1bG7fA273jIBgJfc6YS1vsKfr1qRiNUTU/view) → place under `data/raw/`.

Requires [uv](https://docs.astral.sh/uv/).

## Setup

```bash
uv sync
```

Install Jupyter for EDA (optional):

```bash
uv sync --group dev
```

## Train

```bash
uv run python -m src.pipeline.train
```

Writes `models/hate_detector.joblib` and `reports/metrics.json`.

**Latest run:** test F1 ≈ 0.71, **F1 gap ≈ 3.1%** (`gap_ok: true`). See `reports/metrics.json` for full train/test metrics.

## API

```bash
uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

- `GET /` — service info
- `GET /docs` — interactive Swagger UI
- `GET /health`
- `POST /predict` — body: `{"text": "your comment"}`

Example:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "great video, thanks for sharing"}'
```

## EDA

```bash
uv run jupyter notebook notebooks/01_eda.ipynb
```

## Project layout

```
├── pyproject.toml      # dependencies (uv)
├── uv.lock             # locked versions
├── configs/default.yaml
├── data/raw/
├── models/
├── notebooks/
├── reports/
└── src/
    ├── api/           # FastAPI
    ├── data/          # CSV loading
    ├── evaluation/    # Metrics & gap check
    ├── features/      # Preprocessing (regex, lemmatization)
    └── pipeline/      # Training script
```

## Pipeline

1. Regex cleanup (URLs, HTML, mentions)
2. Tokenization, English stopwords, WordNet lemmatization
3. TF-IDF (1–2 grams)
4. Logistic Regression (`class_weight=balanced`)
