# API reference (FastAPI)

Base URL (local): `http://localhost:8000`  
Interactive docs: `/docs` (Swagger), `/redoc` (ReDoc)

Implementation: [`src/api/main.py`](../src/api/main.py)  
Inference: [`src/service/model_service.py`](../src/service/model_service.py)

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check and active model name |
| `GET` | `/model-info` | Metadata for the loaded model |
| `GET` | `/models` | List available models and active one |
| `PUT` | `/model/{model_name}` | Switch active model (lazy load on next predict) |
| `POST` | `/predict` | Classify one comment |
| `POST` | `/predict-batch` | Classify up to 100 comments |
| `POST` | `/predict-video` | Fetch YouTube comments and classify (needs API key or demo fallback) |

---

## `POST /predict`

**Request body**

```json
{
  "text": "Comment text here",
  "threshold": 0.5
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | yes | 1–5000 characters, non-empty after trim |
| `threshold` | float | no | Toxic if `probability >= threshold` (**0.381** production, **0.5** LR baseline, **0.12** frozen BERT baseline) |

**Response**

```json
{
  "text": "Comment text here",
  "is_toxic": false,
  "probability": 0.0821,
  "labels": [],
  "model_used": "Meta-Feature Stacking (Production)",
  "latency_ms": 15.2
}
```

| Field | Description |
|-------|-------------|
| `is_toxic` | `true` = **Toxic**, `false` = **Safe** |
| `probability` | P(toxic), 0.0–1.0 |
| `labels` | Optional category hints when toxic (keyword/heuristic or HF labels) |
| `model_used` | Active model id from `ModelService` |

**curl**

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Thanks for the tutorial!", "threshold": 0.5}'
```

**Toxic example**

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "You are worthless garbage", "threshold": 0.5}'
```

---

## `POST /predict-batch`

```json
{
  "texts": ["Safe comment", "Another line"],
  "threshold": 0.5
}
```

Response includes `results` (list of predict objects), `total`, `toxic_count`, `latency_ms`.

```bash
curl -s -X POST http://localhost:8000/predict-batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Nice video", "I hate you"], "threshold": 0.5}'
```

---

## `POST /predict-video`

```json
{
  "url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "max_comments": 50,
  "threshold": 0.5
}
```

Set `YOUTUBE_API_KEY` in `.env` for live comment fetch. Without a key, the API may use a limited fallback scraper or demo data (see implementation in `main.py`).

---

## `GET /models` and model switch

Demo models from [`configs/model_catalog.yaml`](../configs/model_catalog.yaml):

| Name | Type | Artifact / weights |
|------|------|-------------------|
| `Meta-Feature Stacking (Production)` | meta_stack | `models/production_final/meta_stack_final.joblib` |
| `LR + TF-IDF (Baseline)` | local | `models/baseline/lr_tfidf.joblib` |
| `Frozen Toxic-BERT (Baseline)` | hf_remote | Hugging Face `unitary/toxic-bert` |

```bash
curl -s http://localhost:8000/models/status

curl -s -X POST http://localhost:8000/models/select \
  -H "Content-Type: application/json" \
  -d '{"model_name": "LR + TF-IDF (Baseline)"}'
```

Default at startup: `Meta-Feature Stacking (Production)` (`MODEL_NAME` in `.env`).

---

## Environment variables

| Variable | Used by | Description |
|----------|---------|-------------|
| `MODEL_NAME` | API startup | Initial model from `AVAILABLE_MODELS` |
| `YOUTUBE_API_KEY` | `/predict-video` | YouTube Data API v3 |
| `ENV` | logging / behavior | `development` or `production` |

Copy from [`.env.example`](../.env.example).

---

## Errors

| Status | When |
|--------|------|
| `422` | Invalid body (e.g. empty `text`) |
| `503` | Model not loaded yet |
| `500` | Prediction failure |
