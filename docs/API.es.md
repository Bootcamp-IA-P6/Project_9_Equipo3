# Referencia API (FastAPI)

URL base (local): `http://localhost:8000`  
Documentación interactiva: `/docs`, `/redoc`

Implementación: [`src/api/main.py`](../src/api/main.py)

---

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/` | Estado del servicio y modelo activo |
| `GET` | `/model-info` | Metadatos del modelo cargado |
| `GET` | `/models` | Modelos disponibles y activo |
| `PUT` | `/model/{model_name}` | Cambiar modelo activo |
| `POST` | `/predict` | Clasificar un comentario |
| `POST` | `/predict-batch` | Hasta 100 comentarios |
| `POST` | `/predict-video` | Comentarios de un vídeo de YouTube |

---

## `POST /predict`

**Cuerpo**

```json
{
  "text": "Texto del comentario",
  "threshold": 0.5
}
```

**Respuesta**

```json
{
  "text": "Texto del comentario",
  "is_toxic": false,
  "probability": 0.08,
  "labels": [],
  "model_used": "Meta-Feature Stacking (Production)",
  "latency_ms": 15.2
}
```

- `is_toxic`: `true` = **Tóxico**, `false` = **Seguro**
- `probability`: probabilidad de clase tóxica (0–1)

**curl**

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "¡Gran vídeo, gracias!", "threshold": 0.5}'
```

---

## `POST /predict-batch`

```bash
curl -s -X POST http://localhost:8000/predict-batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Comentario seguro", "Eres un idiota"], "threshold": 0.5}'
```

---

## `POST /predict-video`

Requiere `YOUTUBE_API_KEY` en `.env` para comentarios reales.

```json
{
  "url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "max_comments": 50,
  "threshold": 0.5
}
```

---

## Modelos del demo

[`configs/model_catalog.yaml`](../configs/model_catalog.yaml) · métricas baselines: [`models/baseline/manifest.json`](../models/baseline/manifest.json)

| Nombre | Artefacto / pesos |
|--------|-------------------|
| `Meta-Feature Stacking (Production)` | `models/production_final/meta_stack_final.joblib` |
| `LR + TF-IDF (Baseline)` | `models/baseline/lr_tfidf.joblib` |
| `Frozen Toxic-BERT (Baseline)` | Hugging Face `unitary/toxic-bert` |

```bash
curl -s -X POST http://localhost:8000/models/select \
  -H "Content-Type: application/json" \
  -d '{"model_name": "LR + TF-IDF (Baseline)"}'
```

---

## Variables de entorno

| Variable | Descripción |
|----------|-------------|
| `MODEL_NAME` | Por defecto: Meta-Feature Stacking (Production) |
| `YOUTUBE_API_KEY` | API de YouTube para `/predict-video` |

Ver [`.env.example`](../.env.example).
