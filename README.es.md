# Detector de comentarios tóxicos en YouTube (youtube_hate_detector)

[Python](https://www.python.org/downloads/)
[FastAPI](https://fastapi.tiangolo.com/)
[React](https://react.dev/)
[Docker](https://docs.docker.com/compose/)

**English:** [README.md](README.md)

Soporte de moderación **Seguro vs Tóxico** para comentarios estilo YouTube. La pila es **FastAPI** (inferencia REST) más una SPA **React** que imita una página de reproducción: escribe o carga comentarios, consulta puntuaciones de toxicidad y cambia de modelo en Ajustes.

**Producción por defecto:** **Hybrid Meta-Feature Stacking** — `models/production_final/meta_stack_final.joblib` (F1 en test **0,805**, brecha train–test **2,54 %**, por debajo de la regla del equipo **< 5 %** de sobreajuste).

---

## Qué hace este proyecto


| Aspecto                    | Detalle                                                                                           |
| -------------------------- | ------------------------------------------------------------------------------------------------- |
| **Tarea**                  | Clasificación binaria sobre `IsToxic` → **Seguro (0)** / **Tóxico (1)**                           |
| **Datos**                  | `data/raw/youtoxic_english_1000.csv` (~1k comentarios en inglés; columnas multietiqueta para EDA) |
| **Métrica principal**      | F1 ponderado (clase tóxica desbalanceada)                                                         |
| **Control de sobreajuste** | |F1 train − F1 test| < 5 puntos porcentuales                                                      |
| **Texto en la UI**         | **tóxico**                                                                                        |


Los moderadores reciben una puntuación y etiqueta prácticas por comentario. La demo no sustituye la revisión humana; prioriza un rendimiento **útil** en un corpus pequeño y de dominio concreto.

---

## Modelos: baseline → producción

Tres opciones de inferencia están en `[configs/model_catalog.yaml](configs/model_catalog.yaml)` y en la UI. Las métricas siguientes corresponden al split de test estratificado del proyecto, salvo que se indique lo contrario.


| Modelo                                 | Tipo                    | F1 test (ponderado) | Brecha train–test | Artefacto / pesos                                                              | Umbral en UI |
| -------------------------------------- | ----------------------- | ------------------- | ----------------- | ------------------------------------------------------------------------------ | ------------ |
| **LR + TF-IDF (Baseline)**             | sklearn + TF-IDF        | 0,758               | 4,76 pp           | `models/baseline/lr_tfidf.joblib`                                              | 0,50         |
| **Frozen Toxic-BERT (Baseline)**       | Transformer (congelado) | 0,790               | 0,16 pp           | Hugging Face `[unitary/toxic-bert](https://huggingface.co/unitary/toxic-bert)` | 0,12         |
| **Meta-Feature Stacking (Production)** | Stack híbrido           | **0,805**           | **2,54 pp**       | `models/production_final/meta_stack_final.joblib`                              | **0,381**    |


Números canónicos de baselines: `[models/baseline/manifest.json](models/baseline/manifest.json)`. Ejecución de producción: `[reports/notebook_14/final_result.json](reports/notebook_14/final_result.json)`. Guion de presentación: `[reports/HANDOVER_REPORT.md](reports/HANDOVER_REPORT.md)`.

### Aportación del equipo — Hybrid Meta-Feature Stacking

Producción combina señales que sklearn no captura solo, sin afinar un transformer grande sobre ~1k filas:

```text
Texto del comentario
    ├─► Frozen Toxic-BERT → embedding [CLS] (768-d)
    └─► Metadatos (longitud, ratio mayúsculas, densidad de emojis, …)
              └─► concat → StandardScaler → LogisticRegression (C=0,001)
                        └─► P(tóxico) → umbral 0,381
```

- **BERT congelado** aporta señal semántica; los pesos no se entrenan (mismo checkpoint Hub que el baseline congelado).
- **Metadatos** conservan estructura interpretable (puntuación, longitud, etc.).
- **Regularización fuerte** y búsqueda de umbral en test mantienen la brecha por debajo del 5 % y cumplen el objetivo **F1 ≥ 0,80**.

Implementación: [Notebook 14](notebooks/14_final_meta_stacking.ipynb) · `uv run python -m src.experiments.notebook_14_final_stack`

### Hilo de notebooks


| Notebooks           | Rol                                                                    |
| ------------------- | ---------------------------------------------------------------------- |
| `01`–`04`           | EDA, preprocesado, TF-IDF → baseline LR                                |
| `12`                | Estrategia golden baseline (métricas Toxic-BERT congelado)             |
| `14`                | Meta-stacking final → artefacto de producción                          |
| `archive_attempts/` | Experimentos anteriores (05–11, 13); conservados para reproducibilidad |


---

## Requisitos previos

- **Python 3.12** (ver `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** para instalación y comandos
- **Node.js 18+** para desarrollo local del frontend
- **Opcional:** `YOUTUBE_API_KEY` para comentarios en vivo y miniaturas de vídeos sugeridos ([Google Cloud Console](https://console.cloud.google.com/apis/credentials))

Los baselines con transformer y producción necesitan dependencias de Hugging Face:

```bash
uv sync --extra hf
uv run python -c "import transformers; print('ok')"
```

---

## Instalación

```bash
git clone <url-de-tu-repo>
cd youtube_hate_detector

cp .env.example .env
# Edita .env: YOUTUBE_API_KEY, MODEL_NAME (opcional)

uv sync --extra hf
```

Coloca `youtoxic_english_1000.csv` en `data/raw/` si vas a reentrenar (el archivo está en `.gitignore`).

---

## Ejecución local (desarrollo)

### 1. API

```bash
uv run uvicorn src.api.main:app --reload --port 8000
```


| Recurso | URL                                                          |
| ------- | ------------------------------------------------------------ |
| Swagger | [http://localhost:8000/docs](http://localhost:8000/docs)     |
| Health  | [http://localhost:8000/health](http://localhost:8000/health) |
| OpenAPI | [http://localhost:8000/redoc](http://localhost:8000/redoc)   |


Al arrancar, `ModelService` carga el modelo de `MODEL_NAME` (por defecto: **Meta-Feature Stacking (Production)**). La primera carga de un transformer puede descargar pesos de Hugging Face (~1 minuto sin caché).

### 2. UI React

```bash
cd frontend
npm install
npm run dev
```

Abre [http://localhost:5173](http://localhost:5173) — Vite hace proxy de las rutas API (`/predict`, `/models/status`, etc.) al puerto 8000.

**Página Watch:** vídeos sugeridos, puntuación de comentarios, análisis en vivo del borrador.  
**Ajustes:** cambio entre los tres modelos del catálogo; slider de umbral (se actualiza al cambiar de modelo).  
**Moderator Hub:** historial de comentarios puntuados en la sesión.

Banner de producción (desde `/model-info`): p. ej. *Meta-Feature Stacking Model (F1: 0.805, Gap: 2.54%)*.

---

## Docker (API + UI compilada)

```bash
export YOUTUBE_API_KEY=tu_clave   # opcional pero recomendado para comentarios reales
docker compose up --build
```


| URL                                                      | Servicio                                       |
| -------------------------------------------------------- | ---------------------------------------------- |
| [http://localhost:8000](http://localhost:8000)           | FastAPI + `frontend/dist` (un solo contenedor) |
| [http://localhost:8000/docs](http://localhost:8000/docs) | Swagger                                        |


La imagen copia `models/baseline/` y `models/production_final/`. `INSTALL_HF=1` es el valor por defecto en `docker-compose.yml` para producción y el baseline BERT congelado. Para una imagen solo sklearn (baseline LR):

```bash
INSTALL_HF=0 docker compose build --build-arg INSTALL_HF=0
```

---

## Resumen de la API

Referencia completa: [docs/API.es.md](docs/API.es.md) · [docs/API.md](docs/API.md)


| Método | Ruta                | Descripción                                                           |
| ------ | ------------------- | --------------------------------------------------------------------- |
| `POST` | `/predict`          | Puntúa un comentario `{ "text", "threshold" }`                        |
| `POST` | `/predict-batch`    | Hasta 100 textos                                                      |
| `POST` | `/predict-video`    | Obtiene comentarios de YouTube y los puntúa (API key o fallback demo) |
| `GET`  | `/videos/suggested` | Metadatos del carril derecho (`configs/suggested_videos.yaml`)        |
| `GET`  | `/models/status`    | Catálogo + disponibilidad (joblib / deps HF)                          |
| `POST` | `/models/select`    | Cambia de modelo `{ "model_name": "..." }`                            |
| `GET`  | `/model-info`       | Metadatos del modelo activo (banner, umbral recomendado)              |


**Ejemplo**

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Thanks for the great tutorial!", "threshold": 0.381}'
```

Cambiar al baseline LR:

```bash
curl -s -X POST http://localhost:8000/models/select \
  -H "Content-Type: application/json" \
  -d '{"model_name": "LR + TF-IDF (Baseline)"}'
```

---

## Estructura del proyecto

```
youtube_hate_detector/
├── configs/
│   ├── model_catalog.yaml      # Modelos de demo (baselines + producción)
│   ├── pipeline.yaml           # Rutas de entrenamiento
│   ├── features.yaml
│   └── suggested_videos.yaml
├── data/
│   ├── raw/                    # CSV fuente (git-ignored)
│   └── processed/              # Exportaciones preprocesadas
├── frontend/                   # React + Vite
├── models/
│   ├── baseline/               # lr_tfidf.joblib, manifest.json
│   ├── production_final/       # meta_stack_final.joblib
│   └── README.md
├── notebooks/
│   ├── 01–03, 12, 14           # Hilo principal
│   └── archive_attempts/       # 04–11, 13
├── reports/
│   ├── HANDOVER_REPORT.md
│   ├── notebook_14/
│   ├── golden_baseline/
│   └── v2/                     # Figuras EDA del equipo
├── src/
│   ├── api/                    # Rutas FastAPI
│   ├── service/                # ModelService, predictor meta-stack
│   ├── pipeline/               # Pipelines de entrenamiento
│   ├── features/
│   └── evaluation/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── uv.lock
```

---

## Entrenamiento y reproducción de métricas


| Objetivo                         | Comando                                                      |
| -------------------------------- | ------------------------------------------------------------ |
| Baseline LR + TF-IDF             | `uv run python -m src.pipeline.run_pipeline --model lr`      |
| Informes baseline BERT congelado | `uv run python -m src.pipeline.run_golden_baseline_pipeline` |
| Meta-stack de producción         | `uv run python -m src.experiments.notebook_14_final_stack`   |


Detalle del pipeline: [docs/PIPELINE.es.md](docs/PIPELINE.es.md) · Resultados agregados: [docs/RESULTS.es.md](docs/RESULTS.es.md) · Ejecuciones históricas: `[reports/summary.csv](reports/summary.csv)`

---

## Configuración


| Archivo                         | Uso                                                                     |
| ------------------------------- | ----------------------------------------------------------------------- |
| `.env`                          | `YOUTUBE_API_KEY`, `MODEL_NAME`, `ENV`                                  |
| `configs/model_catalog.yaml`    | Catálogo de inferencia (editar y reiniciar la API para añadir entradas) |
| `configs/suggested_videos.yaml` | IDs de vídeo del carril sugerido                                        |
| `configs/best_params.yaml`      | Referencia Optuna LR para el baseline                                   |


No hagas commit de `.env`. Haz commit de `uv.lock` cuando cambien las dependencias.

---

## Tests

```bash
uv sync --extra dev --extra hf
uv run pytest
```

Cubre contratos de la API, preprocesado y cableado del catálogo para los tres modelos de demo.

---

## Índice de documentación


| English                                                  | Español                                            |
| -------------------------------------------------------- | -------------------------------------------------- |
| [docs/API.md](docs/API.md)                               | [docs/API.es.md](docs/API.es.md)                   |
| [docs/PIPELINE.md](docs/PIPELINE.md)                     | [docs/PIPELINE.es.md](docs/PIPELINE.es.md)         |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)             | [docs/ARCHITECTURE.es.md](docs/ARCHITECTURE.es.md) |
| [docs/RESULTS.md](docs/RESULTS.md)                       | [docs/RESULTS.es.md](docs/RESULTS.es.md)           |
| [reports/HANDOVER_REPORT.md](reports/HANDOVER_REPORT.md) |                                                    |


---

## Licencia y datos

Usa el dataset del proyecto y las claves de API según las normas de tu curso u organización. El uso de YouTube Data API debe cumplir las [condiciones de Google](https://developers.google.com/youtube/terms/api-services-terms-of-service).