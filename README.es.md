<div align="center">
<br />
<img src="docs/assets/signalmod_logo.png" alt="SignalMod" width="520" />

<br />

<img src="docs/assets/qr.png" alt="Escanea para abrir SignalMod" width="220" />

### Moderación inteligente para comentarios de YouTube

🌐 [English](README.md) · **Español**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-5-646CFF?logo=vite&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-5.9-FFD21E?logo=huggingface&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikitlearn&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-FFD21E?logo=huggingface&logoColor=black)
![Supabase](https://img.shields.io/badge/Supabase-DB-3ECF8E?logo=supabase&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-compose-2496ED?logo=docker&logoColor=white)
![Render](https://img.shields.io/badge/Deploy-Render-46E3B7?logo=render&logoColor=white)

</div>

---

## Descripción del proyecto

**SignalMod** es un asistente de moderación inteligente para comentarios de YouTube. Clasifica automáticamente cada comentario como **Seguro** o **Tóxico**, devuelve una probabilidad entre 0 y 1 y etiqueta categorías de toxicidad (insulto, amenaza, odio identitario, contenido obsceno).

Está construido alrededor del modelo **hybrid meta-feature stacking** del equipo — embeddings de Toxic-BERT congelado combinados con metadatos y una regresión logística regularizada — que alcanza **F1 = 0,805** con una brecha train–test de **2,54 pp** sobre el split de 200 muestras del proyecto.

El producto se entrega como una API REST con FastAPI y una SPA React que imita la experiencia de YouTube Watch: eliges un vídeo, la API descarga los 50 comentarios más recientes vía la YouTube Data API, los puntúa y persiste cada predicción en Supabase para que cualquier visitante pueda ver el histórico completo.

---

## Herramientas y lenguajes

### Lenguajes
- **Python 3.12** — backend, pipelines de ML, evaluación.
- **TypeScript + React 18** — SPA del frontend.
- **SQL (PostgreSQL vía Supabase)** — persistencia de predicciones.

### Backend
- **FastAPI 0.136** — API REST, esquemas Pydantic, carga del modelo en lifespan.
- **Uvicorn** — servidor ASGI con hot reload.
- **scikit-learn 1.8** — baseline TF-IDF + meta-learner LogisticRegression.
- **Optuna** — búsqueda de hiperparámetros del baseline TF-IDF.
- **PyTorch 2.x + Transformers 5.9** — `unitary/toxic-bert` congelado para embeddings CLS.
- **spaCy + NLTK** — lematización, stopwords, limpieza basada en regex.
- **MLflow** — tracking de experimentos.
- **Supabase Python SDK** — persistencia de predicciones con políticas RLS anónimas.
- **google-api-python-client** — integración con YouTube Data API v3.

### Frontend
- **React 18 + Vite 5 + TypeScript** — SPA con hot module reload.
- **CSS modules** — tema oscuro estilo YouTube.

### Tooling y operaciones
- **uv** — gestor de paquetes y entorno virtual de Python (`pyproject.toml` + `uv.lock`).
- **pnpm** — gestor de paquetes del frontend.
- **Docker + Docker Compose** — despliegue en un único contenedor sirviendo API + SPA construida.
- **GNU Make** — `make dev`, `make install`, `make build`, `make docker`.
- **Render** — despliegue gratuito vía blueprint `render.yaml`.
- **Pytest** — tests unitarios de contratos de API y preprocesado.

---

## Arquitectura del proyecto

```
Project_9_Equipo3/
├── configs/                       # Configs YAML para pipelines y catálogo de inferencia
│   ├── pipeline.yaml              # Rutas de datos, target, folds de CV
│   ├── features.yaml              # Preprocesado y ajustes de TF-IDF
│   ├── model_catalog.yaml         # Catálogo de inferencia (3 modelos intercambiables)
│   ├── best_params.yaml           # Ganador de Optuna para el baseline LR
│   ├── suggested_videos.yaml      # IDs de YouTube del rail "Up next"
│   └── *_training.yaml            # Perfiles de entrenamiento (golden, expert, hybrid, …)
├── data/                          # Datasets crudos y procesados (git-ignored)
├── docs/                          # API.md, PIPELINE.md, ARCHITECTURE.md, DEPLOY.md
│   └── assets/signalmod_logo.png  # Activos de marca
├── frontend/                      # SPA React + Vite
│   ├── public/signalmod_logo.png  # Logo servido como activo estático
│   └── src/
│       ├── api/                   # Cliente HTTP tipado
│       ├── components/            # Layout, CommentRow, SuggestedRail, ModelBanner
│       ├── context/               # Estado global (modelo activo, umbral)
│       ├── hooks/                 # useDebouncedPredict
│       ├── pages/                 # WatchPage, HubPage, SettingsPage
│       └── utils/                 # toxicityColor, randomUsername, relativeTime
├── models/
│   ├── baseline/lr_tfidf.joblib   # Baseline LR ajustado con Optuna
│   └── production_final/          # meta_stack_final.joblib — artefacto de producción
├── notebooks/
│   ├── 01–04                      # EDA, preprocesado, TF-IDF, baseline LR
│   ├── 12                         # Golden baseline (Toxic-BERT congelado)
│   ├── 14                         # Meta-stacking final — artefacto de producción
│   └── archive_attempts/          # Experimentos anteriores conservados para reproducibilidad
├── reports/                       # Métricas, gráficos, figuras EDA, summary.csv
├── src/
│   ├── api/                       # App FastAPI
│   │   ├── main.py                # Lifespan, CORS, montaje del SPA estático
│   │   ├── routes/                # health, models, predict (+ /predictions), videos
│   │   ├── schemas.py             # Modelos Pydantic request/response
│   │   ├── services.py            # predict_single, to_predict_response
│   │   ├── state.py               # Estado compartido de la app
│   │   └── youtube.py             # Fetch a YouTube Data API + metadatos sugeridos
│   ├── data/                      # Loader, dual loader para pipelines híbridos
│   ├── db/                        # Cliente Supabase + helpers save_prediction
│   ├── evaluation/                # Evaluator, threshold tuning, CV estable
│   ├── experiments/               # Versiones script de los notebooks 13 / 14
│   ├── features/                  # text_preprocessor, vectorizer, metadata, augmentation
│   ├── models/                    # baseline (LR/RF/XGBoost), hybrid_ensemble, metadata_lr
│   ├── pipeline/                  # run_pipeline + variantes por estrategia
│   ├── service/                   # ModelService, meta_stack_predictor, model_catalog
│   └── utils/                     # Logger
├── supabase/predictions_setup.sql # SQL para crear la tabla predictions + políticas RLS
├── tests/                         # Suite Pytest
├── Dockerfile                     # Build multi-stage (frontend + backend con uv)
├── docker-compose.yml             # Despliegue de un contenedor (API + SPA)
├── render.yaml                    # Blueprint de Render (web service + static site)
├── Procfile                       # Declaración de proceso para Render
├── Makefile                       # make dev / install / build / docker / test
├── pyproject.toml + uv.lock       # Dependencias Python fijadas con uv
└── README.md  /  README.es.md     # Documentación en inglés / español
```

### Flujo de datos

```
                ┌────────────────────────────────────────────────┐
                │  SPA React (Vite)         http://localhost:5173│
                │  Layout · Watch · Hub · Settings               │
                └──────────────────┬─────────────────────────────┘
                                   │ HTTP JSON  (proxy Vite → :8000)
                ┌──────────────────▼─────────────────────────────┐
                │  FastAPI                  http://localhost:8000│
                │  /predict  /predict-batch  /predict-video      │
                │  /predictions (GET — histórico de Supabase)    │
                │  /models  /models/select  /model-info          │
                │  /videos/suggested  /health                    │
                └──────┬─────────────────────────────┬───────────┘
                       │                             │
        ┌──────────────▼─────────────┐ ┌─────────────▼──────────────┐
        │  ModelService              │ │  YouTube Data API v3       │
        │  · local joblib            │ │  · metadatos de vídeo      │
        │  · hf_remote               │ │  · 50 comentarios + nuevos │
        │  · meta_stack (producción) │ │                            │
        └──────┬─────────────────────┘ └────────────────────────────┘
               │
        ┌──────▼──────────────────────────────────────────────────┐
        │  Supabase (PostgreSQL)                                  │
        │  tabla: predictions(id, created_at, text, video_id,     │
        │                     probability, is_toxic, labels, …)   │
        │  RLS: insert anónimo + select anónimo                   │
        └─────────────────────────────────────────────────────────┘
```

### Catálogo de modelos (intercambiable desde la UI)

| Modelo                           | Tipo        | F1 (test) | Brecha train–test | Umbral    | Latencia | Default |
| -------------------------------- | ----------- | --------- | ----------------- | --------- | -------- | ------- |
| **Meta-Feature Stacking**        | Híbrido     | **0,805** | **2,54 pp**       | **0,381** | ~400 ms  | **Sí**  |
| Frozen Toxic-BERT                | Transformer | 0,790     | 0,16 pp           | 0,120     | ~400 ms  | No      |
| LR + TF-IDF (Optuna)             | sklearn     | 0,758     | 4,76 pp           | 0,500     | < 50 ms  | No      |

El modelo de producción concatena el embedding `[CLS]` congelado de `unitary/toxic-bert` (768-d) con metadatos hechos a mano (longitud, ratio de mayúsculas, densidad de emojis…), los escala con `StandardScaler` y los pasa por un meta-learner `LogisticRegression(C=0,001)`.

---

## Instalación y ejecución

### 1. Requisitos previos

| Herramienta  | macOS / Linux                       | Windows                                                  |
| ------------ | ----------------------------------- | -------------------------------------------------------- |
| **Python 3.12** | `brew install python@3.12`      | [python.org/downloads](https://www.python.org/downloads/) (marca *Add Python to PATH*) |
| **uv**       | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | `powershell -c "irm https://astral.sh/uv/install.ps1 \| iex"` |
| **Node.js 18+** | `brew install node`             | [nodejs.org](https://nodejs.org/) (LTS)                 |
| **pnpm**     | `npm i -g pnpm`                     | `npm i -g pnpm`                                          |
| **Make** *(opcional)* | ya instalado               | `winget install GnuWin32.Make`  (o usa WSL)              |

### 2. Clonar y configurar

```bash
git clone https://github.com/Bootcamp-IA-P6/Project_9_Equipo3.git
cd Project_9_Equipo3

cp .env.example .env
# Rellena: YOUTUBE_API_KEY, SUPABASE_URL, SUPABASE_KEY
```

> **PowerShell de Windows**: sustituye `cp` por `Copy-Item .env.example .env`.

Pega `supabase/predictions_setup.sql` en el editor SQL de Supabase antes del primer arranque (crea la tabla `predictions` + políticas RLS).

### 3. Arranque — tres opciones

#### Opción A — Con Makefile (recomendada en macOS / Linux / WSL)

```bash
make install     # uv sync  +  pnpm install
make dev         # FastAPI :8000  +  Vite :5173
```

| Comando       | Qué hace                                       |
| ------------- | ---------------------------------------------- |
| `make install`| Instala deps de Python + frontend              |
| `make dev`    | Arranca API y UI en paralelo (Ctrl+C los para) |
| `make api`    | Solo la API                                    |
| `make ui`     | Solo la UI                                     |
| `make build`  | Compila el SPA a `frontend/dist`               |
| `make test`   | Ejecuta Pytest                                 |
| `make docker` | `docker compose up --build`                    |
| `make stop`   | Mata procesos en los puertos 8000 / 5173       |
| `make clean`  | Borra `.venv`, `node_modules`, `dist`          |

#### Opción B — Manual (macOS / Linux)

Dos terminales.

**Terminal 1 — API**
```bash
uv sync
uv run uvicorn src.api.main:app --reload --port 8000
```

**Terminal 2 — Frontend**
```bash
cd frontend
pnpm install
pnpm dev
```

#### Opción C — Manual (PowerShell de Windows)

Dos terminales.

**Terminal 1 — API**
```powershell
uv sync
uv run uvicorn src.api.main:app --reload --port 8000
```

**Terminal 2 — Frontend**
```powershell
cd frontend
pnpm install
pnpm dev
```

> Si `uv` no se reconoce tras instalarlo, cierra y vuelve a abrir PowerShell para que se recargue el `PATH`.

### 4. Abrir la aplicación

| URL                            | Qué verás                                |
| ------------------------------ | ---------------------------------------- |
| http://localhost:5173          | SPA React — Watch / Hub / Settings       |
| http://localhost:8000/docs     | Swagger de FastAPI                       |
| http://localhost:8000/health   | Health check                             |

### 5. Docker (un solo contenedor — API + SPA compilada)

Mismos comandos en **macOS / Linux / Windows**:

```bash
# Normal — deja imágenes y volúmenes para builds rápidos
docker compose up --build
# → http://localhost:8000  ·  Ctrl+C para parar  ·  docker compose down

# Demo efímera — Ctrl+C borra contenedor + imagen + volúmenes
make docker-demo

# Limpieza manual completa
make docker-clean
# (equivale a: docker compose down --rmi local --volumes --remove-orphans)
```

---

Más detalle: [docs/PIPELINE.es.md](docs/PIPELINE.es.md) para entrenamiento, [docs/API.es.md](docs/API.es.md) para endpoints, [docs/DEPLOY.md](docs/DEPLOY.md) para despliegue en Render.

---

## Colaboradores

<table>
  <tr>
    <td align="center" width="25%">
      <b>Andrés Torrez</b><br/>
      <sub>Backend Developer</sub>
    </td>
    <td align="center" width="25%">
      <b>Mirae Kang</b><br/>
      <sub>Scrum Master</sub>
    </td>
    <td align="center" width="25%">
      <b>Jonathan Brasales</b><br/>
      <sub>AI Developer</sub>
    </td>
    <td align="center" width="25%">
      <b>Roberto Molero</b><br/>
      <sub>Product Owner</sub>
    </td>
  </tr>
</table>
