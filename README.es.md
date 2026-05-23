# Detector de comentarios tóxicos en YouTube (SignalMod)

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/docker-compose-2496ED.svg)](https://docs.docker.com/compose/)

**English:** [README.md](README.md)

Clasificación binaria **Seguro vs Tóxico** para comentarios estilo YouTube. Stack de producción: **FastAPI** (API REST) y **Streamlit** (interfaz tipo página de vídeo). Modelo por defecto: **Regresión logística + TF-IDF** (`models/final_model.joblib`).

---

## Descripción del proyecto

| Elemento | Detalle |
|----------|---------|
| **Objetivo** | Apoyar a moderadores detectando comentarios tóxicos |
| **Dataset** | `data/raw/youtoxic_english_1000.csv` (~1000 comentarios en inglés) |
| **Etiqueta** | `IsToxic` → **Seguro (0)** / **Tóxico (1)** |
| **Métrica principal** | F1 ponderado y ROC-AUC |
| **Control de sobreajuste** | \|F1 CV − F1 test\| &lt; 5 puntos porcentuales |

---

## Arquitectura

```
youtube_hate_detector/
├── configs/              # YAML: pipeline, features, models, best_params
├── data/raw/             # CSV fuente
├── models/               # final_model.joblib, experimentos/
├── reports/              # summary.csv, gráficos, artefactos del pipeline
├── src/
│   ├── api/              # FastAPI
│   ├── app/              # Streamlit (src/app/app.py)
│   ├── evaluation/       # Evaluator
│   ├── features/         # Preprocesado y vectorización
│   ├── models/           # LR, RF, XGBoost
│   ├── pipeline/         # Entrenamiento end-to-end
│   └── service/          # ModelService
├── tests/
├── Dockerfile
└── docker-compose.yml
```

**Flujo:** entrenamiento (`run_pipeline`) → inferencia API o Streamlit vía `ModelService`.

Más detalle: [docs/ARCHITECTURE.es.md](docs/ARCHITECTURE.es.md)

---

## Instalación

```bash
git clone https://github.com/Bootcamp-IA-P6/Project_9_Equipo3.git
cd Project_9_Equipo3

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Coloca `youtoxic_english_1000.csv` en `data/raw/`.

```bash
cp .env.example .env
# Opcional: YOUTUBE_API_KEY, MODEL_NAME
```

---

## Pipeline de entrenamiento

```bash
python -m src.pipeline.run_pipeline --model lr
# lr | rf | xgboost
```

Actualiza [`reports/summary.csv`](reports/summary.csv) y guarda gráficos en `reports/pipeline/{model}/`.

Documentación: [docs/PIPELINE.es.md](docs/PIPELINE.es.md)

---

## Docker

```bash
docker compose up --build
```

| Servicio | URL |
|----------|-----|
| Streamlit | http://localhost:8501 |
| FastAPI | http://localhost:8000 |
| Swagger | http://localhost:8000/docs |

```bash
docker compose down
```

---

## Ejecución local

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
streamlit run src/app/app.py --server.port 8501
```

---

## Ejemplos de API

Ver [docs/API.es.md](docs/API.es.md)

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Great video!", "threshold": 0.5}'
```

---

## Resultados

Mejor modelo **sklearn** en test (`configs/best_params.yaml`):

| Métrica | Valor |
|---------|-------|
| F1 (ponderado, test) | **0.7579** |
| ROC-AUC | **0.81** |
| Falsos positivos | 18 |
| Falsos negativos | 30 |
| Brecha CV–test | **4.76 pp** |

Gráficos EDA: `reports/v2/`.

---

## Informe técnico de resultados

- **Español:** [reports/final_report.es.md](reports/final_report.es.md)
- **English:** [reports/final_report.md](reports/final_report.md)

## Comparativa de modelos

Tabla canónica: [`reports/summary.csv`](reports/summary.csv)  
Resumen: [docs/RESULTS.es.md](docs/RESULTS.es.md)

| Modelo | Familia | F1 (test) | ROC-AUC | Por defecto |
|--------|---------|-----------|---------|-------------|
| LR + TF-IDF (ajustado) | sklearn | 0.7579 | 0.81 | Sí |
| RF / XGBoost | sklearn | — | — | Ejecutar pipeline |
| DistilBERT / toxic-bert / RoBERTa | Hugging Face | — | — | Opcional en API/UI |

---

## Tests

```bash
pytest tests/ -v
```

---

## Índice de documentación

| Español | English |
|---------|---------|
| [docs/API.es.md](docs/API.es.md) | [docs/API.md](docs/API.md) |
| [docs/PIPELINE.es.md](docs/PIPELINE.es.md) | [docs/PIPELINE.md](docs/PIPELINE.md) |
| [docs/ARCHITECTURE.es.md](docs/ARCHITECTURE.es.md) | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| [docs/RESULTS.es.md](docs/RESULTS.es.md) | [docs/RESULTS.md](docs/RESULTS.md) |
| [reports/final_report.es.md](reports/final_report.es.md) | [reports/final_report.md](reports/final_report.md) |
