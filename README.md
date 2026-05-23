# YouTube Toxic Comment Detector (SignalMod)

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/docker-compose-2496ED.svg)](https://docs.docker.com/compose/)
**Español:** [README.es.md](README.es.md)

Automated **Safe vs Toxic** classification for YouTube-style comments. The production stack is **FastAPI** (REST inference) plus **Streamlit** (watch-page style UI). The default model is **Logistic Regression + TF-IDF** (`models/final_model.joblib`).

---

## Project description

| Item | Detail |
|------|--------|
| **Goal** | Help moderation teams flag toxic comments quickly |
| **Dataset** | `data/raw/youtoxic_english_1000.csv` (~1k English comments) |
| **Target** | `IsToxic` → **Safe (0)** / **Toxic (1)** |
| **Primary metric** | Weighted F1 and ROC-AUC (imbalanced classes) |
| **Overfitting check** | \|CV F1 − test F1\| &lt; 5 percentage points (project rubric) |

---

## Architecture

```
youtube_hate_detector/
├── configs/              # YAML: pipeline, features, models, best_params
├── data/raw/             # Source CSV (not committed if gitignored)
├── models/               # final_model.joblib, experiments/
├── reports/              # summary.csv, plots, pipeline artifacts
├── src/
│   ├── api/              # FastAPI — /predict, /predict-batch, …
│   ├── app/              # Streamlit UI (src/app/app.py)
│   ├── data/             # load_raw_data, scraping helpers
│   ├── evaluation/       # Evaluator — metrics, ROC, confusion matrix
│   ├── features/         # TextPreprocessor, Vectorizer
│   ├── models/           # LR, RF, XGBoost baselines
│   ├── pipeline/         # run_pipeline.py — train end-to-end
│   └── service/          # ModelService — shared inference layer
├── tests/
├── Dockerfile
└── docker-compose.yml
```

**Runtime flow**

1. **Training:** `load_raw_data` → `TextPreprocessor` → `build_model().fit()` → `Evaluator` → `reports/summary.csv`
2. **API:** `uvicorn` loads `ModelService` → `POST /predict`
3. **Streamlit:** `ModelService.predict()` in-process (same models as API catalog)

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for more detail.

---

## Installation

**Requirements:** Python 3.12+, ~2 GB disk for dependencies (optional PyTorch if using Hugging Face models in the UI).

```bash
git clone https://github.com/Bootcamp-IA-P6/Project_9_Equipo3.git
cd Project_9_Equipo3   # or your local folder name

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**Data:** place `youtoxic_english_1000.csv` under `data/raw/` (path in `configs/pipeline.yaml`).

**Environment:**

```bash
cp .env.example .env
# Optional: YOUTUBE_API_KEY for /predict-video
# MODEL_NAME must match a key in ModelService (default: LR + TF-IDF (local))
```

---

## Training pipeline

End-to-end training and evaluation:

```bash
python -m src.pipeline.run_pipeline --model lr
# Options: lr | rf | xgboost
```

**Phases:** load data → stratified split → spaCy/NLTK preprocessing → train → 5-fold CV → test metrics → save `models/experiments/{model}/` → MLflow → update [`reports/summary.csv`](reports/summary.csv) and plots under `reports/pipeline/{model}/`.

Config files:

| File | Purpose |
|------|---------|
| `configs/pipeline.yaml` | Paths, `IsToxic`, test_size, CV folds |
| `configs/features.yaml` | Preprocessing + TF-IDF |
| `configs/models.yaml` | Classifier hyperparameters |
| `configs/best_params.yaml` | Optuna winner (LR) |

Details: [docs/PIPELINE.md](docs/PIPELINE.md)

---

## Run with Docker

```bash
docker compose up --build
```

| Service | URL |
|---------|-----|
| Streamlit | http://localhost:8501 |
| FastAPI | http://localhost:8000 |
| Swagger | http://localhost:8000/docs |

```bash
export YOUTUBE_API_KEY=your_key   # optional
docker compose down               # stop
```

Containers: `youtube_hate_detector-api`, `youtube_hate_detector-streamlit`.

---

## Local run (without Docker)

```bash
# Terminal 1 — API
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Streamlit
streamlit run src/app/app.py --server.port 8501
```

---

## API examples

Full reference: [docs/API.md](docs/API.md)

**Health check**

```bash
curl -s http://localhost:8000/ | python -m json.tool
```

**Single prediction**

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This video is amazing, thanks for sharing!", "threshold": 0.5}'
```

Example response:

```json
{
  "text": "This video is amazing, thanks for sharing!",
  "is_toxic": false,
  "probability": 0.08,
  "labels": [],
  "model_used": "LR + TF-IDF (local)",
  "latency_ms": 12.5
}
```

**Batch**

```bash
curl -s -X POST http://localhost:8000/predict-batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great content!", "You are an idiot"], "threshold": 0.5}'
```

**List / switch models**

```bash
curl -s http://localhost:8000/models
curl -s -X PUT http://localhost:8000/model/DistilBERT%20Toxicity
```

---

## Results

Best **sklearn** model on the project test split (from `configs/best_params.yaml`):

| Metric | Value |
|--------|-------|
| F1 (weighted, test) | **0.7579** |
| ROC-AUC | **0.81** |
| False positives | 18 |
| False negatives | 30 |
| CV–test gap | **4.76 pp** (within 5 pp target) |
| Train–test gap | 14.07 pp |

Plots and EDA: `reports/v2/`. Per-run artifacts: `reports/pipeline/{lr,rf,xgboost}/`.

---

## Technical results report

Full write-up (decisions, metrics, error analysis, limitations, roadmap):

- **English:** [reports/final_report.md](reports/final_report.md)
- **Español:** [reports/final_report.es.md](reports/final_report.es.md)

## Model comparison

Canonical table: [`reports/summary.csv`](reports/summary.csv)  
Human-readable: [docs/RESULTS.md](docs/RESULTS.md)

| Model | Family | F1 (test) | ROC-AUC | FP | FN | Production default |
|-------|--------|-----------|---------|----|----|--------------------|
| LR + TF-IDF (tuned) | sklearn | 0.7579 | 0.81 | 18 | 30 | Yes |
| LR + TF-IDF (local) | sklearn | 0.7579 | 0.81 | 18 | 30 | Yes (`final_model.joblib`) |
| RF / XGBoost | sklearn | — | — | — | — | Run pipeline to fill |
| DistilBERT / toxic-bert / RoBERTa | Hugging Face | — | — | — | — | Optional via API/UI |

Re-run `python -m src.pipeline.run_pipeline --model rf` to append RF metrics to `summary.csv`.

---

## Tests

```bash
pytest tests/ -v
```

Covers preprocessor, vectorizer, model binary output, and `/predict` response shape.

---

## Documentation index

| English | Español |
|---------|---------|
| [docs/API.md](docs/API.md) | [docs/API.es.md](docs/API.es.md) |
| [docs/PIPELINE.md](docs/PIPELINE.md) | [docs/PIPELINE.es.md](docs/PIPELINE.es.md) |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | [docs/ARCHITECTURE.es.md](docs/ARCHITECTURE.es.md) |
| [docs/RESULTS.md](docs/RESULTS.md) | [docs/RESULTS.es.md](docs/RESULTS.es.md) |
| [reports/final_report.md](reports/final_report.md) | [reports/final_report.es.md](reports/final_report.es.md) |
