# Arquitectura del sistema

## Componentes

```mermaid
flowchart TB
  subgraph datos [Capa de datos]
    CSV[data/raw/youtoxic_english_1000.csv]
    CFG[configs/*.yaml]
  end

  subgraph entrenamiento [Entrenamiento]
    PIPE[run_pipeline.py]
    PRE[TextPreprocessor]
    BL[build_model]
    EV[Evaluator]
    CSV --> PIPE
    CFG --> PIPE
    PIPE --> PRE --> BL --> EV
    EV --> SUM[reports/summary.csv]
  end

  subgraph inferencia [Inferencia]
    MS[ModelService]
    API[FastAPI]
    UI[Streamlit]
    MS --> API
    MS --> UI
  end
```

## Módulos

| Módulo | Función |
|--------|---------|
| `src/data/loader.py` | Carga del dataset |
| `src/features/text_preprocessor.py` | Limpieza y lematización |
| `src/models/baseline.py` | Modelos sklearn + TF-IDF |
| `src/evaluation/evaluator.py` | Métricas y comparativa |
| `src/pipeline/run_pipeline.py` | Pipeline completo |
| `src/service/model_service.py` | Predicción unificada |
| `src/api/main.py` | API REST |
| `src/app/app.py` | Interfaz Streamlit |

## Etiquetas

- Binario: `IsToxic` → Seguro (0) / Tóxico (1)
- API: `is_toxic`, `probability`

## Docker

Dos servicios: API (8000) y Streamlit (8501), imagen `youtube_hate_detector:latest`.
