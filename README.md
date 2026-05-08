# Project_9_Equipo3

## 🏗️  Arquitectura

```
youtube_hate_detector/
├── data/
│   ├── raw/                # Dataset original (no se sube al repo)
│   └── processed/          # Datos preprocesados
├── notebooks/              # EDA y experimentos
├── models/                 # Modelos entrenados (.joblib)
├── reports/                # Métricas y resultados por experimento
├── src/
│   ├── data/               # Carga y scraping de datos
│   ├── features/           # Preprocesamiento y vectorización
│   ├── models/             # Baseline, ensemble, deep learning
│   ├── evaluation/         # Métricas y análisis de errores
│   ├── pipeline/           # Pipeline end-to-end
│   ├── api/                # FastAPI
│   ├── app/                # Streamlit
│   └── utils/              # Logger, config loader
├── configs/                # Configuración YAML
├── tests/                  # Tests unitarios
└── logs/                   # Logs de ejecución

```