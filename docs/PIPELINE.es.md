# Pipeline de entrenamiento

Punto de entrada: [`src/pipeline/run_pipeline.py`](../src/pipeline/run_pipeline.py)

## Comando

```bash
python -m src.pipeline.run_pipeline --model lr
```

| Flag | Valores | Por defecto |
|------|---------|-------------|
| `--model` | `lr`, `rf`, `xgboost` | `lr` |

Ejecutar desde la raíz del repositorio.

## Fases

1. **Carga** — CSV en `data/raw/youtoxic_english_1000.csv`
2. **Split** — train/test estratificado
3. **Preprocesado** — `TextPreprocessor` (spaCy + NLTK)
4. **Entrenamiento** — `build_model()`
5. **Validación cruzada** — 5 folds
6. **Evaluación** — `Evaluator.evaluate_and_report()` en test
7. **Guardado** — `models/experiments/{model}/`
8. **MLflow** — `mlruns/`
9. **Informes** — `reports/summary.csv` y `reports/pipeline/{model}/`

## Configuración

| Archivo | Uso |
|---------|-----|
| `configs/pipeline.yaml` | Rutas, `IsToxic`, split, CV |
| `configs/features.yaml` | TF-IDF y preprocesado |
| `configs/models.yaml` | Hiperparámetros de clasificadores |
| `configs/best_params.yaml` | Ganador Optuna (LR) |

## Salidas

| Ruta | Contenido |
|------|-----------|
| `reports/summary.csv` | Tabla comparativa de modelos |
| `reports/pipeline/lr/cm_lr.png` | Matriz de confusión |
| `reports/pipeline/lr/roc_lr.png` | Curva ROC |
| `reports/pipeline/lr/errors_lr.csv` | FP / FN |

## Inferencia del demo

Catálogo en [`configs/model_catalog.yaml`](../configs/model_catalog.yaml): **Meta-Feature Stacking** (producción), **LR + TF-IDF** y **Frozen Toxic-BERT** (baselines en `models/baseline/manifest.json`).
