# Resultados y comparativa de modelos

Datos: [`reports/summary.csv`](../reports/summary.csv)  
Hiperparámetros: [`configs/best_params.yaml`](../configs/best_params.yaml)  
**Informe técnico completo:** [`reports/final_report.es.md`](../reports/final_report.es.md) · [EN](../reports/final_report.md)

## Mejor modelo sklearn (producción)

**Ganador:** Regresión logística + TF-IDF (Optuna), archivo `models/final_model.joblib`.

| Métrica | Valor en test | Notas |
|---------|---------------|-------|
| F1 (ponderado) | **0.7579** | Métrica principal |
| ROC-AUC | **0.81** | |
| Falsos positivos | **18** | Seguros marcados como tóxicos |
| Falsos negativos | **30** | Tóxicos no detectados |
| F1 (train) | 0.8987 | |
| Brecha train–test | 14.07 pp | |
| Brecha CV–test | **4.76 pp** | Objetivo &lt; 5 pp |

## Tabla comparativa

| Modelo | Familia | F1 (test) | ROC-AUC | FP | FN | Por defecto |
|--------|---------|-----------|---------|----|----|-------------|
| LR + TF-IDF (ajustado) | sklearn | 0.7579 | 0.81 | 18 | 30 | Sí |
| LR + TF-IDF (local) | sklearn | 0.7579 | 0.81 | 18 | 30 | Sí |
| Random Forest | sklearn | — | — | — | — | Ejecutar `--model rf` |
| XGBoost | sklearn | — | — | — | — | Ejecutar `--model xgboost` |
| DistilBERT Toxicity | Hugging Face | — | — | — | — | Opcional en API |
| toxic-bert | Hugging Face | — | — | — | — | Opcional |
| RoBERTa Toxicity | Hugging Face | — | — | — | — | Opcional |

## Actualizar métricas

```bash
python -m src.pipeline.run_pipeline --model lr
python -m src.pipeline.run_pipeline --model rf
python -m src.pipeline.run_pipeline --model xgboost
```

Salidas: `reports/summary.csv`, gráficos en `reports/pipeline/{model}/`.

## EDA

Figuras adicionales en `reports/v2/`.

## Análisis de errores

Términos frecuentes en FP/FN y ejemplos en `reports/pipeline/*/errors_*.csv`.
