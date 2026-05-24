# Informe técnico de resultados — Detector de comentarios tóxicos (SignalMod)

**Proyecto:** Asistente de moderación binario (Seguro vs Tóxico) para comentarios estilo YouTube  
**Dataset:** `youtoxic_english_1000.csv` (1.000 filas)  
**Modelo en producción:** `models/final_model.joblib` — Regresión logística + TF-IDF (Optuna)  
**Fecha del informe:** 2026-05-23  
**Artefactos:** [`summary.csv`](summary.csv) · [`pipeline/lr/`](pipeline/lr/) · EDA [`v2/`](v2/)

**English version:** [final_report.md](final_report.md)

---

## 1. Resumen ejecutivo

Se implementó un pipeline NLP completo (preprocesado → TF-IDF → clasificador), una API FastAPI y una demo Streamlit. El **modelo seleccionado para producción** es **LR + TF-IDF** ajustado con Optuna, con **F1 (ponderado) = 0,7579** y **ROC-AUC = 0,81** en el test hold-out, y una **brecha CV–test de 4,76 pp** (dentro del objetivo de &lt; 5 pp). Los modelos transformer están disponibles de forma opcional en el catálogo de la API, pero no son el predeterminado por latencia, dependencias y falta de evaluación en el mismo test del proyecto.

---

## 2. Decisiones tomadas

| Área | Decisión | Motivo |
|------|----------|--------|
| **Formulación** | Clasificación binaria sobre `IsToxic` | `configs/pipeline.yaml`; modo `binary` por defecto. |
| **Etiquetas (UI/API)** | **Seguro** / **Tóxico** | API (`is_toxic`) y Streamlit. |
| **Preprocesado** | Minúsculas → regex → lemas spaCy → stopwords NLTK + custom | `TextPreprocessor`. |
| **Tokens sensibles** | Conservar *black*, *white*, *police*, *cop*, etc. | Necesarios para contexto/bigramas (EDA). |
| **Vectorización** | TF-IDF (1–2 gramas) | `max_features=4045`, `min_df=2` (Optuna). |
| **Modelos base** | LR (ganador), RF y XGBoost en pipeline | `build_model()` + `--model`. |
| **Búsqueda de hiperparámetros** | Optuna → `best_params.yaml` | Exportado a `final_model.joblib`. |
| **Métrica principal** | F1 ponderado + ROC-AUC | `configs/models.yaml`. |
| **Sobreajuste** | \|F1 CV − F1 test\| &lt; 5 pp | `cv_test_gap_pp` en `Evaluator`. |
| **Desbalance** | `class_weight: balanced` | LR y RF en configuración. |
| **Serving** | `ModelService` + FastAPI + Streamlit | joblib local; HF vía `PUT /model/{name}`. |
| **Despliegue** | Docker Compose (`youtube_hate_detector`) | API :8000, Streamlit :8501. |
| **Trazabilidad** | MLflow + `reports/summary.csv` | Fases 8–9 del pipeline. |

---

## 3. Dataset y limitaciones

### 3.1 Estadísticas

| Estadística | Valor |
|-------------|-------|
| Comentarios totales | 1.000 |
| Seguros (`IsToxic = 0`) | 538 (53,8 %) |
| Tóxicos (`IsToxic = 1`) | 462 (46,2 %) |
| Longitud media del texto | ~186 caracteres |

Partición: **80 % train / 20 % test** estratificado → **200 muestras de test** por ejecución del pipeline.

### 3.2 Limitaciones del dataset

1. **Tamaño reducido (~1k)** — Alta varianza; subtipos de toxicidad poco representados.
2. **Solo inglés** — No generaliza a otros idiomas sin reentrenar.
3. **Sesgo temático** — Comentarios ligados a vídeos/noticias concretas (p. ej. contexto Ferguson).
4. **Multietiqueta dispersa** — `IsRacist`, `IsSexist`, etc. con muy pocos positivos; se mantuvo binario `IsToxic` (ver `reports/v2/05_multilabel_overlap.png`).
5. **Ruido en etiquetas** — Sarcasmo, ironía y casos límite subjetivos.
6. **Pérdida en preprocesado** — Lematización y stopwords pueden eliminar señales; textos vacíos se rellenan con el original.
7. **Vocabulario identitario** — *black*, *white*, *police* aparecen en FP y FN; riesgo en discurso político no tóxico.

---

## 4. Métricas de todos los modelos

Fuente canónica: [`summary.csv`](summary.csv).

### 4.1 Modelo sklearn en producción (LR Optuna)

Desde `configs/best_params.yaml` — referencia de `final_model.joblib`:

| Métrica | Valor |
|---------|-------|
| F1 (ponderado, test) | **0,7579** |
| F1 (train) | 0,8987 |
| ROC-AUC | **0,81** |
| Falsos positivos (FP) | 18 |
| Falsos negativos (FN) | 30 |
| Brecha train–test | 14,07 pp |
| **Brecha CV–test** | **4,76 pp** ✓ |

**Hiperparámetros:** `C ≈ 0,32`, `max_features = 4045`, bigramas, `min_df = 2`.

### 4.2 Re-ejecución del pipeline (LR, configuración por defecto)

Última corrida: `reports/pipeline/lr/exp_20260523_163600_lr.json`

| Métrica | Valor |
|---------|-------|
| F1 (ponderado, test) | 0,7387 |
| F1 (tóxico) | 0,7045 |
| ROC-AUC | 0,7838 |
| Precisión / recall (ponderados) | 0,7399 / 0,74 |
| FP / FN | 22 / 30 |
| F1 CV medio ± std | 0,7193 ± 0,0382 |
| Brecha CV–test | 1,94 pp |

El artefacto ajustado (**0,7579**) supera esta corrida; producción usa `final_model.joblib` tunado.

### 4.3 Pendientes en la tabla comparativa

| Modelo | Estado |
|--------|--------|
| Random Forest | `python -m src.pipeline.run_pipeline --model rf` |
| XGBoost | `python -m src.pipeline.run_pipeline --model xgboost` |

### 4.4 Modelos Hugging Face (catálogo API)

Disponibles en `ModelService`; **sin métricas en el test del proyecto** en `summary.csv`:

| Modelo | F1 (catálogo externo) | Producción |
|--------|------------------------|------------|
| LR + TF-IDF (local) | ~0,76 | **Sí** |
| DistilBERT Toxicity | ~0,85 | No |
| toxic-bert (multietiqueta) | ~0,88 | No |
| RoBERTa Toxicity | ~0,87 | No |

Experimentos notebook: `reports/v2/nb08_*`.

### 4.5 Tabla resumen

| Modelo | F1 (test) | ROC-AUC | FP | FN | Brecha CV–test | Por defecto |
|--------|-----------|---------|----|----|----------------|-------------|
| **LR + TF-IDF (ajustado)** | **0,7579** | **0,81** | 18 | 30 | **4,76 pp** | Sí |
| LR (re-ejecución pipeline) | 0,7387 | 0,784 | 22 | 30 | 1,94 pp | — |
| RF / XGBoost | — | — | — | — | — | Ejecutar pipeline |
| HF (catálogo) | — | — | — | — | — | Opcional |

---

## 5. Modelo seleccionado y por qué

**Seleccionado:** **Regresión logística + TF-IDF** (`models/final_model.joblib`, Optuna).

**Motivos**

1. **Mejor rendimiento** en el test del proyecto (F1 0,7579, ROC-AUC 0,81).
2. **Cumple el criterio de generalización** — brecha CV–test &lt; 5 pp.
3. **Operación** — inferencia rápida, sin GPU, artefacto pequeño, Docker ligero.
4. **Interpretabilidad** — coeficientes TF-IDF (`reports/v2/11_lr_coeficientes.png`).
5. **Mismo stack** en entrenamiento, API y Streamlit (`ModelService`).

**Por qué no transformer por defecto**

- Más peso (torch/transformers), arranque lento.
- Cifras del catálogo no evaluadas en `youtoxic_english_1000`.
- Útiles como **opción** en demos vía API.

---

## 6. Análisis de errores

Fuente: `reports/pipeline/lr/errors_lr.csv` (última corrida LR, n=200 test).

### 6.1 Resumen de confusión (pipeline LR)

| | Predicho seguro | Predicho tóxico |
|--|-----------------|-----------------|
| **Real seguro** | VN | **22 FP** |
| **Real tóxico** | **30 FN** | VP |

Modelo tunado en producción: **18 FP / 30 FN**.

### 6.2 Términos más frecuentes en errores

**Falsos positivos:** `black(14)`, `white(9)`, `shoot(8)`, `would(9)`, `police(5)`, `cop(6)` — el modelo reacciona a **vocabulario racial/policial en contexto informativo**.

**Falsos negativos:** `police(8)`, `criminal(6)`, `kill(5)`, `black(6)` — **toxicidad indirecta o comentarios largos** por debajo del umbral.

### 6.3 Patrones

| Tipo | Patrón |
|------|--------|
| FP | Debate político/racial sin insulto directo |
| FP | Palabrotas en contexto de frustración no dirigida |
| FN | Odio implícito, sarcasmo, texto muy largo |

Gráficos: [`pipeline/lr/cm_lr.png`](pipeline/lr/cm_lr.png), [`pipeline/lr/roc_lr.png`](pipeline/lr/roc_lr.png).

---

## 7. Posibles mejoras futuras

| Prioridad | Mejora | Beneficio esperado |
|-----------|--------|-------------------|
| Alta | **Más datos etiquetados** | Menos FP en política; mejor recall |
| Alta | **Ajuste de umbral** en validación | Alinear FP/FN con política de moderación |
| Alta | **Evaluar RF, XGBoost y DistilBERT** en el mismo test | Comparativa justa en `summary.csv` |
| Media | **Augmentación** (back-translation) | Menos sobreajuste (`reports/v2/15_*`) |
| Media | **Fine-tuning DistilBERT** en datos del proyecto | Mejor contexto que TF-IDF |
| Media | **Ensemble** LR + transformer | Ver `reports/v2/12_*` |
| Media | **Cola de revisión humana** | Casos con probabilidad 0,4–0,6 |
| Baja | **Cabezas multietiqueta** | Solo si hay positivos suficientes por etiqueta |
| Baja | **Frontend React** | UX tipo YouTube en producción |
| Baja | **PostgreSQL** | Auditoría de predicciones |

---

## 8. Reproducibilidad

```bash
python -m src.pipeline.run_pipeline --model lr
cat reports/summary.csv
docker compose up --build
```

---

## 9. Referencias

| Documento | Ruta |
|-----------|------|
| CSV comparativo | [`summary.csv`](summary.csv) |
| Resultados (ES) | [`../docs/RESULTS.es.md`](../docs/RESULTS.es.md) |
| Pipeline (ES) | [`../docs/PIPELINE.es.md`](../docs/PIPELINE.es.md) |
| API (ES) | [`../docs/API.es.md`](../docs/API.es.md) |
| Mejores hiperparámetros | [`../configs/best_params.yaml`](../configs/best_params.yaml) |
