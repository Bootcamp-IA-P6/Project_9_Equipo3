"""
src/pipeline/run_pipeline.py

Pipeline end-to-end de entrenamiento y evaluación.
Ejecutar con: python -m src.pipeline.run_pipeline [--model lr|rf|xgboost]

Fases:
    1. Carga de datos
    2. Split train/test
    3. Preprocesamiento (spaCy + NLTK)
    4. Entrenamiento (modelo elegido desde config o argumento)
    5. Cross-validation
    6. Evaluación final en test
    7. Guardado del modelo
    8. Registro en MLflow
    9. Informe JSON + CSV resumen

Todo se controla desde:
    configs/pipeline.yaml  → rutas, split, cv_folds
    configs/features.yaml  → preprocesamiento y vectorización
    configs/models.yaml    → hiperparámetros
    configs/best_params.yaml → resultado de Optuna (si existe)
"""

import argparse
import sys
import yaml
import mlflow
import mlflow.sklearn
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split

# ── Setup path ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_raw_data
from src.features.text_preprocessor import TextPreprocessor
from src.models.baseline import build_model
from src.evaluation.evaluator import Evaluator
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(model_type: str = "lr") -> dict:
    """
    Ejecuta el pipeline completo de ML.

    Args:
        model_type: "lr" | "rf" | "xgboost"

    Returns:
        Dict con las métricas del modelo entrenado.
    """
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("=" * 60)
    logger.info(f"🚀 PIPELINE — model={model_type} | run={run_id}")
    logger.info("=" * 60)

    # ── Cargar configuración ──────────────────────────────────────────────────
    cfg_pipe = yaml.safe_load(open(PROJECT_ROOT / "configs" / "pipeline.yaml"))
    cfg_feat = yaml.safe_load(open(PROJECT_ROOT / "configs" / "features.yaml"))

    TARGET     = cfg_pipe["data"]["target_binary"]
    RAND       = cfg_pipe["pipeline"]["random_state"]
    TEST_SIZE  = cfg_pipe["pipeline"]["test_size"]
    CV_FOLDS   = cfg_pipe["pipeline"]["cv_folds"]
    RAW_PATH   = PROJECT_ROOT / cfg_pipe["data"]["raw_path"]
    MODELS_DIR = PROJECT_ROOT / "models"
    #MODELS_DIR.mkdir(exist_ok=True)
    # Carpeta segura para experimentos
    EXPERIMENTS_DIR = MODELS_DIR / "experiments" / model_type
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── FASE 1: Carga de datos ────────────────────────────────────────────────
    logger.info("FASE 1 — Carga de datos")
    df = load_raw_data(RAW_PATH)
    logger.info(f"  {len(df)} comentarios cargados")

    # ── FASE 2: Split ─────────────────────────────────────────────────────────
    logger.info("FASE 2 — Split train/test")
    X = df["Text"]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RAND, stratify=y
    )
    logger.info(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # ── FASE 3: Preprocesamiento ──────────────────────────────────────────────
    logger.info("FASE 3 — Preprocesamiento NLP")
    preprocessor = TextPreprocessor(
        config_path=str(PROJECT_ROOT / "configs" / "features.yaml")
    )
    X_train_clean = preprocessor.transform(X_train)
    X_test_clean  = preprocessor.transform(X_test)

    # Reemplazar vacíos con texto original (evitar pérdida de muestras)
    X_train_clean = X_train_clean.where(X_train_clean != "", X_train)
    X_test_clean  = X_test_clean.where(X_test_clean != "", X_test)

    logger.info(f"  Preprocesamiento completado")

    # ── FASE 4: Entrenamiento ─────────────────────────────────────────────────
    logger.info(f"FASE 4 — Entrenamiento ({model_type.upper()})")
    model = build_model(
        model_type,
        config_path      = str(PROJECT_ROOT / "configs" / "models.yaml"),
        feat_config_path = str(PROJECT_ROOT / "configs" / "features.yaml"),
        best_params_path = str(PROJECT_ROOT / "configs" / "best_params.yaml"),
    )
    model.fit(X_train_clean, y_train)

    # ── FASE 5: Cross-validation ──────────────────────────────────────────────
    logger.info(f"FASE 5 — Cross-validation ({CV_FOLDS} folds)")
    cv_results = model.cross_validate(X_train_clean, y_train, cv_folds=CV_FOLDS, rand=RAND)

    # ── FASE 6: Evaluación en test ────────────────────────────────────────────
    logger.info("FASE 6 — Evaluación en test")
    evaluator = Evaluator(output_dir=PROJECT_ROOT / "reports" / "v2" / "pipeline")

    y_pred  = model.predict(X_test_clean)
    y_proba = model.predict_proba(X_test_clean)[:, 1]

    metrics = evaluator.evaluate(
        model, X_test_clean, y_test,
        model_name  = model_type.upper(),
        X_train     = X_train_clean,
        y_train     = y_train,
        cv_results  = cv_results,
    )

    # Visualizaciones
    evaluator.plot_confusion_matrix(y_test, y_pred, model_type.upper())
    evaluator.plot_roc_curve(y_test, y_proba, model_type.upper())
    evaluator.error_analysis(X_test_clean, y_test, y_pred, y_proba)

    # ── FASE 7: Guardado del modelo ───────────────────────────────────────────
    logger.info("FASE 7 — Guardado del modelo")
    model_path = EXPERIMENTS_DIR / f"{model_type}_pipeline_{run_id}.joblib"
    model.save(model_path)

    """
    # Actualizar final_model.joblib si es el modelo por defecto del proyecto
    final_path = MODELS_DIR / "pipeline" / "final_model.joblib"
    model.save(final_path)
    logger.info(f"  Modelo de producción actualizado: {final_path}")
    """

    # ── FASE 8: MLflow ────────────────────────────────────────────────────────
    logger.info("FASE 8 — Registro en MLflow")
    _log_mlflow(metrics, cv_results, model, model_path, run_id, model_type)

    # ── FASE 9: Informe ───────────────────────────────────────────────────────
    logger.info("FASE 9 — Generando informes")
    metrics["run_id"]    = run_id
    metrics["model_path"]= str(model_path)
    evaluator.save_report(metrics, f"exp_{run_id}_{model_type}")
    evaluator.save_summary([metrics])

    logger.info("=" * 60)
    logger.info(f"✅ Pipeline completado — F1={metrics['f1_weighted']:.4f}")
    logger.info("=" * 60)

    return metrics


# ── MLflow logging ────────────────────────────────────────────────────────────
def _log_mlflow(metrics, cv_results, model, model_path, run_id, model_type):
    """Registra el experimento en MLflow."""
    try:
        mlflow_dir = PROJECT_ROOT / "mlruns"
        mlflow.set_tracking_uri(f"file://{mlflow_dir}")
        mlflow.set_experiment("Youtube_project_experiment_pipeline")

        with mlflow.start_run(run_name=f"{model_type}_{run_id}"):
            # Parámetros
            mlflow.log_param("model_type",  model_type)
            mlflow.log_param("run_id",      run_id)

            # Métricas del pipeline
            mlflow.log_metric("test_f1",          metrics["f1_weighted"])
            mlflow.log_metric("test_roc_auc",      metrics["roc_auc"])
            mlflow.log_metric("test_fp",           metrics["fp"])
            mlflow.log_metric("test_fn",           metrics["fn"])
            mlflow.log_metric("cv_f1_mean",        cv_results["cv_f1_mean"])
            mlflow.log_metric("cv_f1_std",         cv_results["cv_f1_std"])
            mlflow.log_metric("train_test_gap_pp", metrics.get("train_test_gap_pp", 0))

            if "cv_test_gap_pp" in metrics:
                mlflow.log_metric("cv_test_gap_pp", metrics["cv_test_gap_pp"])

            # Modelo como artefacto
            mlflow.sklearn.log_model(model.pipeline, f"{model_type}_pipeline")

            logger.info(f"  MLflow run registrado: {model_type}_{run_id}")

    except Exception as e:
        logger.warning(f"MLflow no disponible: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline de entrenamiento — YouTube Hate Speech Detection"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lr",
        choices=["lr", "rf", "xgboost"],
        help="Tipo de modelo a entrenar (default: lr)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    metrics = run_pipeline(model_type=args.model)
    return metrics


if __name__ == "__main__":
    main()
