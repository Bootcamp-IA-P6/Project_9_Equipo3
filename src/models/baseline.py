"""
src/models/baseline.py

Modelos clásicos de ML para clasificación de texto.
Traducción directa de notebooks 04 y 05.

Todos los modelos siguen la misma interfaz:
    model.fit(X_train, y_train)
    model.predict(X)
    model.predict_proba(X)
    model.save(path)
    Model.load(path)

Uso desde el pipeline:
    model = build_model("lr", config_path="configs/models.yaml")
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)
"""

import yaml
import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_validate
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Clase base ────────────────────────────────────────────────────────────────
class BaseSklearnModel:
    """
    Interfaz común para todos los modelos sklearn del proyecto.
    Hereda LRModel y EnsembleModel.
    """

    def __init__(self):
        self.pipeline = None   # sklearn Pipeline (TF-IDF + clf)
        self.is_fitted = False

    def fit(self, X_train, y_train) -> "BaseSklearnModel":
        """Entrena el pipeline completo."""
        logger.info(f"Entrenando {self.__class__.__name__}...")
        self.pipeline.fit(X_train, y_train)
        self.is_fitted = True
        logger.info("  Entrenamiento completado")
        return self

    def predict(self, X) -> np.ndarray:
        self._check_fitted()
        return self.pipeline.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        self._check_fitted()
        return self.pipeline.predict_proba(X)

    def cross_validate(self, X_train, y_train, cv_folds: int = 5, rand: int = 42) -> dict:
        """
        Evaluación con StratifiedKFold.
        Devuelve medias y desviaciones estándar de las métricas.
        """
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=rand)
        results = cross_validate(
            self.pipeline, X_train, y_train,
            cv=cv,
            scoring={"f1": "f1_weighted", "roc_auc": "roc_auc"},
            return_train_score=True,
            n_jobs=-1,
        )
        summary = {
            "cv_f1_mean"    : results["test_f1"].mean(),
            "cv_f1_std"     : results["test_f1"].std(),
            "cv_roc_mean"   : results["test_roc_auc"].mean(),
            "train_f1_mean" : results["train_f1"].mean(),
            "gap_pp"        : (results["train_f1"].mean() - results["test_f1"].mean()) * 100,
        }
        logger.info(
            f"  CV F1: {summary['cv_f1_mean']:.4f} ± {summary['cv_f1_std']:.4f} | "
            f"Gap: {summary['gap_pp']:.1f}pp"
        )
        return summary

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        logger.info(f"Modelo guardado: {path}")

    @classmethod
    def load(cls, path: str | Path) -> "BaseSklearnModel":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {path}")
        instance = cls.__new__(cls)
        instance.pipeline = joblib.load(path)
        instance.is_fitted = True
        logger.info(f"Modelo cargado: {path}")
        return instance

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("El modelo no está entrenado. Llama a .fit() primero.")


# ── Logistic Regression ────────────────────────────────────────────────────────
class LRModel(BaseSklearnModel):
    """
    Logistic Regression + TF-IDF.

    Mejor modelo del proyecto (notebook 06):
        F1 test = 0.7579 | CV-test gap = 4.76pp
    Parámetros optimizados con Optuna sobre configs/best_params.yaml.
    """

    def __init__(
        self,
        config_path: str = "configs/models.yaml",
        feat_config_path: str = "configs/features.yaml",
        best_params_path: str = "configs/best_params.yaml",
    ):
        super().__init__()

        # Intentar cargar best_params.yaml (resultado de Optuna)
        try:
            import yaml as _yaml
            with open(best_params_path) as f:
                best = _yaml.safe_load(f)
            bp = best.get("hyperparameters", {})
            logger.info("Parámetros cargados desde best_params.yaml")
        except FileNotFoundError:
            bp = {}
            logger.warning("best_params.yaml no encontrado — usando config por defecto")

        # Config base
        with open(config_path) as f:
            mod_cfg = yaml.safe_load(f)["models"]["logistic_regression"]
        with open(feat_config_path) as f:
            vec_cfg = yaml.safe_load(f)["vectorization"]["tfidf"]

        # Prioridad: best_params > yaml config
        ngram_str = str(bp.get("ngram_range", "1_2"))
        ngram     = (1, 1) if ngram_str == "1_1" else (1, 2)

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features  = bp.get("max_features", vec_cfg["max_features"]),
                ngram_range   = ngram,
                sublinear_tf  = bp.get("sublinear_tf", vec_cfg["sublinear_tf"]),
                min_df        = bp.get("min_df", vec_cfg["min_df"]),
                analyzer      = "word",
                strip_accents = "unicode",
            )),
            ("clf", LogisticRegression(
                C            = bp.get("C", mod_cfg["C"]),
                max_iter     = mod_cfg["max_iter"],
                class_weight = mod_cfg["class_weight"],
                solver       = mod_cfg["solver"],
                random_state = 42,
            )),
        ])
        logger.info(f"LRModel creado — C={bp.get('C', mod_cfg['C']):.4f} | ngram={ngram}")


# ── Random Forest ──────────────────────────────────────────────────────────────
class RFModel(BaseSklearnModel):
    """
    Random Forest + TF-IDF.
    Parámetros desde configs/models.yaml.
    """

    def __init__(
        self,
        config_path: str = "configs/models.yaml",
        feat_config_path: str = "configs/features.yaml",
    ):
        super().__init__()

        with open(config_path) as f:
            rf_cfg  = yaml.safe_load(f)["models"]["random_forest"]
        with open(feat_config_path) as f:
            vec_cfg = yaml.safe_load(f)["vectorization"]["tfidf"]

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features  = vec_cfg["max_features"],
                ngram_range   = (1, 1),   # RF + bigramas es muy lento
                sublinear_tf  = vec_cfg["sublinear_tf"],
                min_df        = vec_cfg["min_df"],
                analyzer      = "word",
                strip_accents = "unicode",
            )),
            ("clf", RandomForestClassifier(
                n_estimators     = rf_cfg["n_estimators"],
                max_depth        = rf_cfg.get("max_depth", 8),
                min_samples_leaf = rf_cfg.get("min_samples_leaf", 4),
                max_features     = "sqrt",
                class_weight     = rf_cfg["class_weight"],
                random_state     = 42,
                n_jobs           = -1,
            )),
        ])
        logger.info("RFModel creado")


# ── XGBoost ───────────────────────────────────────────────────────────────────
class XGBModel(BaseSklearnModel):
    """
    XGBoost + TF-IDF.
    Requiere: pip install xgboost
    """

    def __init__(
        self,
        config_path: str = "configs/models.yaml",
        feat_config_path: str = "configs/features.yaml",
    ):
        super().__init__()

        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("Instala XGBoost: pip install xgboost")

        with open(config_path) as f:
            xgb_cfg = yaml.safe_load(f)["models"]["xgboost"]
        with open(feat_config_path) as f:
            vec_cfg = yaml.safe_load(f)["vectorization"]["tfidf"]

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features  = vec_cfg["max_features"],
                ngram_range   = (1, 1),
                sublinear_tf  = True,
                min_df        = vec_cfg["min_df"],
                analyzer      = "word",
                strip_accents = "unicode",
            )),
            ("clf", XGBClassifier(
                n_estimators     = xgb_cfg.get("n_estimators", 200),
                max_depth        = xgb_cfg.get("max_depth", 3),
                learning_rate    = xgb_cfg.get("learning_rate", 0.05),
                subsample        = xgb_cfg.get("subsample", 0.8),
                colsample_bytree = xgb_cfg.get("colsample_bytree", 0.8),
                use_label_encoder= False,
                eval_metric      = "logloss",
                random_state     = 42,
                verbosity        = 0,
            )),
        ])
        logger.info("XGBModel creado")


# ── Factory ───────────────────────────────────────────────────────────────────
def build_model(
    model_type: str,
    config_path: str = "configs/models.yaml",
    feat_config_path: str = "configs/features.yaml",
    best_params_path: str = "configs/best_params.yaml",
) -> BaseSklearnModel:
    """
    Construye el modelo indicado en la configuración.

    Args:
        model_type: "lr" | "rf" | "xgboost"

    Returns:
        Instancia del modelo listo para .fit()
    """
    builders = {
        "lr"     : lambda: LRModel(config_path, feat_config_path, best_params_path),
        "rf"     : lambda: RFModel(config_path, feat_config_path),
        "xgboost": lambda: XGBModel(config_path, feat_config_path),
    }
    if model_type not in builders:
        raise ValueError(f"model_type debe ser uno de: {list(builders.keys())}")

    logger.info(f"Construyendo modelo: {model_type}")
    return builders[model_type]()
