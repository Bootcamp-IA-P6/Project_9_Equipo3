"""
src/evaluation/evaluator.py

Evaluación estandarizada de modelos.
Genera métricas, visualizaciones e informes JSON.

Uso:
    evaluator = Evaluator(output_dir="reports/pipeline")
    metrics = evaluator.evaluate(model, X_test, y_test, model_name="LR")
    evaluator.error_analysis(X_test, y_test, preds, probs)
    evaluator.save_summary(all_metrics, path="reports/summary.csv")
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, accuracy_score,
    confusion_matrix, classification_report,
    RocCurveDisplay,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """
    Evaluador estandarizado de modelos de clasificación binaria.

    Genera:
    - Métricas completas (F1, Precision, Recall, ROC-AUC)
    - Ambas métricas de gap (train-test y CV-test)
    - Matriz de confusión (PNG)
    - Curva ROC (PNG)
    - Análisis de errores (FP y FN más comunes)
    - Informe JSON por experimento
    - CSV resumen de todos los experimentos
    """

    def __init__(self, output_dir: str | Path = "reports/pipeline"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Evaluación principal ─────────────────────────────────────────────────
    def evaluate(
        self,
        model,
        X_test,
        y_test,
        model_name: str,
        X_train=None,
        y_train=None,
        cv_results: dict = None,
    ) -> dict:
        """
        Evalúa un modelo sobre el test set.

        Args:
            model: objeto con método predict() y predict_proba()
            X_test, y_test: datos de test
            model_name: nombre para los reports
            X_train, y_train: opcional — para calcular train_test_gap
            cv_results: opcional — dict con cv_f1_mean para calcular cv_test_gap

        Returns:
            Dict con todas las métricas.
        """
        logger.info(f"Evaluando: {model_name}")

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_test_arr = np.array(y_test)

        # ── Métricas test ────────────────────────────────────────────────────
        metrics = {
            "model"      : model_name,
            "timestamp"  : datetime.now().isoformat(),
            "f1_weighted": round(f1_score(y_test_arr, y_pred, average="weighted"), 4),
            "f1_toxic"   : round(f1_score(y_test_arr, y_pred, pos_label=1), 4),
            "precision"  : round(precision_score(y_test_arr, y_pred, average="weighted"), 4),
            "recall"     : round(recall_score(y_test_arr, y_pred, average="weighted"), 4),
            "accuracy"   : round(accuracy_score(y_test_arr, y_pred), 4),
            "roc_auc"    : round(roc_auc_score(y_test_arr, y_proba), 4),
            "fp"         : int(((y_test_arr == 0) & (y_pred == 1)).sum()),
            "fn"         : int(((y_test_arr == 1) & (y_pred == 0)).sum()),
            "n_test"     : len(y_test_arr),
        }

        # ── Train-test gap (in-sample vs OOS) ────────────────────────────────
        if X_train is not None and y_train is not None:
            y_train_pred = model.predict(X_train)
            f1_train = f1_score(np.array(y_train), y_train_pred, average="weighted")
            metrics["f1_train"]         = round(f1_train, 4)
            metrics["train_test_gap_pp"]= round((f1_train - metrics["f1_weighted"]) * 100, 2)

        # ── CV-test gap (OOS vs OOS — métrica correcta para la rúbrica) ──────
        if cv_results and "cv_f1_mean" in cv_results:
            cv_mean = cv_results["cv_f1_mean"]
            metrics["cv_f1_mean"]    = round(cv_mean, 4)
            metrics["cv_f1_std"]     = round(cv_results.get("cv_f1_std", 0), 4)
            metrics["cv_test_gap_pp"]= round(abs(cv_mean - metrics["f1_weighted"]) * 100, 2)

        self._print_summary(metrics)
        return metrics

    # ── Visualizaciones ──────────────────────────────────────────────────────
    def plot_confusion_matrix(
        self,
        y_test,
        y_pred,
        model_name: str,
        save: bool = True,
    ) -> Path | None:
        """Genera y guarda la matriz de confusión."""
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["No tóxico", "Tóxico"],
            yticklabels=["No tóxico", "Tóxico"],
            linewidths=0.5,
        )
        ax.set_title(f"{model_name} — Confusion Matrix", fontweight="bold")
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        plt.tight_layout()

        if save:
            safe = model_name.lower().replace(" ", "_").replace("/", "_")
            path = self.output_dir / f"cm_{safe}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.show()
            logger.info(f"Confusion matrix guardada: {path}")
            return path

        plt.show()
        return None

    def plot_roc_curve(
        self,
        y_test,
        y_proba,
        model_name: str,
        save: bool = True,
    ) -> Path | None:
        """Genera y guarda la curva ROC."""
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(
            y_test, y_proba, ax=ax, name=model_name, color="#7F77DD"
        )
        ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="Random")
        ax.set_title(f"{model_name} — Curva ROC", fontweight="bold")
        ax.legend()
        plt.tight_layout()

        if save:
            safe = model_name.lower().replace(" ", "_").replace("/", "_")
            path = self.output_dir / f"roc_{safe}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.show()
            logger.info(f"Curva ROC guardada: {path}")
            return path

        plt.show()
        return None

    # ── Análisis de errores ──────────────────────────────────────────────────
    def error_analysis(
        self,
        X_test,
        y_test,
        y_pred,
        y_proba,
        n_examples: int = 5,
    ) -> dict:
        """
        Analiza los falsos positivos y falsos negativos más relevantes.

        FP → comentarios OK que el modelo censura (peor UX)
        FN → hate speech que se escapa (peor para el objetivo del proyecto)
        """
        texts = np.array(X_test) if not isinstance(X_test, np.ndarray) else X_test
        y_arr = np.array(y_test)

        error_df = pd.DataFrame({
            "text"      : texts,
            "real"      : y_arr,
            "pred"      : y_pred,
            "prob_toxic": y_proba,
        })

        fp = error_df[(error_df["real"] == 0) & (error_df["pred"] == 1)]
        fn = error_df[(error_df["real"] == 1) & (error_df["pred"] == 0)]

        logger.info(f"Errores: FP={len(fp)} | FN={len(fn)}")

        print(f"\n{'='*65}")
        print(f"FALSOS NEGATIVOS — hate speech que NO detectó ({len(fn)} total)")
        print(f"{'='*65}")
        for _, row in fn.nsmallest(n_examples, "prob_toxic").iterrows():
            print(f"  Prob: {row['prob_toxic']:.3f} | {row['text'][:110]}")
            print()

        print(f"{'='*65}")
        print(f"FALSOS POSITIVOS — comentarios OK censurados ({len(fp)} total)")
        print(f"{'='*65}")
        for _, row in fp.nlargest(n_examples, "prob_toxic").iterrows():
            print(f"  Prob: {row['prob_toxic']:.3f} | {row['text'][:110]}")
            print()

        return {"fp_examples": fp.head(n_examples).to_dict("records"),
                "fn_examples": fn.head(n_examples).to_dict("records")}

    # ── Reports ──────────────────────────────────────────────────────────────
    def save_report(self, metrics: dict, experiment_id: str) -> Path:
        """Guarda las métricas de un experimento en JSON."""
        path = self.output_dir / f"{experiment_id}.json"
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Report guardado: {path}")
        return path

    def save_summary(self, all_metrics: list[dict], path: str | Path = None) -> Path:
        """
        Guarda un CSV con todos los experimentos para comparar.
        Este es el 'reports/summary.csv' que mencionaba el roadmap.
        """
        path = Path(path or self.output_dir / "summary.csv")
        df = pd.DataFrame(all_metrics)

        # Ordenar por F1 descendente
        if "f1_weighted" in df.columns:
            df = df.sort_values("f1_weighted", ascending=False)

        df.to_csv(path, index=False)
        logger.info(f"Summary guardado: {path}")
        print(df[["model", "f1_weighted", "roc_auc", "fp", "fn"]].to_string(index=False))
        return path

    # ── Interno ──────────────────────────────────────────────────────────────
    def _print_summary(self, metrics: dict) -> None:
        gap_str = ""
        if "cv_test_gap_pp" in metrics:
            ok  = "✅" if metrics["cv_test_gap_pp"] < 5 else "⚠️"
            gap_str = f"CV-test gap: {metrics['cv_test_gap_pp']:.2f}pp {ok}"
        elif "train_test_gap_pp" in metrics:
            ok  = "✅" if metrics["train_test_gap_pp"] < 5 else "⚠️"
            gap_str = f"Train-test gap: {metrics['train_test_gap_pp']:.2f}pp {ok}"

        print(f"\n{'='*55}")
        print(f"RESULTADOS — {metrics['model']}")
        print(f"{'='*55}")
        print(f"  F1 weighted : {metrics['f1_weighted']:.4f}")
        print(f"  ROC-AUC     : {metrics['roc_auc']:.4f}")
        print(f"  FP / FN     : {metrics['fp']} / {metrics['fn']}")
        if gap_str:
            print(f"  {gap_str}")
        print(f"{'='*55}")
