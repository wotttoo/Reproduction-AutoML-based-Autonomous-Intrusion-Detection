import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from typing import Dict


class ModelEvaluator:
    """
    Evaluates trained models, plots confusion matrices, and persists reports
    to the output directory.
    """

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)

    # ------------------------------------------------------------------
    # Single-model evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model,
        X_test,
        y_test,
        model_name: str,
        save: bool = True,
    ) -> Dict:
        t0 = time.time()
        y_pred = model.predict(X_test)
        pred_ms = (time.time() - t0) / len(X_test) * 1000

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
        report = classification_report(y_test, y_pred)

        print(f"\n=== {model_name} ===")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1-score  : {f1:.4f}")
        print(f"  Pred time : {pred_ms:.4f} ms/sample")
        print(report)

        if save:
            path = os.path.join(self.output_dir, "reports", f"{model_name}_report.txt")
            with open(path, "w") as fp:
                fp.write(f"=== {model_name} ===\n")
                fp.write(
                    f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\n"
                    f"Recall: {rec:.4f}\nF1: {f1:.4f}\n\n"
                )
                fp.write(report)

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "predictions": y_pred,
        }

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------

    def plot_confusion_matrix(
        self,
        y_test,
        y_pred,
        model_name: str,
        save: bool = True,
    ) -> None:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {model_name}")
        plt.tight_layout()
        if save:
            path = os.path.join(self.output_dir, "plots", f"{model_name}_cm.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.show()

    # ------------------------------------------------------------------
    # Cross-model comparison
    # ------------------------------------------------------------------

    def compare_models(
        self,
        results: Dict[str, Dict],
        save: bool = True,
    ) -> pd.DataFrame:
        df = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1"]]
        print("\n=== Model Comparison ===")
        print(df.to_string())

        if save:
            df.to_csv(os.path.join(self.output_dir, "reports", "model_comparison.csv"))

        fig, ax = plt.subplots(figsize=(12, 5))
        df.plot(kind="bar", ax=ax, colormap="Set2")
        ax.set_title("Model Performance Comparison")
        ax.set_ylabel("Score")
        ax.set_xticklabels(df.index, rotation=45, ha="right")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        if save:
            plt.savefig(
                os.path.join(self.output_dir, "plots", "model_comparison.png"),
                dpi=150, bbox_inches="tight",
            )
        plt.show()
        return df
