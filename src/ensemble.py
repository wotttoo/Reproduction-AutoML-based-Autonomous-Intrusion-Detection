import time
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from typing import Dict, List, Tuple


class ModelSelector:
    """Ranks base models by mean CV accuracy and selects the top-k."""

    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self.ranked: List[str] = []
        self.top_models: List[str] = []
        self.mean_scores: Dict[str, float] = {}

    def fit(self, cv_scores: Dict[str, np.ndarray]) -> "ModelSelector":
        self.mean_scores = {name: float(np.mean(s)) for name, s in cv_scores.items()}
        self.ranked = sorted(self.mean_scores, key=self.mean_scores.get, reverse=True)
        self.top_models = self.ranked[: self.top_k]
        print("Model ranking (by mean CV accuracy):")
        for rank, name in enumerate(self.ranked, 1):
            marker = " <--" if name in self.top_models else ""
            print(f"  {rank}. {name:5s}  {self.mean_scores[name]:.4f}{marker}")
        return self


class EnsembleBuilder:
    """
    Builds three stacking ensemble variants using the top-k base models:
      1. Traditional Stacking  – stacks hard predictions
      2. Confidence-based Stacking (CBS) – stacks softmax probabilities
      3. Hybrid Stacking (OCSE) – stacks predictions + probabilities
    Meta-learner: LightGBM (fast, regularised, handles mixed inputs well).
    """

    def __init__(self, random_state: int = 0):
        self.random_state = random_state
        self.fitted_ensembles: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _train_meta(
        self,
        x_train: np.ndarray,
        x_test: np.ndarray,
        y_train,
        y_test,
        label: str,
    ) -> Tuple[object, np.ndarray]:
        t0 = time.time()
        meta = lgb.LGBMClassifier(random_state=self.random_state, verbose=-1)
        meta.fit(x_train, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        y_pred = meta.predict(x_test)
        pred_ms = (time.time() - t0) / len(x_test) * 1000

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
        print(f"\n[{label}]")
        print(f"  Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
        print(f"  Train={train_time:.2f}s  Predict={pred_ms:.4f}ms/sample")
        print(classification_report(y_test, y_pred))

        self.fitted_ensembles[label] = meta
        return meta, y_pred

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def traditional_stacking(
        self,
        predictions: Dict,
        top_models: List[str],
        y_train, y_test,
    ) -> Tuple[object, np.ndarray]:
        """Stacks hard (argmax) predictions of the top-k models."""
        x_train = np.concatenate(
            [predictions[m]["train"].reshape(-1, 1) for m in top_models], axis=1
        )
        x_test = np.concatenate(
            [predictions[m]["test"].reshape(-1, 1) for m in top_models], axis=1
        )
        return self._train_meta(x_train, x_test, y_train, y_test, "Traditional Stacking")

    def confidence_stacking(
        self,
        predictions: Dict,
        top_models: List[str],
        y_train, y_test,
    ) -> Tuple[object, np.ndarray]:
        """Stacks predicted class probabilities of the top-k models."""
        x_train = np.concatenate(
            [predictions[m]["prob_train"] for m in top_models], axis=1
        )
        x_test = np.concatenate(
            [predictions[m]["prob_test"] for m in top_models], axis=1
        )
        return self._train_meta(x_train, x_test, y_train, y_test, "Confidence-based Stacking")

    def hybrid_stacking(
        self,
        predictions: Dict,
        top_models: List[str],
        y_train, y_test,
    ) -> Tuple[object, np.ndarray]:
        """Stacks both hard predictions and probabilities (OCSE)."""
        x_train = np.concatenate(
            [predictions[m]["train"].reshape(-1, 1) for m in top_models]
            + [predictions[m]["prob_train"] for m in top_models],
            axis=1,
        )
        x_test = np.concatenate(
            [predictions[m]["test"].reshape(-1, 1) for m in top_models]
            + [predictions[m]["prob_test"] for m in top_models],
            axis=1,
        )
        return self._train_meta(x_train, x_test, y_train, y_test, "Hybrid Stacking (OCSE)")
