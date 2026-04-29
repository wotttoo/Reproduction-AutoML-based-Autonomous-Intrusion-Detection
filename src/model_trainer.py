from __future__ import annotations

import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from typing import Dict


class ModelTrainer:
    """
    Trains all six base classifiers (DT, RF, ET, XGBoost, LightGBM, CatBoost)
    with k-fold cross-validation, and records predictions and probabilities for
    downstream ensemble use.
    """

    MODEL_KEYS = ("dt", "rf", "et", "xg", "lgbm", "cat")

    def __init__(self, random_state: int = 0, cv: int = 3):
        self.random_state = random_state
        self.cv = cv
        self.trained_models: Dict = {}
        self.cv_scores: Dict = {}
        self.predictions: Dict = {}
        self.timings: Dict = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self, name: str, params: dict = None):
        rs = self.random_state
        params = params or {}
        mapping = {
            "dt":   lambda: DecisionTreeClassifier(random_state=rs, **params),
            "rf":   lambda: RandomForestClassifier(random_state=rs, **params),
            "et":   lambda: ExtraTreesClassifier(random_state=rs, **params),
            "xg":   lambda: xgb.XGBClassifier(random_state=rs, **params),
            "lgbm": lambda: lgb.LGBMClassifier(random_state=rs, verbose=-1, **params),
            "cat":  lambda: CatBoostClassifier(random_state=rs, verbose=False, **params),
        }
        if name not in mapping:
            raise ValueError(f"Unknown model '{name}'. Choose from {self.MODEL_KEYS}.")
        return mapping[name]()

    def _record(self, name: str, model, X_train, X_test) -> None:
        self.predictions[name] = {
            "train":      model.predict(X_train),
            "test":       model.predict(X_test),
            "prob_train": model.predict_proba(X_train),
            "prob_test":  model.predict_proba(X_test),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_one(self, name: str, X_train, y_train, X_test, params: dict = None):
        """Train and record a single model."""
        model = self._build_model(name, params)
        scores = cross_val_score(model, X_train, y_train, cv=self.cv)
        self.cv_scores[name] = scores

        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        test_pred = model.predict(X_test)
        pred_ms = (time.time() - t0) / len(X_test) * 1000

        self.trained_models[name] = model
        self.timings[name] = {"train_s": train_time, "predict_ms": pred_ms}
        self._record(name, model, X_train, X_test)

        print(
            f"  [{name:5s}]  CV={np.mean(scores):.4f}  "
            f"train={train_time:.2f}s  predict={pred_ms:.4f}ms/sample"
        )
        return model

    def train_all(self, X_train, y_train, X_test, model_params: dict = None):
        """Train all six base models sequentially."""
        model_params = model_params or {}
        for name in self.MODEL_KEYS:
            self.train_one(name, X_train, y_train, X_test, model_params.get(name))
        return self

    def replace_model(self, name: str, model, X_train, X_test, cv_score: float | None = None):
        """Replace an existing model entry (e.g. after hyperparameter tuning)."""
        self.trained_models[name] = model
        if cv_score is not None:
            self.cv_scores[name] = np.array([cv_score])
        self._record(name, model, X_train, X_test)
