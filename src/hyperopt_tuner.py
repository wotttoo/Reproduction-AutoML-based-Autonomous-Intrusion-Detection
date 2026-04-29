import numpy as np
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from typing import Dict, Tuple


class HyperparameterTuner:
    """
    Bayesian Optimization with Tree-structured Parzen Estimator (BO-TPE)
    via hyperopt.  Supports all six base classifiers.
    """

    _SPACES = {
        "dt": {
            "max_depth":         hp.quniform("max_depth",         1,   50, 1),
            "min_samples_split": hp.quniform("min_samples_split", 2,   11, 1),
            "min_samples_leaf":  hp.quniform("min_samples_leaf",  1,   11, 1),
            "criterion":         hp.choice("criterion", [0, 1]),
        },
        "rf": {
            "n_estimators":      hp.quniform("n_estimators",      10, 200, 1),
            "max_depth":         hp.quniform("max_depth",          5,  50, 1),
            "max_features":      hp.quniform("max_features",       1,  40, 1),
            "min_samples_split": hp.quniform("min_samples_split",  2,  11, 1),
            "min_samples_leaf":  hp.quniform("min_samples_leaf",   1,  11, 1),
            "criterion":         hp.choice("criterion", [0, 1]),
        },
        "et": {
            "n_estimators":      hp.quniform("n_estimators",      10, 200, 1),
            "max_depth":         hp.quniform("max_depth",          1,  50, 1),
            "max_features":      hp.quniform("max_features",       1,  20, 1),
            "min_samples_split": hp.quniform("min_samples_split",  2,  11, 1),
            "min_samples_leaf":  hp.quniform("min_samples_leaf",   1,  11, 1),
            "criterion":         hp.choice("criterion", [0, 1]),
        },
        "xg": {
            "n_estimators":  hp.quniform("n_estimators",  10, 100, 5),
            "max_depth":     hp.quniform("max_depth",      4, 100, 1),
            "learning_rate": hp.normal("learning_rate", 0.01, 0.9),
        },
        "lgbm": {
            "n_estimators":      hp.quniform("n_estimators",      10, 100,  5),
            "max_depth":         hp.quniform("max_depth",          4, 100,  1),
            "learning_rate":     hp.normal("learning_rate",     0.01,  0.9),
            "num_leaves":        hp.quniform("num_leaves",        10, 200,  5),
            "min_child_samples": hp.quniform("min_child_samples",  5, 100,  5),
        },
        "cat": {
            "depth":         hp.quniform("depth",          2,   12,   1),
            "learning_rate": hp.uniform("learning_rate", 0.01, 0.5),
            "iterations":    hp.quniform("iterations",   100, 1000, 100),
        },
    }

    def __init__(self, max_evals: int = 20, random_state: int = 0, cv: int = 3):
        self.max_evals = max_evals
        self.random_state = random_state
        self.cv = cv
        self.best_params: Dict = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(name: str, raw: dict) -> dict:
        criteria = ["gini", "entropy"]
        if name == "dt":
            return {
                "max_depth":         int(raw["max_depth"]),
                "min_samples_split": int(raw["min_samples_split"]),
                "min_samples_leaf":  int(raw["min_samples_leaf"]),
                "criterion":         criteria[int(raw["criterion"])],
            }
        if name in ("rf", "et"):
            return {
                "n_estimators":      int(raw["n_estimators"]),
                "max_depth":         int(raw["max_depth"]),
                "max_features":      int(raw["max_features"]),
                "min_samples_split": int(raw["min_samples_split"]),
                "min_samples_leaf":  int(raw["min_samples_leaf"]),
                "criterion":         criteria[int(raw["criterion"])],
            }
        if name == "xg":
            return {
                "n_estimators":  int(raw["n_estimators"]),
                "max_depth":     int(raw["max_depth"]),
                "learning_rate": abs(float(raw["learning_rate"])),
            }
        if name == "lgbm":
            return {
                "n_estimators":      int(raw["n_estimators"]),
                "max_depth":         int(raw["max_depth"]),
                "learning_rate":     abs(float(raw["learning_rate"])),
                "num_leaves":        int(raw["num_leaves"]),
                "min_child_samples": int(raw["min_child_samples"]),
            }
        if name == "cat":
            return {
                "depth":         int(raw["depth"]),
                "learning_rate": float(raw["learning_rate"]),
                "iterations":    int(raw["iterations"]),
            }
        raise ValueError(f"Unknown model: {name}")

    def _build(self, name: str, params: dict):
        rs = self.random_state
        if name == "dt":
            return DecisionTreeClassifier(random_state=rs, **params)
        if name == "rf":
            return RandomForestClassifier(random_state=rs, **params)
        if name == "et":
            return ExtraTreesClassifier(random_state=rs, **params)
        if name == "xg":
            return xgb.XGBClassifier(
                use_label_encoder=False, eval_metric="mlogloss", random_state=rs, **params
            )
        if name == "lgbm":
            return lgb.LGBMClassifier(random_state=rs, verbose=-1, **params)
        if name == "cat":
            return cb.CatBoostClassifier(random_state=rs, verbose=False, **params)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tune(
        self,
        name: str,
        X_train, y_train,
        X_val=None, y_val=None,
    ) -> Tuple[object, dict]:
        """
        Run BO-TPE for `name`.  If X_val/y_val are provided the objective uses
        hold-out accuracy (faster for boosters); otherwise it uses CV.
        Returns (fitted_model, best_params).
        """
        space = self._SPACES[name]

        def objective(raw):
            params = self._parse(name, raw)
            model = self._build(name, params)
            if X_val is not None:
                model.fit(X_train, y_train)
                score = accuracy_score(y_val, model.predict(X_val))
            else:
                score = cross_val_score(
                    model, X_train, y_train,
                    scoring="accuracy",
                    cv=StratifiedKFold(n_splits=self.cv),
                ).mean()
            return {"loss": -score, "status": STATUS_OK}

        best_raw = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            verbose=False,
        )
        best = self._parse(name, best_raw)
        self.best_params[name] = best

        model = self._build(name, best)
        model.fit(X_train, y_train)
        print(f"  [{name:5s}]  best params = {best}")
        return model, best

    def tune_all(
        self,
        model_names,
        X_train, y_train,
        X_val=None, y_val=None,
    ) -> Dict:
        """Tune every model in *model_names* and return a dict of fitted models."""
        tuned = {}
        for name in model_names:
            model, _ = self.tune(name, X_train, y_train, X_val, y_val)
            tuned[name] = model
        return tuned
