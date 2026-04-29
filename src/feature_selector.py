from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List


class FeatureSelector:
    """
    Automated feature selection using averaged feature importances from the
    top-performing tree-based models.  Selects the minimal feature set that
    accumulates to `importance_threshold` of total importance.
    """

    def __init__(self, importance_threshold: float = 0.9):
        self.importance_threshold = importance_threshold
        self.selected_features: List[str] = []
        self.avg_importance: np.ndarray | None = None
        self.feature_names: List[str] = []

    def fit(
        self,
        top_models: List[str],
        trained_models: Dict,
        feature_names: List[str],
    ) -> "FeatureSelector":
        """
        Args:
            top_models: ordered list of model keys (e.g. ['rf', 'et', 'xg'])
            trained_models: dict mapping key → fitted sklearn-compatible model
            feature_names: column names matching the training data
        """
        self.feature_names = list(feature_names)
        importances = []
        for name in top_models:
            imp = trained_models[name].feature_importances_.copy().astype(float)
            # LightGBM and CatBoost may not sum to 1 by default
            if imp.sum() > 0:
                imp = imp / imp.sum()
            importances.append(imp)

        self.avg_importance = np.mean(importances, axis=0)
        f_list = sorted(
            zip(self.avg_importance, self.feature_names), key=lambda x: x[0], reverse=True
        )

        cumsum, self.selected_features = 0.0, []
        for score, feat in f_list:
            cumsum += score
            self.selected_features.append(feat)
            if cumsum >= self.importance_threshold:
                break
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.selected_features]

    def fit_transform(
        self,
        top_models: List[str],
        trained_models: Dict,
        feature_names: List[str],
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        self.fit(top_models, trained_models, feature_names)
        return self.transform(df)

    def plot(self, title: str = "Average Feature Importance", save_path: str | None = None):
        if self.avg_importance is None:
            raise RuntimeError("Call fit() first.")

        feat_df = pd.DataFrame(
            {"Feature": self.feature_names, "Importance": self.avg_importance}
        ).sort_values("Importance", ascending=False)
        feat_df["Cumulative"] = feat_df["Importance"].cumsum()
        selected_df = feat_df[feat_df["Feature"].isin(self.selected_features)]

        plt.figure(figsize=(13, 6))
        sc = plt.scatter(
            x="Feature", y="Importance", s=200,
            c="Importance", cmap="viridis", alpha=0.7, data=selected_df,
        )
        plt.colorbar(sc, label="Normalized Importance")
        plt.xticks(rotation=45, ha="right")
        plt.title(title)
        plt.ylabel("Normalized Feature Importance")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
