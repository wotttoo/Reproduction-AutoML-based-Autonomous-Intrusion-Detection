from __future__ import annotations

import pandas as pd
import numpy as np
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer


class DataBalancer:
    """
    Balances a multi-class dataset using TVAE (Tabular Variational Auto-Encoder).
    Minority classes (those with < half the average class count) receive
    synthetic samples until they reach the class average.
    """

    def __init__(self, label_col: str = "Label"):
        self.label_col = label_col
        self._synthesizers: dict = {}

    def fit_resample(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        metadata_csv_path: str,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Args:
            X_train: feature DataFrame (no label column)
            y_train: label Series
            metadata_csv_path: path to the feature-selected CSV used to build
                               SDV SingleTableMetadata (must include label column)
        Returns:
            X_balanced, y_balanced
        """
        metadata = SingleTableMetadata()
        metadata.detect_from_csv(filepath=metadata_csv_path)

        counts = y_train.value_counts()
        average = counts.mean()
        minority_classes = counts[counts < average / 2].index.tolist()

        synthetic_frames = []
        for cls in minority_classes:
            cls_mask = y_train == cls
            cls_X = X_train[cls_mask].copy()
            cls_y = y_train[cls_mask]
            n_needed = int(average - len(cls_X))

            synthesizer = TVAESynthesizer(metadata=metadata)
            synthesizer.fit(cls_X.assign(**{self.label_col: cls_y}))
            new_rows = synthesizer.sample(n_needed)
            synthetic_frames.append(new_rows)
            self._synthesizers[cls] = synthesizer

        original = X_train.assign(**{self.label_col: y_train})
        balanced = pd.concat([original] + synthetic_frames, ignore_index=True)
        X_balanced = balanced.drop([self.label_col], axis=1)
        y_balanced = balanced[self.label_col]
        return X_balanced, y_balanced
