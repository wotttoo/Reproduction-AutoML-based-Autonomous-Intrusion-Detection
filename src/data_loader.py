from __future__ import annotations

import pandas as pd
import numpy as np


class DataLoader:
    """Loads a CSV dataset and exposes basic inspection helpers."""

    def __init__(self, filepath: str, label_col: str = "Label"):
        self.filepath = filepath
        self.label_col = label_col
        self.df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.filepath)
        return self.df

    def class_distribution(self) -> pd.Series:
        if self.df is None:
            raise RuntimeError("Call load() before class_distribution().")
        return self.df[self.label_col].value_counts()

    def summary(self) -> None:
        if self.df is None:
            raise RuntimeError("Call load() before summary().")
        print(f"Shape       : {self.df.shape}")
        print(f"Columns     : {list(self.df.columns)}")
        print(f"Missing     : {self.df.isnull().sum().sum()}")
        print(f"Inf values  : {np.isinf(self.df.select_dtypes(include='number')).sum().sum()}")
        print(f"\nClass distribution:\n{self.class_distribution()}")
