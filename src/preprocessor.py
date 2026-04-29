import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """Encodes labels, fills missing/infinite values, and splits the dataset."""

    def __init__(
        self,
        label_col: str = "Label",
        test_size: float = 0.2,
        random_state: int = 0,
    ):
        self.label_col = label_col
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoder = LabelEncoder()

    def encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.label_col] = self.label_encoder.fit_transform(df[self.label_col])
        return df

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if df.isnull().values.any() or np.isinf(df.select_dtypes(include="number")).values.any():
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True)
        return df

    def split(self, df: pd.DataFrame):
        X = df.drop([self.label_col], axis=1).values
        y = np.ravel(df[self.label_col].values)
        return train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

    def preprocess(self, df: pd.DataFrame):
        """Convenience method: encode → clean → split."""
        df = self.encode_labels(df)
        df = self.handle_missing(df)
        return self.split(df)

    @property
    def classes(self):
        return self.label_encoder.classes_
