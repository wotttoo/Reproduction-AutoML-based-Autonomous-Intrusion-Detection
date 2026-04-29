# API Reference

All public classes live in the `src/` package and are re-exported from `src/__init__.py`.

```python
from src import (
    DataLoader, DataPreprocessor, FeatureSelector,
    DataBalancer, ModelTrainer, HyperparameterTuner,
    ModelSelector, EnsembleBuilder, ModelEvaluator,
)
```

---

## `DataLoader`

```python
DataLoader(filepath: str, label_col: str = "Label")
```

| Method | Returns | Description |
|--------|---------|-------------|
| `load()` | `pd.DataFrame` | Read CSV from `filepath` |
| `class_distribution()` | `pd.Series` | Value counts of label column |
| `summary()` | `None` | Print shape, missing values, class distribution |

---

## `DataPreprocessor`

```python
DataPreprocessor(
    label_col: str = "Label",
    test_size: float = 0.2,
    random_state: int = 0,
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `encode_labels(df)` | `pd.DataFrame` | Fit LabelEncoder on label column |
| `handle_missing(df)` | `pd.DataFrame` | Replace inf → NaN → 0 |
| `split(df)` | `X_train, X_test, y_train, y_test` | Stratified train/test split |
| `preprocess(df)` | `X_train, X_test, y_train, y_test` | encode → clean → split |
| `.classes` | `np.ndarray` | Original class labels before encoding |

---

## `FeatureSelector`

```python
FeatureSelector(importance_threshold: float = 0.9)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `fit(top_models, trained_models, feature_names)` | `self` | Compute avg importance and select features |
| `transform(df)` | `pd.DataFrame` | Return df with selected columns only |
| `fit_transform(top_models, trained_models, feature_names, df)` | `pd.DataFrame` | fit + transform |
| `plot(title, save_path)` | `None` | Scatter plot of selected feature importances |

**Attributes:** `selected_features: List[str]`, `avg_importance: np.ndarray`

---

## `DataBalancer`

```python
DataBalancer(label_col: str = "Label")
```

| Method | Returns | Description |
|--------|---------|-------------|
| `fit_resample(X_train, y_train, metadata_csv_path)` | `X_balanced, y_balanced` | TVAE-based minority class oversampling |

`metadata_csv_path` must point to the feature-selected CSV (with label column) used to build SDV `SingleTableMetadata`.

---

## `ModelTrainer`

```python
ModelTrainer(random_state: int = 0, cv: int = 3)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `train_one(name, X_train, y_train, X_test, params)` | `model` | Train a single model |
| `train_all(X_train, y_train, X_test, model_params)` | `self` | Train all six base models |
| `replace_model(name, model, X_train, X_test, cv_score)` | `None` | Swap in a tuned model |

**Model keys:** `"dt"`, `"rf"`, `"et"`, `"xg"`, `"lgbm"`, `"cat"`

**Attributes after training:**
- `trained_models: Dict[str, model]`
- `cv_scores: Dict[str, np.ndarray]`
- `predictions: Dict[str, {"train", "test", "prob_train", "prob_test"}]`
- `timings: Dict[str, {"train_s", "predict_ms"}]`

---

## `HyperparameterTuner`

```python
HyperparameterTuner(max_evals: int = 20, random_state: int = 0, cv: int = 3)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `tune(name, X_train, y_train, X_val, y_val)` | `(model, best_params)` | BO-TPE for one model |
| `tune_all(model_names, X_train, y_train, X_val, y_val)` | `Dict[str, model]` | Tune multiple models |

If `X_val`/`y_val` are provided, the objective uses hold-out accuracy (faster for boosters); otherwise k-fold CV is used.

**Attribute:** `best_params: Dict[str, dict]`

---

## `ModelSelector`

```python
ModelSelector(top_k: int = 3)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `fit(cv_scores)` | `self` | Rank models and select top-k |

**Attributes:** `top_models: List[str]`, `mean_scores: Dict[str, float]`

---

## `EnsembleBuilder`

```python
EnsembleBuilder(random_state: int = 0)
```

All methods return `(fitted_meta_model, y_predictions)`.

| Method | Ensemble type |
|--------|--------------|
| `traditional_stacking(predictions, top_models, y_train, y_test)` | Hard predictions |
| `confidence_stacking(predictions, top_models, y_train, y_test)` | Probabilities |
| `hybrid_stacking(predictions, top_models, y_train, y_test)` | Predictions + probabilities (OCSE) |

`predictions` is the `ModelTrainer.predictions` dict.

---

## `ModelEvaluator`

```python
ModelEvaluator(output_dir: str = "output")
```

| Method | Returns | Description |
|--------|---------|-------------|
| `evaluate(model, X_test, y_test, model_name, save)` | `dict` | Metrics + optional text report |
| `plot_confusion_matrix(y_test, y_pred, model_name, save)` | `None` | Heatmap; saved to `output/plots/` |
| `compare_models(results, save)` | `pd.DataFrame` | Bar chart + CSV comparison |

The returned dict from `evaluate()` has keys: `accuracy`, `precision`, `recall`, `f1`, `predictions`.
