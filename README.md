<div align="center">

# AutoML-IDS

### Autonomous Intrusion Detection System via AutoML

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/github/license/wotttoo/Reproduction-AutoML-based-Autonomous-Intrusion-Detection)](LICENSE)
[![Release](https://img.shields.io/github/v/release/wotttoo/Reproduction-AutoML-based-Autonomous-Intrusion-Detection)](https://github.com/wotttoo/Reproduction-AutoML-based-Autonomous-Intrusion-Detection/releases)
[![Paper](https://img.shields.io/badge/Paper-ACM%20CCS%202024-red)](https://doi.org/10.1145/3689933.3690833)
[![arXiv](https://img.shields.io/badge/arXiv-2409.03141-b31b1b)](https://arxiv.org/abs/2409.03141)

Reproduction of **"Towards Autonomous Cybersecurity: An Intelligent AutoML Framework for Autonomous Intrusion Detection"**  
Li Yang · Abdallah Shami — *AutonomousCyber '24, ACM CCS 2024*

<img src="Framework.jpg" width="720" alt="AutoML-IDS Framework"/>

</div>

---

## Overview

AutoML-IDS is a **fully autonomous machine learning pipeline** for network intrusion detection on 5G/6G traffic. Given a raw network traffic CSV, it requires **zero manual intervention** to produce a trained, evaluated, and saved ensemble classifier.

**Key features:**
- End-to-end pipeline: preprocessing → feature selection → data balancing → training → hyperparameter tuning → ensemble → evaluation
- 6 tree-based classifiers trained and ranked automatically
- Bayesian hyperparameter optimisation (BO-TPE) per model
- TVAE synthetic oversampling for imbalanced minority attack classes
- OCSE stacking ensemble with LightGBM meta-learner
- All artifacts (models, plots, reports) saved with per-dataset namespacing

---

## Results

Benchmarks on the bundled dataset samples, fast mode (`--no-balance --no-tune`).

<details open>
<summary><strong>CICIDS2017</strong> — 55 k rows · 7 classes · 38 / 77 features selected</summary>

| Model | Accuracy | Precision | Recall | F1-score |
|:------|:--------:|:---------:|:------:|:--------:|
| **XGBoost** | **99.76%** | **99.76%** | **99.76%** | **99.76%** |
| CatBoost | 99.73% | 99.73% | 99.73% | 99.72% |
| Random Forest | 99.61% | 99.61% | 99.61% | 99.61% |
| Decision Tree | 99.51% | 99.51% | 99.51% | 99.51% |
| Extra Trees | 99.23% | 99.24% | 99.23% | 99.23% |
| LightGBM | 73.55% | 69.36% | 73.55% | 71.07% |
| Traditional Stacking | 99.66% | 99.66% | 99.66% | 99.65% |
| Confidence Stacking | 99.59% | 99.59% | 99.59% | 99.59% |
| Hybrid Stacking (OCSE) | 92.08% | 92.06% | 92.08% | 91.72% |

</details>

<details open>
<summary><strong>5G-NIDD</strong> — 48 k rows · 9 classes · 17 / 48 features selected</summary>

| Model | Accuracy | Precision | Recall | F1-score |
|:------|:--------:|:---------:|:------:|:--------:|
| **Extra Trees** | **99.92%** | **99.92%** | **99.92%** | **99.92%** |
| **LightGBM** | **99.92%** | **99.92%** | **99.92%** | **99.92%** |
| Random Forest | 99.91% | 99.91% | 99.91% | 99.91% |
| XGBoost | 99.90% | 99.90% | 99.90% | 99.90% |
| Decision Tree | 99.86% | 99.86% | 99.86% | 99.86% |
| CatBoost | 99.86% | 99.87% | 99.86% | 99.86% |
| Traditional Stacking | 99.88% | 99.88% | 99.88% | 99.88% |
| Hybrid Stacking (OCSE) | 98.50% | 98.66% | 98.50% | 98.44% |
| Confidence Stacking | 98.18% | 98.31% | 98.18% | 98.12% |

</details>

> Running the full pipeline (`--balance --tune`) is expected to further improve recall on minority attack classes via TVAE oversampling and BO-TPE tuning.

---

## Project Structure

```
AutoML-IDS/
├── data/
│   ├── raw/
│   │   ├── CICIDS2017_sample_0.02.csv   # 2% stratified sample (~55 k rows)
│   │   └── 5G-NIDD_0.04.csv            # 4% stratified sample (~48 k rows)
│   └── processed/                       # Feature-selected CSVs (auto-generated)
│
├── notebooks/
│   ├── 01_CICIDS2017_Pipeline.ipynb     # Interactive pipeline — CICIDS2017
│   └── 02_5GNIDD_Pipeline.ipynb         # Interactive pipeline — 5G-NIDD
│
├── src/                                 # Reusable Python package
│   ├── data_loader.py                   # DataLoader
│   ├── preprocessor.py                  # DataPreprocessor
│   ├── feature_selector.py              # FeatureSelector
│   ├── data_balancer.py                 # DataBalancer (TVAE)
│   ├── model_trainer.py                 # ModelTrainer
│   ├── hyperopt_tuner.py                # HyperparameterTuner (BO-TPE)
│   ├── ensemble.py                      # ModelSelector + EnsembleBuilder (OCSE)
│   └── evaluator.py                     # ModelEvaluator
│
├── output/                              # Generated artifacts (git-ignored)
│   ├── models/                          # Serialised models (.pkl)
│   ├── plots/                           # Confusion matrices, charts
│   └── reports/                         # Classification reports, CSVs
│
├── docs/
│   ├── project_description.md
│   └── api_reference.md
│
├── run.py                               # CLI entry point
├── requirements.txt
└── Framework.jpg
```

---

## Quick Start

**Requirements:** Python 3.9+ · pip

```bash
# 1. Clone
git clone https://github.com/wotttoo/Reproduction-AutoML-based-Autonomous-Intrusion-Detection.git
cd Reproduction-AutoML-based-Autonomous-Intrusion-Detection

# 2. Create virtual environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run — fast mode (feature selection + training + ensemble)
python run.py --dataset cicids2017 --no-balance --no-tune

# 4. Run — full pipeline (TVAE balancing + BO-TPE tuning, ~1–2 h)
python run.py --dataset cicids2017
```

Results are saved to `output/` with `<dataset>_` prefix on all filenames.

**Interactive notebooks:**
```bash
jupyter lab
# → notebooks/01_CICIDS2017_Pipeline.ipynb
# → notebooks/02_5GNIDD_Pipeline.ipynb
```

---

## AutoML Pipeline

| Step | Stage | Module | Description |
|:----:|-------|--------|-------------|
| 1 | **Pre-processing** | `preprocessor.py` | Label encoding, inf/NaN cleaning, stratified 80/20 split |
| 2 | **Feature Selection** | `feature_selector.py` | Train all 6 classifiers; average top-3 importances; keep features up to 90% cumulative importance |
| 3 | **Data Balancing** | `data_balancer.py` | Identify minority classes; synthesise samples with TVAE |
| 4 | **Model Training** | `model_trainer.py` | Re-train on balanced data; 3-fold CV; rank by mean accuracy |
| 5 | **Hyperparameter Tuning** | `hyperopt_tuner.py` | BO-TPE search; 20 evaluations per model; per-model search space |
| 6 | **Ensemble (OCSE)** | `ensemble.py` | Traditional / Confidence / Hybrid stacking with LightGBM meta-learner |
| 7 | **Evaluation** | `evaluator.py` | Accuracy, Precision, Recall, F1; confusion matrices; comparison chart |

Steps 3, 5, and 6 can be individually skipped via CLI flags.

---

## Key Techniques

<details>
<summary><strong>TVAE — Tabular Variational Auto-Encoder</strong></summary>

A VAE trained on tabular data ([SDV](https://docs.sdv.dev/sdv)) that learns the joint feature distribution and generates statistically coherent synthetic rows for minority attack classes. Unlike SMOTE, TVAE preserves feature correlations and handles mixed data types, producing higher-fidelity samples that improve recall on rare attack categories.

</details>

<details>
<summary><strong>BO-TPE — Bayesian Optimisation with Tree-structured Parzen Estimator</strong></summary>

[Hyperopt](https://github.com/hyperopt/hyperopt) models the objective as a probabilistic surrogate and allocates evaluations toward promising hyperparameter regions. Each classifier has its own search space; the objective maximises 3-fold CV accuracy (tree models) or hold-out accuracy (gradient boosters). Outperforms random and grid search at a fraction of the compute budget.

</details>

<details>
<summary><strong>OCSE — Optimised Confidence-based Stacking Ensemble</strong></summary>

Three stacking variants are built from the top-k base models:

| Variant | Meta-features |
|---------|--------------|
| Traditional Stacking | Hard class predictions |
| Confidence Stacking | Softmax class probabilities |
| **Hybrid Stacking (OCSE)** | **Predictions + probabilities** |

The OCSE meta-learner (LightGBM) receives both hard predictions and calibrated probabilities, preserving confidence signals that pure hard-voting discards. Particularly effective on imbalanced multi-class traffic data.

</details>

---

## Base Classifiers

| Key | Algorithm | Library |
|-----|-----------|---------|
| `dt` | Decision Tree | scikit-learn |
| `rf` | Random Forest | scikit-learn |
| `et` | Extra Trees | scikit-learn |
| `xg` | XGBoost | xgboost |
| `lgbm` | LightGBM | lightgbm |
| `cat` | CatBoost | catboost |

All six are trained and cross-validated every run. The top-3 by CV accuracy proceed to the ensemble stage.

---

## CLI Reference

```
python run.py --dataset {cicids2017,5gnidd} [OPTIONS]

Pipeline control:
  --no-balance        Skip TVAE data balancing
  --no-tune           Skip BO-TPE hyperparameter optimisation
  --no-ensemble       Skip ensemble construction

Tuning:
  --top-k     INT     Models to include in ensemble      (default: 3)
  --max-evals INT     BO-TPE evaluations per model       (default: 20)
  --cv        INT     Cross-validation folds             (default: 3)

Output:
  --output    PATH    Artifact output directory          (default: output)
```

**Examples:**

```bash
# Fast dry-run
python run.py --dataset cicids2017 --no-balance --no-tune

# Full pipeline on 5G-NIDD
python run.py --dataset 5gnidd

# Extended tuning budget, larger ensemble
python run.py --dataset cicids2017 --top-k 5 --max-evals 50

# Custom output path
python run.py --dataset cicids2017 --no-balance --no-tune --output results/exp1
```

---

## Using `src` as a Library

```python
from src import (
    DataLoader, DataPreprocessor, FeatureSelector, DataBalancer,
    ModelTrainer, HyperparameterTuner, ModelSelector, EnsembleBuilder, ModelEvaluator,
)
```

**Minimal example:**

```python
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from src import DataLoader, DataPreprocessor, FeatureSelector, ModelTrainer, ModelSelector, ModelEvaluator

# Load & preprocess
loader = DataLoader("data/raw/CICIDS2017_sample_0.02.csv", label_col="Label")
df = loader.load()
prep = DataPreprocessor(label_col="Label")
df = prep.encode_labels(df)
df = prep.handle_missing(df)

features = [c for c in df.columns if c != "Label"]
X, y = df[features].values, df["Label"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Feature selection
trainer = ModelTrainer(cv=3)
trainer.train_all(X_train, y_train, X_test)

selector = ModelSelector(top_k=3)
selector.fit(trainer.cv_scores)

fs = FeatureSelector(importance_threshold=0.9)
fs.fit(selector.top_models, trainer.trained_models, features)
X_train_fs = df[fs.selected_features].values[:len(X_train)]
X_test_fs  = df[fs.selected_features].values[len(X_train):]

# Evaluate & save
evaluator = ModelEvaluator(output_dir="output", prefix="cicids2017")
for name, model in trainer.trained_models.items():
    res = evaluator.evaluate(model, X_test_fs, y_test, name)
    evaluator.plot_confusion_matrix(y_test, res["predictions"], name)

trainer.save_models(output_dir="output", prefix="cicids2017")

# Load saved model for inference
model = joblib.load("output/models/cicids2017_xg.pkl")
predictions = model.predict(X_test_fs)
```

See [`docs/api_reference.md`](docs/api_reference.md) for full method signatures.

---

## Datasets

| Dataset | Attack Types | Full Size | Included Sample |
|---------|-------------|-----------|-----------------|
| [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) | BENIGN, DoS ×4, PortScan, BruteForce, WebAttack, Bot, Infiltration | ~2.8 M rows | 2% · ~55 k rows |
| [5G-NIDD](https://ieee-dataport.org/documents/5g-nidd-comprehensive-network-intrusion-detection-dataset-generated-over-5g-wireless) | Benign, DDoS ×3, Reconnaissance, Mirai ×3 | ~1.3 M rows | 4% · ~48 k rows |

Stratified samples are committed to `data/raw/` so the pipeline runs out of the box. To use the full datasets, download from the links above and update the `path` entries in the `DATASETS` dict in `run.py`.

---

## Output Artifacts

All files are prefixed with the dataset name to prevent collisions across runs.

```
output/
├── models/
│   ├── <dataset>_{dt,rf,et,xg,lgbm,cat}.pkl
│   ├── <dataset>_ensemble_traditional.pkl
│   ├── <dataset>_ensemble_confidence.pkl
│   └── <dataset>_ensemble_ocse.pkl
├── plots/
│   ├── <dataset>_feature_importance.png
│   ├── <dataset>_{dt,rf,et,xg,lgbm,cat}_cm.png
│   └── <dataset>_model_comparison.png
└── reports/
    ├── <dataset>_{dt,rf,et,xg,lgbm,cat}_report.txt
    └── <dataset>_model_comparison.csv
```

---

## Requirements

| Package | Version | Role |
|---------|---------|------|
| Python | ≥ 3.9 | Runtime |
| numpy | ≥ 2.4 | Numerical arrays |
| pandas | ≥ 2.3 | DataFrames |
| scipy | ≥ 1.17 | Statistical utilities |
| scikit-learn | ≥ 1.8 | DT, RF, ET, metrics |
| joblib | ≥ 1.5 | Model serialisation |
| xgboost | ≥ 3.2 | XGBoost classifier |
| lightgbm | ≥ 4.6 | LightGBM + meta-learner |
| catboost | ≥ 1.2 | CatBoost classifier |
| hyperopt | ≥ 0.2.7 | BO-TPE optimisation |
| sdv | ≥ 1.36 | TVAE data generation |
| matplotlib | ≥ 3.10 | Plotting |
| seaborn | ≥ 0.13 | Heatmaps |
| jupyterlab | ≥ 4.5 | Notebooks |

```bash
pip install -r requirements.txt
```

---

## Citation

```bibtex
@inproceedings{yang2024autonomouscyber,
  author    = {Yang, Li and Shami, Abdallah},
  title     = {Towards Autonomous Cybersecurity: An Intelligent AutoML Framework
               for Autonomous Intrusion Detection},
  booktitle = {Proceedings of the Workshop on Autonomous Cybersecurity
               (AutonomousCyber '24), ACM CCS 2024},
  year      = {2024},
  pages     = {1--11},
  doi       = {10.1145/3689933.3690833}
}
```

---

<div align="center">

**Original authors:** Li Yang · Abdallah Shami — ANTS Lab (Ontario Tech) / OC2 Lab (Western University)

For issues with this reproduction, please open a [GitHub Issue](https://github.com/wotttoo/Reproduction-AutoML-based-Autonomous-Intrusion-Detection/issues).

</div>
