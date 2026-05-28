# AutoML-IDS — AutoML-based Autonomous Intrusion Detection System

> Implementation of **"Towards Autonomous Cybersecurity: An Intelligent AutoML Framework for Autonomous Intrusion Detection"**  
> Li Yang · Abdallah Shami — *AutonomousCyber '24, ACM CCS 2024* · [Paper](https://arxiv.org/pdf/2409.03141) · [DOI](https://doi.org/10.1145/3689933.3690833)

<p>
  <img src="Framework.jpg" width="700" alt="AutoML-IDS Framework Diagram"/>
</p>

AutoML-IDS is a **fully autonomous ML pipeline** for network intrusion detection on 5G/6G traffic. It takes a raw CSV and — with zero manual intervention — selects features, balances classes, trains and tunes six classifiers, and produces a stacking ensemble. Every artifact (models, plots, reports) is saved to `output/` with per-dataset prefixes.

---

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [AutoML Pipeline](#automl-pipeline)
- [Key Techniques](#key-techniques)
- [Base Classifiers](#base-classifiers)
- [CLI Reference](#cli-reference)
- [Using `src` as a Library](#using-src-as-a-library)
- [Datasets](#datasets)
- [Output Artifacts](#output-artifacts)
- [Requirements](#requirements)
- [Citation](#citation)

---

## Overview

Traditional ML-based IDSs require substantial expert effort — manual feature engineering, model selection, and hyperparameter tuning. As 5G/6G networks move toward **Zero-Touch Network (ZTN)** management, security systems must match that autonomy. AutoML-IDS automates the full ML analytics pipeline:

```
Raw CSV → Preprocessing → Feature Selection → Data Balancing
       → Model Training → Hyperparameter Optimisation → Ensemble → Evaluation
```

The final model is an **OCSE (Optimised Confidence-based Stacking Ensemble)**: a LightGBM meta-learner trained on both hard predictions and class probabilities from the top-3 base classifiers.

---

## Results

Results below are from the fast mode (`--no-balance --no-tune`) on the bundled dataset samples.

### CICIDS2017 (55 k rows · 7 classes · 38/77 features selected)

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| XGBoost | **99.76%** | **99.76%** | **99.76%** | **99.76%** |
| CatBoost | 99.73% | 99.73% | 99.73% | 99.72% |
| Random Forest | 99.61% | 99.61% | 99.61% | 99.61% |
| Decision Tree | 99.51% | 99.51% | 99.51% | 99.51% |
| Extra Trees | 99.23% | 99.24% | 99.23% | 99.23% |
| LightGBM | 73.55% | 69.36% | 73.55% | 71.07% |
| Traditional Stacking | 99.66% | 99.66% | 99.66% | 99.65% |
| Confidence Stacking | 99.59% | 99.59% | 99.59% | 99.59% |
| Hybrid Stacking (OCSE) | 92.08% | 92.06% | 92.08% | 91.72% |

### 5G-NIDD (48 k rows · 9 classes · 17/48 features selected)

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Extra Trees | **99.92%** | **99.92%** | **99.92%** | **99.92%** |
| LightGBM | **99.92%** | **99.92%** | **99.92%** | **99.92%** |
| Random Forest | 99.91% | 99.91% | 99.91% | 99.91% |
| XGBoost | 99.90% | 99.90% | 99.90% | 99.90% |
| Decision Tree | 99.86% | 99.86% | 99.86% | 99.86% |
| CatBoost | 99.86% | 99.87% | 99.86% | 99.86% |
| Traditional Stacking | 99.88% | 99.88% | 99.88% | 99.88% |
| Confidence Stacking | 98.18% | 98.31% | 98.18% | 98.12% |
| Hybrid Stacking (OCSE) | 98.50% | 98.66% | 98.50% | 98.44% |

> Full pipeline results (with TVAE balancing + BO-TPE tuning) are expected to further improve recall on minority attack classes.

---

## Project Structure

```
AutoML-IDS/
│
├── data/
│   ├── raw/
│   │   ├── CICIDS2017_sample_0.02.csv    # 2% stratified sample (~55 k rows)
│   │   └── 5G-NIDD_0.04.csv             # 4% stratified sample (~48 k rows)
│   └── processed/                        # Feature-selected CSVs (auto-generated)
│
├── notebooks/
│   ├── 01_CICIDS2017_Pipeline.ipynb      # Interactive pipeline — CICIDS2017
│   └── 02_5GNIDD_Pipeline.ipynb          # Interactive pipeline — 5G-NIDD
│
├── src/                                  # Importable Python package
│   ├── __init__.py                       # Re-exports all public classes
│   ├── data_loader.py                    # DataLoader
│   ├── preprocessor.py                   # DataPreprocessor
│   ├── feature_selector.py               # FeatureSelector
│   ├── data_balancer.py                  # DataBalancer (TVAE)
│   ├── model_trainer.py                  # ModelTrainer (6 base models, 3-fold CV)
│   ├── hyperopt_tuner.py                 # HyperparameterTuner (BO-TPE)
│   ├── ensemble.py                       # ModelSelector + EnsembleBuilder (OCSE)
│   └── evaluator.py                      # ModelEvaluator
│
├── output/                               # All pipeline artifacts (git-ignored)
│   ├── models/                           # <dataset>_<model>.pkl  (base + ensemble)
│   ├── plots/                            # <dataset>_<model>_cm.png, feature_importance, comparison
│   └── reports/                          # <dataset>_<model>_report.txt, model_comparison.csv
│
├── docs/
│   ├── project_description.md            # Architecture & technique details
│   └── api_reference.md                  # Full public API reference
│
├── run.py                                # CLI entry point
├── requirements.txt                      # Python dependencies
├── Framework.jpg                         # Pipeline diagram
├── Paper_2409.03141v1.pdf                # Original paper (local copy)
├── .gitignore
└── LICENSE
```

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/wotttoo/AutonomousCyber-AutoML-based-Autonomous-Intrusion-Detection-System.git
cd AutonomousCyber-AutoML-based-Autonomous-Intrusion-Detection-System
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Python 3.9+** required. Tested on Python 3.13.

### 4. Run the pipeline

```bash
# Fast mode — feature selection + training + ensemble (no TVAE, no tuning)
python run.py --dataset cicids2017 --no-balance --no-tune

# Full pipeline — all stages including TVAE balancing and BO-TPE tuning (~1–2 h)
python run.py --dataset cicids2017

# Run on 5G-NIDD
python run.py --dataset 5gnidd --no-balance --no-tune
```

Results are saved to `output/` with `<dataset>_` prefix on all filenames.

### 5. Run interactively (Jupyter)

```bash
jupyter lab
# notebooks/01_CICIDS2017_Pipeline.ipynb
# notebooks/02_5GNIDD_Pipeline.ipynb
```

---

## AutoML Pipeline

```
Raw CSV
  │
  ▼  Step 1 — Automated Pre-processing
     • Label encoding (LabelEncoder)
     • inf / NaN → 0
     • Stratified 80/20 train-test split

  ▼  Step 2 — Automated Feature Selection
     • Train all 6 tree-based classifiers on full feature set
     • Average feature importances from the top-3 models
     • Greedily select the minimal subset reaching 90% cumulative importance

  ▼  Step 3 — Automated Data Balancing  (skippable: --no-balance)
     • Identify minority classes (count < 50% of mean class count)
     • Generate synthetic samples per class with TVAE

  ▼  Step 4 — Automated Model Training
     • Re-train all 6 base classifiers on the balanced, reduced dataset
     • 3-fold cross-validation; rank by mean accuracy
     • Select top-k (default: 3) for the ensemble

  ▼  Step 5 — Hyperparameter Optimisation  (skippable: --no-tune)
     • Bayesian Optimisation with Tree-structured Parzen Estimator (BO-TPE)
     • Per-model search space; 20 function evaluations each
     • Objective: maximise CV / hold-out accuracy

  ▼  Step 6 — Automated Ensemble (OCSE)  (skippable: --no-ensemble)
     • Traditional Stacking  — meta-features: hard class predictions
     • Confidence Stacking   — meta-features: softmax probabilities
     • Hybrid Stacking (OCSE)— meta-features: predictions + probabilities
     • Meta-learner: LightGBM

  ▼  Step 7 — Evaluation & Artifact Export
     • Accuracy, Precision, Recall, F1 (weighted)
     • Per-class classification report (saved to reports/)
     • Confusion matrix heatmap (saved to plots/)
     • Cross-model comparison chart + CSV (saved to plots/ and reports/)
     • All models serialised to output/models/ as .pkl via joblib
```

### Module mapping

| Step | Module | Class |
|------|--------|-------|
| Load | `data_loader.py` | `DataLoader` |
| Preprocess | `preprocessor.py` | `DataPreprocessor` |
| Feature selection | `feature_selector.py` | `FeatureSelector` |
| Data balancing | `data_balancer.py` | `DataBalancer` |
| Training | `model_trainer.py` | `ModelTrainer` |
| Hyperparameter optimisation | `hyperopt_tuner.py` | `HyperparameterTuner` |
| Model ranking | `ensemble.py` | `ModelSelector` |
| Ensemble | `ensemble.py` | `EnsembleBuilder` |
| Evaluation | `evaluator.py` | `ModelEvaluator` |

---

## Key Techniques

### TVAE Data Balancing

A **Tabular Variational Auto-Encoder** (SDV library) learns the joint distribution of all features and synthesises realistic, statistically coherent rows for minority attack classes. Unlike SMOTE, TVAE captures feature correlations and categorical constraints, producing higher-quality synthetic samples that improve recall on rare attack types.

### BO-TPE Hyperparameter Optimisation

[Hyperopt](https://github.com/hyperopt/hyperopt) implements Bayesian Optimisation guided by a **Tree-structured Parzen Estimator**. Compared to grid or random search, BO-TPE allocates more evaluations to promising regions of the search space. Each model family has its own search space; the objective maximises 3-fold CV accuracy (tree models) or hold-out accuracy (gradient boosters).

### OCSE — Optimised Confidence-based Stacking Ensemble

The hybrid stacking variant feeds both hard predictions **and** class probabilities from the top-k base models into the LightGBM meta-learner. This preserves the calibrated confidence signal that pure hard-voting discards, which is especially beneficial for imbalanced multi-class network traffic data.

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

All six are trained, cross-validated, and ranked in every run. The top-3 proceed to the ensemble stage.

---

## CLI Reference

```
python run.py --dataset {cicids2017,5gnidd} [OPTIONS]

Required:
  --dataset {cicids2017,5gnidd}   Dataset to run the pipeline on

Pipeline control:
  --no-balance                    Skip TVAE data balancing
  --no-tune                       Skip BO-TPE hyperparameter optimisation
  --no-ensemble                   Skip ensemble construction

Tuning:
  --top-k   INT   Number of top models used in ensemble (default: 3)
  --max-evals INT  BO-TPE evaluations per model (default: 20)
  --cv      INT   Cross-validation folds (default: 3)

Output:
  --output  PATH  Output directory (default: output)
  -h, --help      Show help and exit
```

**Examples:**

```bash
# Fast dry-run — skip balancing and tuning
python run.py --dataset cicids2017 --no-balance --no-tune

# Full pipeline — all stages
python run.py --dataset 5gnidd

# Custom tuning budget — 50 evals per model, top-5 ensemble
python run.py --dataset cicids2017 --top-k 5 --max-evals 50

# Save to a custom directory
python run.py --dataset cicids2017 --no-balance --no-tune --output results/exp1
```

---

## Using `src` as a Library

All classes are importable from the `src` package:

```python
from src import (
    DataLoader, DataPreprocessor, FeatureSelector,
    DataBalancer, ModelTrainer, HyperparameterTuner,
    ModelSelector, EnsembleBuilder, ModelEvaluator,
)
```

**Minimal example:**

```python
import joblib
from src import DataLoader, DataPreprocessor, FeatureSelector, ModelTrainer, ModelEvaluator

# 1. Load & preprocess
loader = DataLoader("data/raw/CICIDS2017_sample_0.02.csv", label_col="Label")
df = loader.load()

prep = DataPreprocessor(label_col="Label")
df = prep.encode_labels(df)
df = prep.handle_missing(df)

from sklearn.model_selection import train_test_split
import numpy as np
features = [c for c in df.columns if c != "Label"]
X, y = df[features].values, df["Label"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# 2. Feature selection
trainer = ModelTrainer(cv=3)
trainer.train_all(X_train, y_train, X_test)

from src import ModelSelector
selector = ModelSelector(top_k=3)
selector.fit(trainer.cv_scores)

fs = FeatureSelector(importance_threshold=0.9)
fs.fit(selector.top_models, trainer.trained_models, features)
X_train_fs = df[fs.selected_features].values[:len(X_train)]
X_test_fs  = df[fs.selected_features].values[len(X_train):]

# 3. Evaluate & save
evaluator = ModelEvaluator(output_dir="output", prefix="cicids2017")
for name, model in trainer.trained_models.items():
    res = evaluator.evaluate(model, X_test_fs, y_test, name)
    evaluator.plot_confusion_matrix(y_test, res["predictions"], name)

trainer.save_models(output_dir="output", prefix="cicids2017")
```

See `docs/api_reference.md` for full method signatures.

---

## Datasets

| Dataset | Classes | Full size | Sample included | Source |
|---------|---------|-----------|-----------------|--------|
| CICIDS2017 | BENIGN, DoS (×4), PortScan, BruteForce, WebAttack, Bot, Infiltration | ~2.8 M rows | 2% (~55 k rows) | [UNB CIC](https://www.unb.ca/cic/datasets/ids-2017.html) |
| 5G-NIDD | Benign, DDoS (UDP/ICMP/TCP SYN), Reconnaissance, Mirai (×3) | ~1.3 M rows | 4% (~48 k rows) | [IEEE DataPort](https://ieee-dataport.org/documents/5g-nidd-comprehensive-network-intrusion-detection-dataset-generated-over-5g-wireless) |

Stratified samples are committed to `data/raw/` so the pipeline runs out of the box. To use the full datasets, download from the links above and update the `path` entries in `DATASETS` in `run.py`.

---

## Output Artifacts

All files are prefixed with the dataset name to avoid collisions when running multiple datasets.

```
output/
├── models/
│   ├── <dataset>_dt.pkl
│   ├── <dataset>_rf.pkl
│   ├── <dataset>_et.pkl
│   ├── <dataset>_xg.pkl
│   ├── <dataset>_lgbm.pkl
│   ├── <dataset>_cat.pkl
│   ├── <dataset>_ensemble_traditional.pkl
│   ├── <dataset>_ensemble_confidence.pkl
│   └── <dataset>_ensemble_ocse.pkl
│
├── plots/
│   ├── <dataset>_feature_importance.png
│   ├── <dataset>_<model>_cm.png          (one per base classifier)
│   └── <dataset>_model_comparison.png
│
└── reports/
    ├── <dataset>_<model>_report.txt       (one per base classifier)
    └── <dataset>_model_comparison.csv
```

Models are serialised with `joblib` and can be loaded for inference:

```python
import joblib
model = joblib.load("output/models/cicids2017_xg.pkl")
predictions = model.predict(X_test)
```

---

## Requirements

| Package | Minimum version | Role |
|---------|----------------|------|
| Python | 3.9 | Runtime |
| numpy | 2.4 | Numerical arrays |
| pandas | 2.3 | DataFrame operations |
| scipy | 1.17 | Statistical utilities |
| scikit-learn | 1.8 | DT, RF, ET, metrics, preprocessing |
| joblib | 1.5 | Model serialisation (.pkl) |
| xgboost | 3.2 | XGBoost classifier |
| lightgbm | 4.6 | LightGBM classifier + meta-learner |
| catboost | 1.2 | CatBoost classifier |
| hyperopt | 0.2.7 | BO-TPE hyperparameter search |
| sdv | 1.36 | TVAE synthetic data generation |
| matplotlib | 3.10 | Plotting |
| seaborn | 0.13 | Heatmap visualisations |
| jupyterlab | 4.5 | Interactive notebooks |

```bash
pip install -r requirements.txt
```

---

## Citation

If you use this code or build on this work, please cite the original paper:

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

**Original authors:** Li Yang ([liyanghart@gmail.com](mailto:liyanghart@gmail.com)) · Abdallah Shami — ANTS Lab (Ontario Tech) / OC2 Lab (Western University)  
**Reproduction:** For issues with this implementation, open a GitHub issue on this repository.
