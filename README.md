# AutoML-IDS — AutoML-based Autonomous Intrusion Detection System

Implementation of the paper ["Towards Autonomous Cybersecurity: An Intelligent AutoML Framework for Autonomous Intrusion Detection"](https://arxiv.org/pdf/2409.03141) (AutonomousCyber '24, ACM CCS 2024).

**Authors:** Li Yang · Abdallah Shami  
**Organizations:** ANTS Lab (Ontario Tech) · OC2 Lab (Western University)

<p>
  <img src="Framework.jpg" width="600" alt="AutoML-IDS Framework"/>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [AutoML Pipeline](#automl-pipeline)
- [Key Techniques](#key-techniques)
- [Base Classifiers](#base-classifiers)
- [Using `src` as a Library](#using-src-as-a-library)
- [CLI Reference](#cli-reference)
- [Datasets](#datasets)
- [Output Artifacts](#output-artifacts)
- [Requirements](#requirements)
- [Citation](#citation)
- [Contact](#contact)

---

## Overview

AutoML-IDS is a fully autonomous machine learning framework for network intrusion detection, targeting next-generation (5G/6G) networks. It automates every stage of the ML analytics pipeline — from raw traffic CSV to a final optimised ensemble classifier — with **zero manual feature engineering or model tuning**.

The system chains eight automated stages: preprocessing → feature selection → data balancing → multi-model training → Bayesian hyperparameter optimisation → ensemble construction → evaluation. The final OCSE (Optimised Confidence-based Stacking Ensemble) combines hard predictions and class probabilities from the top-3 classifiers using a LightGBM meta-learner.

---

## Motivation

Traditional ML-based intrusion detection systems demand significant expert effort: manual feature selection, model choice, and hyperparameter tuning. As 5G and 6G networks move towards **Zero-Touch Network (ZTN)** management, security must be equally autonomous. AutoML-IDS addresses this gap by fully automating the ML pipeline, enabling deployment in environments where continuous expert oversight is impractical.

---

## Project Structure

```
AutoML-IDS/
├── data/
│   ├── raw/                              # Original dataset CSVs (included)
│   │   ├── CICIDS2017_sample_0.02.csv    # 2 % stratified sample (~55k rows)
│   │   └── 5G-NIDD_0.04.csv             # 4 % stratified sample (~52k rows)
│   └── processed/                        # Feature-selected CSVs (generated at runtime)
│
├── notebooks/
│   ├── 01_CICIDS2017_Pipeline.ipynb      # Full interactive pipeline on CICIDS2017
│   └── 02_5GNIDD_Pipeline.ipynb          # Full interactive pipeline on 5G-NIDD
│
├── src/                                  # Reusable Python package
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
├── output/
│   ├── models/                           # Saved model artefacts (.pkl / .json)
│   ├── plots/                            # Confusion matrices, feature importance charts
│   └── reports/                          # Text classification reports, CSV comparison
│
├── docs/
│   ├── project_description.md            # Architecture and technique details
│   └── api_reference.md                  # Public API for all src classes
│
├── run.py                                # CLI entry point for the full pipeline
├── requirements.txt
├── Framework.jpg                         # Pipeline diagram
├── Paper_2409.03141v1.pdf                # Original paper (local copy)
├── .gitignore
└── LICENSE
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/wotttoo/Reproduction-AutoML-based-Autonomous-Intrusion-Detection.git
cd Reproduction-AutoML-based-Autonomous-Intrusion-Detection
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> Python 3.9 or higher is required.

### 4. Run the full pipeline (CLI)

```bash
# CICIDS2017 — all eight stages
python run.py --dataset cicids2017

# 5G-NIDD — skip balancing and tuning for a fast dry-run
python run.py --dataset 5gnidd --no-balance --no-tune

# Only feature selection + base training (no ensemble)
python run.py --dataset cicids2017 --no-balance --no-tune --no-ensemble

# Full options
python run.py --help
```

### 5. Run interactively (Jupyter)

```bash
jupyter lab
# Open notebooks/01_CICIDS2017_Pipeline.ipynb
# Open notebooks/02_5GNIDD_Pipeline.ipynb
```

Each notebook walks through all pipeline stages with inline visualisations and explanatory markdown cells.

---

## AutoML Pipeline

```
Raw CSV
  │
  ▼
1. Automated Data Pre-Processing
   └─ Label encoding (LabelEncoder)
   └─ inf / NaN → 0 cleaning
   └─ Stratified 80/20 train/test split

  ▼
2. Automated Feature Engineering
   └─ Train all six tree-based base learners on full feature set
   └─ Average feature importances from the top-3 models
   └─ Select minimal feature subset reaching 90 % cumulative importance

  ▼
3. Automated Data Balancing (TVAE)
   └─ Identify minority classes (count < 50 % of mean class count)
   └─ Generate realistic synthetic samples per minority class with TVAE

  ▼
4. Automated Model Selection
   └─ Re-train all six base learners on balanced, feature-selected data
   └─ 3-fold cross-validation; rank by mean accuracy
   └─ Select top-3 models for the ensemble

  ▼
5. Hyperparameter Optimisation (BO-TPE)
   └─ Bayesian Optimisation with Tree-structured Parzen Estimator (hyperopt)
   └─ Per-model search space; maximise cross-validated / hold-out accuracy
   └─ Default: 20 function evaluations per model

  ▼
6. Automated Model Ensemble (OCSE)
   └─ Traditional Stacking    — meta-features: hard class predictions
   └─ Confidence Stacking     — meta-features: softmax probabilities
   └─ Hybrid Stacking (OCSE)  — meta-features: predictions + probabilities
   └─ Meta-learner: LightGBM

  ▼
7. Evaluation
   └─ Accuracy, Precision, Recall, F1 (weighted)
   └─ Per-class classification report
   └─ Confusion matrix heatmap
   └─ Cross-model comparison chart + CSV
   └─ Training time (s) and prediction latency (ms/sample)
```

### Stage-by-stage module mapping

| Step | Module | Key class | Description |
|------|--------|-----------|-------------|
| 1. Load | `data_loader.py` | `DataLoader` | Read CSV, inspect class distribution |
| 2. Preprocess | `preprocessor.py` | `DataPreprocessor` | Encode labels, fill inf/NaN, stratified split |
| 3. Feature selection | `feature_selector.py` | `FeatureSelector` | Average top-3 importances, 90 % threshold |
| 4. Data balancing | `data_balancer.py` | `DataBalancer` | TVAE synthetic oversampling for minority classes |
| 5. Model training | `model_trainer.py` | `ModelTrainer` | 3-fold CV on balanced feature-selected data |
| 6. HPO | `hyperopt_tuner.py` | `HyperparameterTuner` | BO-TPE, 20 evals per model |
| 7. Model selection | `ensemble.py` | `ModelSelector` | Rank by mean CV accuracy, keep top-k |
| 8. Ensemble | `ensemble.py` | `EnsembleBuilder` | Traditional / Confidence / Hybrid (OCSE) stacking |
| 9. Evaluate | `evaluator.py` | `ModelEvaluator` | Metrics, confusion matrices, comparison chart |

---

## Key Techniques

### TVAE Data Balancing

A **Tabular Variational Auto-Encoder** (from the [SDV](https://docs.sdv.dev/sdv) library) generates realistic synthetic samples for minority attack classes. Unlike simple oversampling (SMOTE), TVAE learns the joint distribution of all features and produces coherent, statistically plausible rows. This avoids the class-imbalance bias that degrades recall on rare attack types.

### BO-TPE Hyperparameter Optimisation

[Hyperopt](https://github.com/hyperopt/hyperopt) implements Bayesian Optimisation guided by a **Tree-structured Parzen Estimator**. Each model family has its own search space; the tuner maximises cross-validated accuracy (k-fold CV for tree models, hold-out accuracy for gradient boosters). With 20 evaluations per model it typically outperforms random or grid search at a fraction of the compute budget.

### OCSE — Optimised Confidence-based Stacking Ensemble

The hybrid stacking variant concatenates both hard predictions **and** class probabilities from the top-k models as features for the LightGBM meta-learner. This preserves the calibrated confidence signal that pure hard-voting loses, leading to improved performance on imbalanced multi-class problems typical of network intrusion data.

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

All six are trained in every pipeline run, cross-validated, and ranked. The top-3 proceed to the ensemble stage.

---

## Using `src` as a Library

All classes are importable from the `src` package directly:

```python
from src import (
    DataLoader, DataPreprocessor, FeatureSelector,
    DataBalancer, ModelTrainer, HyperparameterTuner,
    ModelSelector, EnsembleBuilder, ModelEvaluator,
)
```

### Minimal end-to-end example

```python
# 1. Load & preprocess
loader = DataLoader("data/raw/CICIDS2017_sample_0.02.csv")
df     = loader.load()
loader.summary()                          # prints shape, class counts, missing values

prep   = DataPreprocessor(test_size=0.2, random_state=0)
df     = prep.encode_labels(df)
df     = prep.handle_missing(df)
X_train, X_test, y_train, y_test = prep.split(df)

# 2. Feature selection
trainer    = ModelTrainer(random_state=0, cv=3)
trainer.train_all(X_train, y_train, X_test)

selector   = ModelSelector(top_k=3)
selector.fit(trainer.cv_scores)           # ranks models by mean CV accuracy

fs         = FeatureSelector(importance_threshold=0.9)
X_train_fs = fs.fit_transform(
    selector.top_models, trainer.trained_models,
    list(X_train.columns), X_train
)
X_test_fs  = fs.transform(X_test)
print(f"Selected {len(fs.selected_features)} features")

# 3. Balance training data
balancer = DataBalancer()
X_bal, y_bal = balancer.fit_resample(
    X_train_fs, y_train,
    metadata_csv_path="data/processed/cicids2017_fs.csv"
)

# 4. Re-train on balanced data
trainer2 = ModelTrainer(random_state=0, cv=3)
trainer2.train_all(X_bal, y_bal, X_test_fs)

# 5. Hyperparameter optimisation
tuner = HyperparameterTuner(max_evals=20)
tuned = tuner.tune_all(selector.top_models, X_bal, y_bal, X_test_fs, y_test)
for name, model in tuned.items():
    trainer2.replace_model(name, model, X_bal, X_test_fs, cv_score=None)

# 6. Ensemble
builder   = EnsembleBuilder(random_state=0)
_, y_pred = builder.hybrid_stacking(
    trainer2.predictions, selector.top_models, y_bal, y_test
)

# 7. Evaluate
evaluator = ModelEvaluator(output_dir="output")
results   = evaluator.evaluate(None, X_test_fs, y_test, "OCSE_Hybrid", save=True)
evaluator.plot_confusion_matrix(y_test, y_pred, "OCSE_Hybrid", save=True)
```

See `docs/api_reference.md` for the complete method signatures of every class.

---

## CLI Reference

```
python run.py [OPTIONS]

Options:
  --dataset {cicids2017,5gnidd}   Dataset to run (required)
  --no-balance                    Skip TVAE data balancing step
  --no-tune                       Skip BO-TPE hyperparameter optimisation
  --no-ensemble                   Skip ensemble construction
  -h, --help                      Show this message and exit
```

**Example invocations:**

```bash
# Full pipeline — all steps
python run.py --dataset cicids2017

# Fast run — feature selection + base training only
python run.py --dataset 5gnidd --no-balance --no-tune --no-ensemble

# Tune but skip balancing
python run.py --dataset cicids2017 --no-balance
```

All outputs (models, plots, reports) are written to `output/`.

---

## Datasets

| Dataset | Attack types | Full size | Subset included |
|---------|-------------|-----------|-----------------|
| [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) | DoS (Hulk, Slowloris, SlowHTTPTest, GoldenEye), PortScan, Infiltration, BruteForce, Web attacks, Botnet | ~2.8 M rows | 2 % (~55k rows) |
| [5G-NIDD](https://ieee-dataport.org/documents/5g-nidd-comprehensive-network-intrusion-detection-dataset-generated-over-5g-wireless) | DDoS (UDP, ICMP, TCP SYN), Reconnaissance, Mirai (Ackflooding, HTTP Flooding, UDP Flooding) | ~1.3 M rows | 4 % (~52k rows) |

Sampled subsets are committed to `data/raw/` so the pipeline runs out of the box without downloading the full datasets. To use the full datasets, download them from the links above and update the paths in `run.py` or the notebooks.

---

## Output Artifacts

After a complete pipeline run, the following files are written to `output/`:

| Path | Description |
|------|-------------|
| `output/plots/<model>_confusion_matrix.png` | Per-model confusion matrix heatmap |
| `output/plots/feature_importance.png` | Scatter plot of selected feature importances |
| `output/plots/model_comparison.png` | Bar chart comparing all models |
| `output/reports/<model>_classification_report.txt` | Per-class precision / recall / F1 |
| `output/reports/model_comparison.csv` | Accuracy, F1, training time, latency for all models |
| `data/processed/<dataset>_fs.csv` | Feature-selected training data (used by TVAE) |

---

## Requirements

| Package | Minimum version | Purpose |
|---------|----------------|---------|
| Python | 3.9 | Language runtime |
| numpy | 1.24 | Numerical arrays |
| pandas | 2.0 | DataFrame operations |
| scipy | 1.10 | Statistical utilities |
| scikit-learn | 1.3 | DT, RF, ET, preprocessing, metrics |
| xgboost | 2.0 | XGBoost classifier |
| lightgbm | 4.0 | LightGBM classifier + meta-learner |
| catboost | 1.2 | CatBoost classifier |
| hyperopt | 0.2.7 | BO-TPE hyperparameter search |
| sdv | 1.9 | TVAE synthetic data generation |
| matplotlib | 3.7 | Plotting |
| seaborn | 0.12 | Heatmaps |
| jupyterlab | 4.0 | Interactive notebooks |

Install everything with:

```bash
pip install -r requirements.txt
```

---

## Citation

If you use this code or build on this work, please cite the original paper:

```bibtex
@INPROCEEDINGS{3690833,
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

## Contact

**Li Yang** (original author)  
[liyanghart@gmail.com](mailto:liyanghart@gmail.com) · [GitHub](https://github.com/LiYangHart) · [Google Scholar](https://scholar.google.com.eg/citations?user=XEfM7bIAAAAJ&hl=en)

For issues with this reproduction, open a GitHub issue on this repository.
