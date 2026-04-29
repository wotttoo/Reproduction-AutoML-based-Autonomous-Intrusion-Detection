# AutoML-IDS — AutoML-based Autonomous Intrusion Detection System

Implementation of the paper ["Towards Autonomous Cybersecurity: An Intelligent AutoML Framework for Autonomous Intrusion Detection"](https://arxiv.org/pdf/2409.03141) (AutonomousCyber '24, ACM CCS 2024).

**Authors:** Li Yang · Abdallah Shami  
**Organizations:** ANTS Lab (Ontario Tech) · OC2 Lab (Western University)

<p>
  <img src="Framework.jpg" width="540" alt="AutoML-IDS Framework"/>
</p>

---

## Project Structure

```
AutoML-IDS/
├── data/
│   ├── raw/                        # Original dataset CSVs
│   │   ├── CICIDS2017_sample_0.02.csv
│   │   └── 5G-NIDD_0.04.csv
│   └── processed/                  # Feature-selected CSVs (generated at runtime)
│
├── notebooks/
│   ├── 01_CICIDS2017_Pipeline.ipynb   # Full pipeline on CICIDS2017
│   └── 02_5GNIDD_Pipeline.ipynb       # Full pipeline on 5G-NIDD
│
├── src/                            # Reusable Python package
│   ├── __init__.py
│   ├── data_loader.py              # DataLoader
│   ├── preprocessor.py             # DataPreprocessor
│   ├── feature_selector.py         # FeatureSelector
│   ├── data_balancer.py            # DataBalancer (TVAE)
│   ├── model_trainer.py            # ModelTrainer (6 base models)
│   ├── hyperopt_tuner.py           # HyperparameterTuner (BO-TPE)
│   ├── ensemble.py                 # ModelSelector + EnsembleBuilder (OCSE)
│   └── evaluator.py                # ModelEvaluator
│
├── output/
│   ├── models/                     # Saved model artefacts
│   ├── plots/                      # Confusion matrices, feature importance charts
│   └── reports/                    # Text classification reports, CSV comparison
│
├── docs/
│   ├── project_description.md      # Architecture and technique details
│   └── api_reference.md            # Public API for all src classes
│
├── run.py                          # CLI entry point for the full pipeline
├── requirements.txt
├── .gitignore
├── Framework.jpg
├── Paper_2409.03141v1.pdf
└── LICENSE
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline (CLI)

```bash
# CICIDS2017 — all steps
python run.py --dataset cicids2017

# 5G-NIDD — skip balancing and tuning for a fast dry-run
python run.py --dataset 5gnidd --no-balance --no-tune

# Full options
python run.py --help
```

### 3. Run interactively (Jupyter)

```bash
jupyter lab
# Open notebooks/01_CICIDS2017_Pipeline.ipynb
# Open notebooks/02_5GNIDD_Pipeline.ipynb
```

---

## AutoML Pipeline

| Step | Module | Description |
|------|--------|-------------|
| 1. Load | `DataLoader` | Read CSV, inspect class distribution |
| 2. Preprocess | `DataPreprocessor` | Encode labels, fill inf/NaN, stratified split |
| 3. Feature selection | `FeatureSelector` | Train all 6 models, average top-3 importances, 90 % threshold |
| 4. Data balancing | `DataBalancer` | TVAE synthetic oversampling for minority classes |
| 5. Model training | `ModelTrainer` | 3-fold CV on balanced feature-selected data |
| 6. HPO | `HyperparameterTuner` | BO-TPE (hyperopt) per model |
| 7. Ensemble | `EnsembleBuilder` | Traditional / Confidence / Hybrid (OCSE) stacking |
| 8. Evaluate | `ModelEvaluator` | Metrics, confusion matrices, comparison chart |

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

---

## Using `src` as a library

```python
from src import (
    DataLoader, DataPreprocessor, FeatureSelector,
    DataBalancer, ModelTrainer, HyperparameterTuner,
    ModelSelector, EnsembleBuilder, ModelEvaluator,
)

# Load & preprocess
loader = DataLoader("data/raw/CICIDS2017_sample_0.02.csv")
df     = loader.load()
prep   = DataPreprocessor()
df     = prep.encode_labels(df)
df     = prep.handle_missing(df)
X_train, X_test, y_train, y_test = prep.split(df)

# Train, select features, balance, tune, ensemble …
# See notebooks/ or docs/api_reference.md for the complete workflow.
```

---

## Datasets

| Dataset | Attacks | Samples (full) | Subset used |
|---------|---------|---------------|-------------|
| [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) | DoS, PortScan, Infiltration, BruteForce, … | ~2.8 M | 2 % |
| [5G-NIDD](https://ieee-dataport.org/documents/5g-nidd-comprehensive-network-intrusion-detection-dataset-generated-over-5g-wireless) | DDoS, Recon, Mirai, … | ~1.3 M | 4 % |

---

## Requirements

```
Python 3.9+
scikit-learn, xgboost, lightgbm, catboost
hyperopt
sdv (≥ 1.9)
pandas, numpy, matplotlib, seaborn
```

Install with: `pip install -r requirements.txt`

---

## Citation

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

Li Yang · [liyanghart@gmail.com](mailto:liyanghart@gmail.com) · [GitHub](https://github.com/LiYangHart) · [Google Scholar](https://scholar.google.com.eg/citations?user=XEfM7bIAAAAJ&hl=en)
