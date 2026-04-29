# Project Description

## Overview

AutoML-IDS is an **Automated Machine Learning (AutoML) framework for autonomous network intrusion detection**, designed for next-generation (5G/6G) networks.  The system automates every stage of the ML data analytics pipeline, from raw traffic data to a final optimised ensemble classifier — requiring zero manual feature engineering or model tuning.

This implementation accompanies the paper:

> L. Yang and A. Shami, "Towards Autonomous Cybersecurity: An Intelligent AutoML Framework for Autonomous Intrusion Detection," *AutonomousCyber '24, ACM CCS 2024*, doi: 10.1145/3689933.3690833.

---

## Motivation

Traditional ML-based IDSs demand significant expert effort: manual feature selection, model choice, and hyperparameter tuning.  As 5G and 6G networks move towards **Zero-Touch Network (ZTN)** management, security must be equally autonomous.  AutoML-IDS addresses this gap by fully automating the ML pipeline.

---

## AutoML Pipeline

```
Raw CSV
  │
  ▼
1. Automated Data Pre-Processing
   └─ Label encoding, inf/NaN handling, stratified train/test split

  ▼
2. Automated Feature Engineering
   └─ Train all six tree-based base learners
   └─ Average feature importances of top-3 models
   └─ Select minimal feature set reaching 90 % cumulative importance

  ▼
3. Automated Data Balancing
   └─ Identify minority classes (< 50 % of mean class count)
   └─ Generate synthetic samples per class with TVAE

  ▼
4. Automated Model Selection
   └─ Re-train all base learners on balanced, feature-selected data
   └─ Cross-validate and rank by mean accuracy
   └─ Select top-3 models for ensemble

  ▼
5. Hyperparameter Optimisation (BO-TPE)
   └─ Bayesian Optimisation with Tree-structured Parzen Estimator
   └─ Independent search space and objective per model family

  ▼
6. Automated Model Ensemble (OCSE)
   └─ Traditional Stacking   — stacks hard predictions
   └─ Confidence-based Stacking — stacks softmax probabilities
   └─ Hybrid Stacking (OCSE) — stacks predictions + probabilities
   └─ Meta-learner: LightGBM
```

---

## Base Classifiers

| Key    | Algorithm          | Library       |
|--------|--------------------|---------------|
| `dt`   | Decision Tree      | scikit-learn  |
| `rf`   | Random Forest      | scikit-learn  |
| `et`   | Extra Trees        | scikit-learn  |
| `xg`   | XGBoost            | xgboost       |
| `lgbm` | LightGBM           | lightgbm      |
| `cat`  | CatBoost           | catboost      |

---

## Datasets

| Dataset | Description | Source |
|---------|-------------|--------|
| CICIDS2017 | Network traffic with common attacks (DoS, PortScan, Infiltration, …) | [UNB CIC](https://www.unb.ca/cic/datasets/ids-2017.html) |
| 5G-NIDD   | 5G network intrusion dataset (DDoS, Reconnaissance, …)              | [IEEE DataPort](https://ieee-dataport.org/documents/5g-nidd-comprehensive-network-intrusion-detection-dataset-generated-over-5g-wireless) |

Sampled subsets (2 % CICIDS2017, 4 % 5G-NIDD) are included in `data/raw/`.

---

## Key Techniques

### TVAE Data Balancing
A **Tabular Variational Auto-Encoder** (from the [SDV](https://docs.sdv.dev/sdv) library) generates realistic synthetic samples for minority attack classes.  This avoids the class imbalance bias that degrades recall on rare attack types.

### BO-TPE Hyperparameter Optimisation
[Hyperopt](https://github.com/hyperopt/hyperopt) implements Bayesian Optimisation guided by a **Tree-structured Parzen Estimator**.  Each model family has its own search space; the tuner maximises cross-validated accuracy (or hold-out accuracy for boosters).

### OCSE — Optimised Confidence-based Stacking Ensemble
The hybrid stacking variant concatenates both hard predictions and class probabilities from the top-k models as features for the meta-learner.  This preserves the confidence signal that pure hard-voting loses.

---

## Evaluation Metrics

- Accuracy
- Weighted Precision, Recall, F1-score
- Per-class classification report
- Confusion matrix
- Training time (seconds)
- Prediction latency (ms / sample)
