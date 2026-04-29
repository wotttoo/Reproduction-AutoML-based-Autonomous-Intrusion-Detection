#!/usr/bin/env python3
"""
AutoML-IDS: Command-line entry point for the full AutoML pipeline.

Usage examples
--------------
# Run on CICIDS2017 with all steps enabled:
    python run.py --dataset cicids2017

# Skip TVAE balancing and hyperparameter tuning (much faster):
    python run.py --dataset 5gnidd --no-balance --no-tune

# Only run feature selection + base training (no ensemble):
    python run.py --dataset cicids2017 --no-balance --no-tune --no-ensemble
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Anchor all relative paths to the directory that contains this file,
# so the script works regardless of the current working directory.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.feature_selector import FeatureSelector
from src.data_balancer import DataBalancer
from src.model_trainer import ModelTrainer
from src.hyperopt_tuner import HyperparameterTuner
from src.ensemble import ModelSelector, EnsembleBuilder
from src.evaluator import ModelEvaluator

# ------------------------------------------------------------------
# Dataset registry
# ------------------------------------------------------------------
DATASETS = {
    "cicids2017": {
        "path":      "data/raw/CICIDS2017_sample_0.02.csv",
        "label":     "Label",
        "fs_csv":    "data/processed/cicids2017_fs.csv",
        "plot_title": "Feature Importance — CICIDS2017",
    },
    "5gnidd": {
        "path":      "data/raw/5G-NIDD_0.04.csv",
        "label":     "Label",
        "fs_csv":    "data/processed/5gnidd_fs.csv",
        "plot_title": "Feature Importance — 5G-NIDD",
    },
}


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

def run_pipeline(
    dataset: str,
    balance: bool = True,
    tune: bool = True,
    ensemble: bool = True,
    top_k: int = 3,
    max_evals: int = 20,
    cv: int = 3,
    output_dir: str = "output",
):
    cfg = DATASETS[dataset]

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)

    sep = "=" * 65
    print(f"\n{sep}\n  AutoML-IDS Pipeline: {dataset}\n{sep}")

    # ── Step 1: Load & inspect ──────────────────────────────────────
    print("\n[Step 1/7] Loading data...")
    loader = DataLoader(cfg["path"], label_col=cfg["label"])
    df = loader.load()
    loader.summary()

    # ── Step 2: Preprocess ──────────────────────────────────────────
    print("\n[Step 2/7] Preprocessing...")
    prep = DataPreprocessor(label_col=cfg["label"])
    df = prep.encode_labels(df)
    df = prep.handle_missing(df)
    feature_names = df.drop([cfg["label"]], axis=1).columns.tolist()

    X = df[feature_names].values
    y = np.ravel(df[cfg["label"]].values)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

    # ── Step 3: Initial training (for feature importance) ───────────
    print("\n[Step 3/7] Training base models for feature selection...")
    trainer_fs = ModelTrainer(cv=cv)
    trainer_fs.train_all(X_train, y_train, X_test)

    # ── Step 4: Automated feature selection ─────────────────────────
    print("\n[Step 4/7] Automated feature selection...")
    selector = ModelSelector(top_k=top_k)
    selector.fit(trainer_fs.cv_scores)
    top_models_fs = selector.top_models

    fs = FeatureSelector(importance_threshold=0.9)
    fs.fit(top_models_fs, trainer_fs.trained_models, feature_names)
    print(f"  Selected {len(fs.selected_features)} / {len(feature_names)} features")

    X_fs = df[fs.selected_features].values
    X_train_fs, X_test_fs, y_train_fs, y_test_fs = train_test_split(
        X_fs, y, test_size=0.2, random_state=0, stratify=y
    )

    df_fs = pd.DataFrame(X_fs, columns=fs.selected_features)
    df_fs[cfg["label"]] = y
    df_fs.to_csv(cfg["fs_csv"], index=False)

    fs.plot(
        title=cfg["plot_title"],
        save_path=os.path.join(output_dir, "plots", f"{dataset}_feature_importance.png"),
    )

    # ── Step 5: Data balancing ──────────────────────────────────────
    if balance:
        print("\n[Step 5/7] Balancing training data with TVAE...")
        X_train_df = pd.DataFrame(X_train_fs, columns=fs.selected_features)
        y_train_s  = pd.Series(y_train_fs)
        balancer   = DataBalancer(label_col=cfg["label"])
        X_train_bal, y_train_bal = balancer.fit_resample(X_train_df, y_train_s, cfg["fs_csv"])
        print(f"  Balanced: {X_train_bal.shape}  "
              f"class dist: {dict(pd.Series(y_train_bal).value_counts())}")
        X_tr = X_train_bal.values
        y_tr = y_train_bal.values
    else:
        print("\n[Step 5/7] Skipping data balancing.")
        X_tr, y_tr = X_train_fs, y_train_fs

    # ── Step 6: Train models on feature-selected (balanced) data ────
    print("\n[Step 6/7] Training models on processed dataset...")
    trainer = ModelTrainer(cv=cv)
    trainer.train_all(X_tr, y_tr, X_test_fs)

    # ── Step 6b: Hyperparameter tuning ──────────────────────────────
    if tune:
        print("\n  ► Hyperparameter optimisation (BO-TPE)...")
        tuner = HyperparameterTuner(max_evals=max_evals)
        for name in ModelTrainer.MODEL_KEYS:
            tuned_model, _ = tuner.tune(name, X_tr, y_tr, X_test_fs, y_test_fs)
            trainer.replace_model(name, tuned_model, X_tr, X_test_fs)
    else:
        print("\n  ► Skipping hyperparameter tuning.")

    # ── Step 7: Evaluate base models & ensemble ─────────────────────
    print(f"\n[Step 7/7] Evaluation & ensemble learning...")
    evaluator = ModelEvaluator(output_dir=output_dir)
    results: dict = {}

    for name, model in trainer.trained_models.items():
        res = evaluator.evaluate(model, X_test_fs, y_test_fs, name)
        evaluator.plot_confusion_matrix(y_test_fs, res["predictions"], name)
        results[name] = res

    if ensemble:
        selector2 = ModelSelector(top_k=top_k)
        selector2.fit(trainer.cv_scores)
        top_models_ens = selector2.top_models

        builder = EnsembleBuilder()
        for method, fn in [
            ("Traditional Stacking",   builder.traditional_stacking),
            ("Confidence Stacking",    builder.confidence_stacking),
            ("Hybrid Stacking (OCSE)", builder.hybrid_stacking),
        ]:
            _, pred = fn(trainer.predictions, top_models_ens, y_tr, y_test_fs)
            acc = accuracy_score(y_test_fs, pred)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_test_fs, pred, average="weighted"
            )
            results[method] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    evaluator.compare_models(results)
    print(f"\n{sep}\n  Done.  Results saved to '{output_dir}/'\n{sep}\n")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="AutoML-IDS: Autonomous Intrusion Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset", choices=list(DATASETS), default="cicids2017",
        help="Dataset to run the pipeline on (default: cicids2017)",
    )
    p.add_argument("--no-balance",  action="store_true", help="Skip TVAE data balancing")
    p.add_argument("--no-tune",     action="store_true", help="Skip BO-TPE hyperparameter tuning")
    p.add_argument("--no-ensemble", action="store_true", help="Skip ensemble learning")
    p.add_argument("--top-k",  type=int, default=3,  help="Number of top models for ensemble (default: 3)")
    p.add_argument("--max-evals", type=int, default=20, help="BO-TPE iterations per model (default: 20)")
    p.add_argument("--cv",     type=int, default=3,  help="Cross-validation folds (default: 3)")
    p.add_argument("--output", default="output",     help="Output directory (default: output)")
    return p.parse_args()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    args = parse_args()
    run_pipeline(
        dataset=args.dataset,
        balance=not args.no_balance,
        tune=not args.no_tune,
        ensemble=not args.no_ensemble,
        top_k=args.top_k,
        max_evals=args.max_evals,
        cv=args.cv,
        output_dir=args.output,
    )
