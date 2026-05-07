#!/usr/bin/env python3
"""
Train XGBoost Genetic Branch
==============================
Trains an XGBoost classifier on the 12-dimensional genetic feature vector
to predict patient-level seizure risk.

Input:  data/processed/genetic_vectors/genetic_training_cohort.csv
        Columns: 9 mutation flags + 2 pLI scores + 1 PRS
Output: models/xgboost_genetic/model.pkl, metrics.json, plots/

Usage:
    python cloud_training/03_train_xgboost_genetic.py
    python cloud_training/03_train_xgboost_genetic.py --data data/processed/genetic_vectors/genetic_training_cohort.csv
    python cloud_training/03_train_xgboost_genetic.py --target preictal_ratio   # regression mode
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # headless for cloud
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
)
from sklearn.model_selection import train_test_split

# ── Project root ──
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# Config
# ============================================================
GENETIC_FEATURES = [
    "SCN1A_mutation", "SCN8A_mutation", "KCNQ2_mutation", "SCN2A_mutation",
    "KCNT1_mutation", "DEPDC5_mutation", "PCDH19_mutation", "GRIN2A_mutation",
    "GABRA1_mutation", "SCN1A_pLI", "SCN8A_pLI", "polygenic_risk_score",
]


def detect_device():
    """Use CUDA for XGBoost if available, else CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def load_data(csv_path):
    """Load genetic training cohort."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} patients from {csv_path}")
    print(f"  Seizure rate: {df['has_seizure'].mean()*100:.1f}%")
    print(f"  Genetic features: {len(GENETIC_FEATURES)}")
    return df


def prepare_data(df, target_col="has_seizure", test_size=0.2, val_size=0.15, seed=42):
    """Split into train/val/test."""
    X = df[GENETIC_FEATURES].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    # Stratified split for classification
    if target_col == "has_seizure":
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_size/(1-test_size),
            random_state=seed, stratify=y_trainval
        )
    else:
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_size/(1-test_size),
            random_state=seed
        )

    print(f"\nData splits:")
    print(f"  Train: {len(y_train)}  (seizure={y_train.sum():.0f})")
    print(f"  Val:   {len(y_val)}    (seizure={y_val.sum():.0f})")
    print(f"  Test:  {len(y_test)}   (seizure={y_test.sum():.0f})")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def build_model(classification=True):
    """Build XGBoost model."""
    device = detect_device()
    print(f"\nXGBoost device: {device}")

    if classification:
        return xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=3,  # lighter than EEG because data is more balanced
            eval_metric="auc",
            tree_method="hist",
            device=device,
            random_state=42,
            n_jobs=-1,
        )
    else:
        return xgb.XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="rmse",
            tree_method="hist",
            device=device,
            random_state=42,
            n_jobs=-1,
        )


def train_model(model, X_train, y_train, X_val, y_val, output_dir):
    """Train with early stopping."""
    fit_kwargs = {
        "X": X_train,
        "y": y_train,
        "eval_set": [(X_val, y_val)],
        "verbose": 50,
    }

    # Try modern API first
    try:
        callbacks = [xgb.callback.EarlyStopping(rounds=20, save_best=True)]
        fit_kwargs["callbacks"] = callbacks
        model.fit(**fit_kwargs)
        print(f"  Early stopping via callbacks (rounds=20)")
    except TypeError:
        fit_kwargs.pop("callbacks", None)
        try:
            fit_kwargs["early_stopping_rounds"] = 20
            model.fit(**fit_kwargs)
            print(f"  Early stopping via early_stopping_rounds=20")
        except TypeError:
            fit_kwargs.pop("early_stopping_rounds", None)
            print("  [WARNING] Training without early stopping")
            model.fit(**fit_kwargs)

    # Save model
    model_path = output_dir / "xgboost_genetic_model.pkl"
    joblib.dump(model, model_path)
    print(f"  Model saved: {model_path}")
    return model


def evaluate_classification(model, X_test, y_test, output_dir, threshold=0.5):
    """Evaluate classifier and generate plots."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "auc": float(auc),
        "average_precision": float(ap),
        "accuracy": float(acc),
        "precision": float(report["1"]["precision"]),
        "recall": float(report["1"]["recall"]),
        "f1": float(report["1"]["f1-score"]),
        "threshold": threshold,
        "n_test": len(y_test),
        "n_test_seizure": int(y_test.sum()),
    }

    print(f"\nTest Results:")
    print(f"  AUC:  {auc:.4f}")
    print(f"  AP:   {ap:.4f}")
    print(f"  Acc:  {acc:.4f}")
    print(f"  Prec: {metrics['precision']:.4f}")
    print(f"  Rec:  {metrics['recall']:.4f}")
    print(f"  F1:   {metrics['f1']:.4f}")
    print(f"  Confusion matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

    # Save metrics
    with open(output_dir / "xgboost_genetic_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Plots ──
    fig_dir = output_dir / "plots"
    fig_dir.mkdir(exist_ok=True)

    # 1. Feature importance
    fig, ax = plt.subplots(figsize=(8, 5))
    xgb.plot_importance(model, ax=ax, importance_type="gain", max_num_features=12)
    ax.set_title("XGBoost Genetic Branch — Feature Importance (Gain)")
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_importance.png", dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {fig_dir / 'feature_importance.png'}")

    # 2. ROC curve
    fig, ax = plt.subplots(figsize=(6, 6))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Genetic XGBoost")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    fig.tight_layout()
    fig.savefig(fig_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    # 3. Precision-Recall curve
    fig, ax = plt.subplots(figsize=(6, 6))
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ax.plot(recall, precision, lw=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Genetic XGBoost")
    ax.legend(loc="lower left")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    fig.tight_layout()
    fig.savefig(fig_dir / "pr_curve.png", dpi=150)
    plt.close(fig)

    # 4. Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Seizure", "Seizure"])
    ax.set_yticklabels(["No Seizure", "Seizure"])
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    ax.set_title("Confusion Matrix — Genetic XGBoost")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(fig_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    return metrics


def evaluate_regression(model, X_test, y_test, output_dir):
    """Evaluate regressor and generate plots."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    metrics = {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "n_test": len(y_test),
    }

    print(f"\nTest Results (Regression):")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")

    with open(output_dir / "xgboost_genetic_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    fig_dir = output_dir / "plots"
    fig_dir.mkdir(exist_ok=True)

    # Feature importance
    fig, ax = plt.subplots(figsize=(8, 5))
    xgb.plot_importance(model, ax=ax, importance_type="gain", max_num_features=12)
    ax.set_title("XGBoost Genetic Branch — Feature Importance (Gain)")
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_importance.png", dpi=150)
    plt.close(fig)

    # Predicted vs Actual
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.5, edgecolors="none")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=1)
    ax.set_xlabel("Actual Preictal Ratio")
    ax.set_ylabel("Predicted Preictal Ratio")
    ax.set_title(f"Predicted vs Actual — Genetic XGBoost\nMAE={mae:.4f}, RMSE={rmse:.4f}")
    fig.tight_layout()
    fig.savefig(fig_dir / "predicted_vs_actual.png", dpi=150)
    plt.close(fig)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost on genetic features")
    parser.add_argument("--data", type=str,
                        default="data/processed/genetic_vectors/genetic_training_cohort.csv",
                        help="Path to genetic training cohort CSV")
    parser.add_argument("--target", type=str, default="has_seizure",
                        choices=["has_seizure", "preictal_ratio"],
                        help="Target variable to predict")
    parser.add_argument("--output-dir", type=str, default="models/xgboost_genetic",
                        help="Directory to save model and results")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold")
    args = parser.parse_args()

    data_path = PROJECT_ROOT / args.data
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    classification = (args.target == "has_seizure")
    task_name = "classification" if classification else "regression"

    print("=" * 60)
    print(f"XGBoost Genetic Branch Training ({task_name})")
    print("=" * 60)
    print(f"Start: {datetime.now().isoformat()}")
    print(f"Data:  {data_path}")
    print(f"Out:   {output_dir}")
    t0 = time.time()

    # 1. Load data
    df = load_data(data_path)

    # 2. Prepare splits
    train, val, test = prepare_data(df, target_col=args.target)

    # 3. Build & train model
    model = build_model(classification=classification)
    model = train_model(model, train[0], train[1], val[0], val[1], output_dir)

    # 4. Evaluate
    if classification:
        metrics = evaluate_classification(model, test[0], test[1], output_dir, threshold=args.threshold)
    else:
        metrics = evaluate_regression(model, test[0], test[1], output_dir)

    # 5. Save training log
    log = {
        "timestamp": datetime.now().isoformat(),
        "data_path": str(data_path),
        "output_dir": str(output_dir),
        "n_patients": len(df),
        "n_features": len(GENETIC_FEATURES),
        "target": args.target,
        "task": task_name,
        "device": detect_device(),
        "duration_sec": round(time.time() - t0, 2),
        "metrics": metrics,
    }
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Training complete in {time.time()-t0:.1f}s")
    print(f"Outputs saved to: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
