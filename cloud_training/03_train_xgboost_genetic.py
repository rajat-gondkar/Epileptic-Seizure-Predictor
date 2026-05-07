#!/usr/bin/env python3
"""
Train XGBoost Genetic Branch (v2 — Literature-Backed)
======================================================
Trains an XGBoost classifier on the 16-dim genetic feature vector
using literature-derived risk weights and engineered features.

Follows the specification from Genetic xgboost.md:
  - Tier-weighted mutation flags (not raw binary)
  - 4 engineered features (burden, ion channel, tier1, scn1a proxy)
  - Stratified 5-fold CV on full cohort
  - LOPO evaluation on real patients
  - AUC-PR as primary metric
  - SHAP interpretability
  - Sample weights (real=3x, synthetic=1x)

Usage:
    python cloud_training/03_train_xgboost_genetic.py
    python cloud_training/03_train_xgboost_genetic.py --target preictal_ratio
"""

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")

# -- Project root --
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# Feature names (16-dim genetic vector)
# ============================================================
FEATURE_COLS = [
    # 0-8: Weighted mutation flags
    "SCN1A_mutation", "SCN8A_mutation", "KCNQ2_mutation", "SCN2A_mutation",
    "KCNT1_mutation", "DEPDC5_mutation", "PCDH19_mutation", "GRIN2A_mutation",
    "GABRA1_mutation",
    # 9-10: pLI scores
    "SCN1A_pLI", "SCN8A_pLI",
    # 11: PRS
    "polygenic_risk_score",
    # 12-15: Engineered features
    "mutation_burden", "ion_channel_burden", "tier1_flag", "scn1a_proxy",
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
    print(f"  Real: {(df['source']=='real').sum()}, Synthetic: {(df['source']=='synthetic').sum()}")
    print(f"  Seizure rate: {df['has_seizure'].mean()*100:.1f}%")
    print(f"  Genetic features: {len(FEATURE_COLS)}")
    return df


def compute_sample_weights(df):
    """Assign 3x weight to real patients, 1x to synthetic."""
    return np.where(df["source"] == "real", 3.0, 1.0)


def split_train_val_test(df, target_col, test_size=0.15, val_size=0.15, seed=42):
    """Stratified split into train/val/test."""
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)
    sample_weights = compute_sample_weights(df)

    if target_col == "has_seizure":
        X_trainval, X_test, y_trainval, y_test, sw_trainval, sw_test = train_test_split(
            X, y, sample_weights, test_size=test_size, random_state=seed, stratify=y
        )
        X_train, X_val, y_train, y_val, sw_train, sw_val = train_test_split(
            X_trainval, y_trainval, sw_trainval,
            test_size=val_size / (1 - test_size), random_state=seed, stratify=y_trainval
        )
    else:
        X_trainval, X_test, y_trainval, y_test, sw_trainval, sw_test = train_test_split(
            X, y, sample_weights, test_size=test_size, random_state=seed
        )
        X_train, X_val, y_train, y_val, sw_train, sw_val = train_test_split(
            X_trainval, y_trainval, sw_trainval,
            test_size=val_size / (1 - test_size), random_state=seed
        )

    print(f"\nData splits:")
    print(f"  Train: {len(y_train)}  (seizure={y_train.sum():.0f})")
    print(f"  Val:   {len(y_val)}    (seizure={y_val.sum():.0f})")
    print(f"  Test:  {len(y_test)}   (seizure={y_test.sum():.0f})")
    return (X_train, y_train, sw_train), (X_val, y_val, sw_val), (X_test, y_test, sw_test)


def build_model(classification=True, scale_pos_weight=1.0):
    """Build XGBoost model with literature-backed hyperparameters."""
    device = detect_device()
    print(f"\nXGBoost device: {device}")

    common = dict(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=4,
        subsample=0.75,
        colsample_bytree=0.80,
        reg_alpha=0.5,
        reg_lambda=2.0,
        tree_method="hist",
        device=device,
        random_state=42,
        n_jobs=-1,
    )

    if classification:
        return xgb.XGBClassifier(
            **common,
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
        )
    return xgb.XGBRegressor(
        **common,
        objective="reg:squarederror",
        eval_metric="rmse",
    )


def train_model(model, X_train, y_train, X_val, y_val, sw_train):
    """Train with early stopping."""
    fit_kwargs = {
        "X": X_train,
        "y": y_train,
        "sample_weight": sw_train,
        "eval_set": [(X_val, y_val)],
        "verbose": 50,
    }

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

    return model


def safe_classification_metrics(y_true, y_pred, y_prob):
    """Compute metrics safely even when one class is missing."""
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    ap = average_precision_score(y_true, y_prob) if y_true.sum() > 0 else 0.0
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Safe extraction of positive-class metrics
    pos = report.get("1", {"precision": 0.0, "recall": 0.0, "f1-score": 0.0})

    # Specificity = TN / (TN + FP)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "auc": float(auc),
        "aucpr": float(ap),
        "accuracy": float(acc),
        "precision": float(pos["precision"]),
        "recall": float(pos["recall"]),
        "f1": float(pos["f1-score"]),
        "specificity": float(specificity),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def evaluate_classification(model, X_test, y_test, output_dir, threshold=0.5):
    """Evaluate classifier and generate plots."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = safe_classification_metrics(y_test, y_pred, y_prob)
    metrics["threshold"] = threshold
    metrics["n_test"] = len(y_test)
    metrics["n_test_seizure"] = int(y_test.sum())

    print(f"\nTest Results:")
    print(f"  AUC:   {metrics['auc']:.4f}")
    print(f"  AUC-PR: {metrics['aucpr']:.4f}")
    print(f"  Acc:   {metrics['accuracy']:.4f}")
    print(f"  Prec:  {metrics['precision']:.4f}")
    print(f"  Rec:   {metrics['recall']:.4f}")
    print(f"  F1:    {metrics['f1']:.4f}")
    print(f"  Spec:  {metrics['specificity']:.4f}")
    print(f"  CM:    TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']} TP={metrics['tp']}")

    with open(output_dir / "xgboost_genetic_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # -- Plots --
    fig_dir = output_dir / "plots"
    fig_dir.mkdir(exist_ok=True)

    # 1. Feature importance
    fig, ax = plt.subplots(figsize=(8, 5))
    xgb.plot_importance(model, ax=ax, importance_type="gain", max_num_features=16)
    ax.set_title("XGBoost Genetic Branch — Feature Importance (Gain)")
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_importance.png", dpi=150)
    plt.close(fig)
    print(f"  Plot: {fig_dir / 'feature_importance.png'}")

    # 2. ROC curve
    if metrics["auc"] > 0.5:
        fig, ax = plt.subplots(figsize=(6, 6))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {metrics['auc']:.3f}")
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
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    ax.plot(recall_vals, precision_vals, lw=2, label=f"AP = {metrics['aucpr']:.3f}")
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
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
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

    # 5. SHAP summary
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(shap_values, X_test, feature_names=FEATURE_COLS, show=False)
        fig.tight_layout()
        fig.savefig(fig_dir / "shap_summary.png", dpi=150)
        plt.close(fig)
        print(f"  Plot: {fig_dir / 'shap_summary.png'}")
    except Exception as e:
        print(f"  [WARNING] SHAP plot skipped: {e}")

    return metrics


def run_lopo(df, output_dir, target_col="has_seizure"):
    """Leave-One-Patient-Out evaluation on real patients."""
    real_df = df[df["source"] == "real"].copy()
    if len(real_df) == 0:
        print("\n[LOPO] No real patients found — skipping.")
        return None

    print("\n" + "=" * 60)
    print("LEAVE-ONE-PATIENT-OUT (LOPO) EVALUATION")
    print("=" * 60)

    real_patients = real_df["patient_id"].unique().tolist()
    lopo_results = []

    for test_patient in real_patients:
        train_df = real_df[real_df["patient_id"] != test_patient]
        test_df  = real_df[real_df["patient_id"] == test_patient]

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        X_train = train_df[FEATURE_COLS].values.astype(np.float32)
        y_train = train_df[target_col].values.astype(np.float32)
        X_test = test_df[FEATURE_COLS].values.astype(np.float32)
        y_test = test_df[target_col].values.astype(np.float32)

        # Add synthetic data as training support
        synth_df = df[df["source"] == "synthetic"]
        X_synth = synth_df[FEATURE_COLS].values.astype(np.float32)
        y_synth = synth_df[target_col].values.astype(np.float32)
        sw_synth = compute_sample_weights(synth_df)

        X_train = np.vstack([X_train, X_synth])
        y_train = np.concatenate([y_train, y_synth])
        sw_train = np.concatenate([np.ones(len(train_df)) * 3.0, sw_synth])

        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        spw = n_neg / max(n_pos, 1)

        model = build_model(classification=True, scale_pos_weight=spw)
        model.fit(X_train, y_train, sample_weight=sw_train, verbose=False)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = safe_classification_metrics(y_test, y_pred, y_prob)
        metrics["test_patient"] = test_patient
        lopo_results.append(metrics)

        print(f"  {test_patient}: AUC={metrics['auc']:.3f} AUC-PR={metrics['aucpr']:.3f} "
              f"Prec={metrics['precision']:.3f} Rec={metrics['recall']:.3f}")

    if lopo_results:
        avg_auc = np.mean([r["auc"] for r in lopo_results])
        avg_aucpr = np.mean([r["aucpr"] for r in lopo_results])
        print(f"\nLOPO Average: AUC={avg_auc:.4f}  AUC-PR={avg_aucpr:.4f}")
        with open(output_dir / "lopo_results.json", "w") as f:
            json.dump({"results": lopo_results, "avg_auc": avg_auc, "avg_aucpr": avg_aucpr}, f, indent=2)
        return {"results": lopo_results, "avg_auc": avg_auc, "avg_aucpr": avg_aucpr}
    return None


def run_stratified_cv(df, target_col="has_seizure", n_splits=5, seed=42):
    """5-fold stratified cross-validation."""
    print("\n" + "=" * 60)
    print(f"STRATIFIED {n_splits}-FOLD CROSS-VALIDATION")
    print("=" * 60)

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)
    sw = compute_sample_weights(df)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        sw_train = sw[train_idx]

        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        spw = n_neg / max(n_pos, 1)

        model = build_model(classification=True, scale_pos_weight=spw)
        model.fit(X_train, y_train, sample_weight=sw_train, verbose=False)

        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = safe_classification_metrics(y_val, y_pred, y_prob)
        cv_scores.append(metrics)

        print(f"  Fold {fold}: AUC={metrics['auc']:.4f} AUC-PR={metrics['aucpr']:.4f} "
              f"F1={metrics['f1']:.4f} Spec={metrics['specificity']:.4f}")

    avg = {k: float(np.mean([s[k] for s in cv_scores])) for k in ["auc", "aucpr", "accuracy", "precision", "recall", "f1", "specificity"]}
    print(f"\nCV Average: AUC={avg['auc']:.4f} AUC-PR={avg['aucpr']:.4f} F1={avg['f1']:.4f}")
    return {"folds": cv_scores, "average": avg}


def evaluate_regression(model, X_test, y_test, output_dir):
    """Evaluate regressor and generate plots."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    metrics = {"mae": float(mae), "rmse": float(rmse), "n_test": len(y_test)}
    print(f"\nTest Results (Regression): MAE={mae:.4f} RMSE={rmse:.4f}")

    with open(output_dir / "xgboost_genetic_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    fig_dir = output_dir / "plots"
    fig_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    xgb.plot_importance(model, ax=ax, importance_type="gain", max_num_features=16)
    ax.set_title("XGBoost Genetic Branch — Feature Importance (Gain)")
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_importance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.5, edgecolors="none")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=1)
    ax.set_xlabel("Actual Preictal Ratio")
    ax.set_ylabel("Predicted Preictal Ratio")
    ax.set_title(f"Predicted vs Actual\nMAE={mae:.4f} RMSE={rmse:.4f}")
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
                        help="Target variable")
    parser.add_argument("--output-dir", type=str, default="models/xgboost_genetic",
                        help="Directory to save outputs")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold")
    parser.add_argument("--lopo", action="store_true", default=True,
                        help="Run LOPO evaluation on real patients")
    parser.add_argument("--cv", action="store_true", default=True,
                        help="Run stratified 5-fold CV")
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

    # 1. Load
    df = load_data(data_path)

    # 2. Compute fresh scale_pos_weight
    y_all = df[args.target].values
    n_neg = (y_all == 0).sum()
    n_pos = (y_all == 1).sum()
    spw = n_neg / max(n_pos, 1)
    print(f"  scale_pos_weight: {spw:.2f} (neg={n_neg}, pos={n_pos})")

    # 3. Split
    train, val, test = split_train_val_test(df, args.target)

    # 4. Build & train
    model = build_model(classification=classification, scale_pos_weight=spw)
    model = train_model(model, train[0], train[1], val[0], val[1], train[2])

    # 5. Save model
    model_path = output_dir / "xgboost_genetic_model.pkl"
    joblib.dump(model, model_path)
    print(f"  Model saved: {model_path}")

    # 6. Test evaluation
    if classification:
        metrics = evaluate_classification(model, test[0], test[1], output_dir, threshold=args.threshold)
    else:
        metrics = evaluate_regression(model, test[0], test[1], output_dir)

    # 7. Stratified CV
    cv_results = None
    if args.cv and classification:
        cv_results = run_stratified_cv(df, target_col=args.target)
        if cv_results:
            with open(output_dir / "cv_results.json", "w") as f:
                json.dump(cv_results, f, indent=2, default=float)

    # 8. LOPO
    lopo_results = None
    if args.lopo and classification:
        lopo_results = run_lopo(df, output_dir, target_col=args.target)

    # 9. Save training log
    log = {
        "timestamp": datetime.now().isoformat(),
        "data_path": str(data_path),
        "output_dir": str(output_dir),
        "n_patients": len(df),
        "n_features": len(FEATURE_COLS),
        "target": args.target,
        "task": task_name,
        "scale_pos_weight": float(spw),
        "device": detect_device(),
        "duration_sec": round(time.time() - t0, 2),
        "metrics": metrics,
        "cv_average": cv_results.get("average") if cv_results else None,
        "lopo_average": {"auc": lopo_results.get("avg_auc"), "aucpr": lopo_results.get("avg_aucpr")} if lopo_results else None,
    }
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2, default=float)

    print(f"\n{'=' * 60}")
    print(f"Training complete in {time.time()-t0:.1f}s")
    print(f"Outputs: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
