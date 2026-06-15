#!/usr/bin/env python3
"""
Fusion Model Evaluation
=======================
Loads the same data as train_fusion_clean.py, reproduces the split,
loads the trained fusion model, and compares Fusion vs EEG-only vs Genetic-only.

Usage:
    python scripts/evaluate_fusion.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score,
    matthews_corrcoef
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'data_pipeline'))

from src.training.fusion import AttentionGateFusion, FusionDataset
from genetic_feature_engineering import FEATURE_NAMES

OUTPUT_DIR = PROJECT_ROOT / 'models' / 'fusion'
CACHE_FILE = OUTPUT_DIR / 'eeg_embeddings.npz'
PLOTS_DIR = OUTPUT_DIR / 'plots'
SEED = 42


def load_data():
    """Reproduce the exact same data pipeline as training."""
    # 1. Load EEG embeddings
    cache = np.load(CACHE_FILE)
    eeg_embeddings = cache['embeddings']
    eeg_labels_raw = cache['labels']
    eeg_labels = (eeg_labels_raw > 0).astype(np.float32)

    # 2. Load genetic data and compute risk scores
    genetic_csv = 'data/processed/genetic_vectors/genetic_training_cohort.csv'
    xgb_path = 'models/xgboost_genetic/xgboost_genetic_model.pkl'

    df = pd.read_csv(genetic_csv)
    available = [f for f in FEATURE_NAMES if f in df.columns]
    genetic_features = df[available].values.astype(np.float32)

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(xgb_path)
    genetic_risk_scores = xgb_model.predict_proba(genetic_features)[:, 1]

    # 3. Align genetic scores to EEG windows
    rng = np.random.RandomState(SEED)
    unique_labels = np.unique(eeg_labels_raw)
    patient_scores = rng.choice(genetic_risk_scores, size=len(unique_labels), replace=True)
    label_to_score = dict(zip(unique_labels, patient_scores))
    genetic_scores_aligned = np.array([label_to_score[l] for l in eeg_labels_raw]).reshape(-1, 1)

    # 4. Reproduce exact same split
    X_eeg_temp, X_eeg_test, X_gen_temp, X_gen_test, y_temp, y_test = train_test_split(
        eeg_embeddings, genetic_scores_aligned, eeg_labels,
        test_size=0.2, stratify=eeg_labels, random_state=SEED
    )
    X_eeg_train, X_eeg_val, X_gen_train, X_gen_val, y_train, y_val = train_test_split(
        X_eeg_temp, X_gen_temp, y_temp,
        test_size=0.25, stratify=y_temp, random_state=SEED
    )

    return X_eeg_test, X_gen_test, y_test, eeg_embeddings, genetic_scores_aligned, eeg_labels


def load_fusion_model():
    """Load trained fusion model."""
    model = AttentionGateFusion(eeg_embedding_dim=512, genetic_dim=1, hidden_dim=128, dropout=0.3)
    model.load_state_dict(torch.load(OUTPUT_DIR / 'fusion_best.pt', map_location='cpu', weights_only=False))
    model.eval()
    return model


def evaluate_branch(y_true, scores, name):
    """Compute metrics for one model."""
    auc = roc_auc_score(y_true, scores)
    ap = average_precision_score(y_true, scores)

    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1s[:-1])
    opt_threshold = thresholds[best_idx]

    y_pred = (scores >= opt_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        'branch': name,
        'auc': float(auc),
        'aucpr': float(ap),
        'threshold': float(opt_threshold),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'specificity': float(tn / (tn + fp) if (tn + fp) > 0 else 0),
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Fusion Model Evaluation")
    print("=" * 70)

    # ---- Load data ----
    print("\n--- Loading Data ---")
    X_eeg_test, X_gen_test, y_test, all_eeg, all_gen, all_labels = load_data()
    print(f"  Test samples: {len(y_test)}")
    print(f"  Test positive: {int(y_test.sum())} ({y_test.mean()*100:.1f}%)")

    # ---- Load model ----
    print("\n--- Loading Fusion Model ---")
    model = load_fusion_model()
    print(f"  Model loaded from {OUTPUT_DIR / 'fusion_best.pt'}")

    # ---- Get fusion predictions ----
    test_dataset = FusionDataset(X_eeg_test, X_gen_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    fused_scores_list = []
    attention_list = []

    with torch.no_grad():
        for eeg_emb, gen_score, _ in test_loader:
            risk, alpha = model(eeg_emb, gen_score)
            fused_scores_list.extend(risk.numpy().flatten())
            attention_list.extend(alpha.numpy().flatten())

    fused_scores = np.array(fused_scores_list)
    attention_weights = np.array(attention_list)

    # ---- Get EEG-only scores ----
    eeg_scores_list = []
    with torch.no_grad():
        for eeg_emb, gen_score, _ in test_loader:
            eeg_proj = model.eeg_projection(eeg_emb)
            gen_proj = model.genetic_projection(gen_score)
            combined = torch.cat([eeg_proj, gen_proj], dim=1)
            alpha = model.attention(combined)
            eeg_only = alpha.squeeze() * eeg_proj.mean(dim=1)
            eeg_scores_list.extend(eeg_only.numpy().flatten())

    eeg_scores = np.array(eeg_scores_list)

    # ---- Get genetic-only scores ----
    genetic_scores_flat = X_gen_test.flatten()

    # ---- Evaluate ----
    print("\n--- Branch Evaluation ---")
    eeg_m = evaluate_branch(y_test, eeg_scores, 'EEG-only')
    gen_m = evaluate_branch(y_test, genetic_scores_flat, 'Genetic-only')
    fuse_m = evaluate_branch(y_test, fused_scores, 'Fusion')
    all_m = [eeg_m, gen_m, fuse_m]

    for m in all_m:
        print(f"\n  {m['branch']}:")
        print(f"    AUC:         {m['auc']:.4f}")
        print(f"    AUC-PR:      {m['aucpr']:.4f}")
        print(f"    F1:          {m['f1']:.4f}")
        print(f"    Precision:   {m['precision']:.4f}")
        print(f"    Recall:      {m['recall']:.4f}")
        print(f"    Specificity: {m['specificity']:.4f}")
        print(f"    MCC:         {m['mcc']:.4f}")

    # ---- Improvement ----
    print("\n--- Improvement Analysis ---")
    print(f"  Fusion vs EEG-only:   AUC {fuse_m['auc'] - eeg_m['auc']:+.4f}  F1 {fuse_m['f1'] - eeg_m['f1']:+.4f}")
    print(f"  Fusion vs Genetic:    AUC {fuse_m['auc'] - gen_m['auc']:+.4f}  F1 {fuse_m['f1'] - gen_m['f1']:+.4f}")

    # ---- Attention analysis ----
    print("\n--- Attention Analysis ---")
    print(f"  Mean EEG attention:     {attention_weights.mean():.3f}")
    print(f"  Mean Genetic attention: {1 - attention_weights.mean():.3f}")
    print(f"  Std attention:          {attention_weights.std():.3f}")
    if y_test.sum() > 0:
        print(f"  Attention (seizure):    {attention_weights[y_test == 1].mean():.3f}")
    print(f"  Attention (non-seizure):{attention_weights[y_test == 0].mean():.3f}")

    # ==================== PLOTS ====================
    print("\n--- Generating Plots ---")
    scores_dict = {'EEG-only': eeg_scores, 'Genetic-only': genetic_scores_flat, 'Fusion': fused_scores}
    colors = {'EEG-only': '#2196F3', 'Genetic-only': '#4CAF50', 'Fusion': '#FF9800'}

    # 1. ROC comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, sc in scores_dict.items():
        fpr, tpr, _ = roc_curve(y_test, sc)
        ax.plot(fpr, tpr, color=colors[name], linewidth=2,
                label=f'{name} (AUC={roc_auc_score(y_test, sc):.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('ROC Curve Comparison', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'roc_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: roc_comparison.png")

    # 2. PR comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, sc in scores_dict.items():
        prec, rec, _ = precision_recall_curve(y_test, sc)
        ax.plot(rec, prec, color=colors[name], linewidth=2,
                label=f'{name} (AP={average_precision_score(y_test, sc):.3f})')
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title('Precision-Recall Curve Comparison', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'pr_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: pr_comparison.png")

    # 3. Bar chart comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    branches = [m['branch'] for m in all_m]
    metrics_to_plot = ['auc', 'f1', 'recall', 'specificity']
    x = np.arange(len(branches))
    width = 0.2
    bar_colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    for i, metric in enumerate(metrics_to_plot):
        values = [m[metric] for m in all_m]
        ax.bar(x + i * width, values, width, label=metric.upper(), color=bar_colors[i], alpha=0.8)
    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Performance Comparison', fontsize=16)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(branches, fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'improvement_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: improvement_comparison.png")

    # 4. Attention distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(attention_weights, bins=50, color='#FF9800', alpha=0.7, edgecolor='black')
    ax.axvline(attention_weights.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean={attention_weights.mean():.3f}')
    ax.set_xlabel('Attention Weight (EEG)', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('Attention Weight Distribution', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'attention_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: attention_distribution.png")

    # 5. Confusion matrix
    y_pred = (fused_scores >= fuse_m['threshold']).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Non-seizure', 'Seizure'],
                yticklabels=['Non-seizure', 'Seizure'])
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('Actual', fontsize=14)
    ax.set_title('Confusion Matrix (Fusion)', fontsize=16)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: confusion_matrix.png")

    # ---- Save JSON ----
    results = {
        'test_samples': len(y_test),
        'positive_samples': int(y_test.sum()),
        'positive_rate': float(y_test.mean()),
        'metrics': {m['branch']: m for m in all_m},
        'attention': {
            'mean_eeg': float(attention_weights.mean()),
            'mean_genetic': float(1 - attention_weights.mean()),
            'std': float(attention_weights.std()),
        },
    }
    with open(OUTPUT_DIR / 'fusion_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\n  {'Model':<20s} {'AUC':>8s} {'AUC-PR':>8s} {'F1':>8s} {'Recall':>8s}")
    print(f"  {'-'*56}")
    for m in all_m:
        print(f"  {m['branch']:<20s} {m['auc']:>8.4f} {m['aucpr']:>8.4f} {m['f1']:>8.4f} {m['recall']:>8.4f}")
    print(f"\n  All plots saved to: {PLOTS_DIR}")


if __name__ == '__main__':
    main()
