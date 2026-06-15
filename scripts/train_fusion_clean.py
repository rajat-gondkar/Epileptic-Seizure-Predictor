#!/usr/bin/env python3
"""
Fusion Layer Training Pipeline (Clean Version)
================================================
Trains the attention-gated fusion layer combining real EEG embeddings
with real genetic risk scores.

Data flow:
1. Load cached EEG embeddings (512-dim from BiLSTM)
2. Load real genetic data (CTGAN, 10K patients, 22 features)
3. Compute genetic risk scores via XGBoost
4. Align: assign each of 11 CHB-MIT patients a genetic risk score
5. Train attention gate

Usage:
    python scripts/train_fusion_clean.py --epochs 30
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'data_pipeline'))

from src.training.fusion import AttentionGateFusion, FusionDataset, FusionTrainer
from genetic_feature_engineering import FEATURE_NAMES


OUTPUT_DIR = PROJECT_ROOT / 'models' / 'fusion'
CACHE_FILE = OUTPUT_DIR / 'eeg_embeddings.npz'


def load_genetic_data_and_scores(genetic_csv, xgb_model_path):
    """Load genetic data and compute risk scores."""
    print(f"\n  Loading genetic data from {genetic_csv}")
    df = pd.read_csv(genetic_csv)
    
    # Select only the 22 features XGBoost was trained on
    available = [f for f in FEATURE_NAMES if f in df.columns]
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        print(f"    WARNING: Missing features: {missing}")
    
    genetic_features = df[available].values.astype(np.float32)
    labels = df['has_seizure'].values
    
    print(f"    Patients: {len(labels)}")
    print(f"    Seizure rate: {labels.mean()*100:.1f}% ({int(labels.sum())} positive)")
    print(f"    Features: {genetic_features.shape[1]} dimensions")
    
    # Compute risk scores
    print(f"  Computing genetic risk scores from XGBoost...")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(xgb_model_path)
    risk_scores = xgb_model.predict_proba(genetic_features)[:, 1].reshape(-1, 1)
    
    print(f"    Risk range: [{risk_scores.min():.3f}, {risk_scores.max():.3f}]")
    print(f"    Mean risk: {risk_scores.mean():.3f}")
    
    return risk_scores, labels


def align_genetic_to_eeg(eeg_labels, genetic_risk_scores, seed=42):
    """
    Align genetic risk scores to EEG windows.
    
    Since genetic data (10K synthetic patients) doesn't map to
    CHB-MIT patients (11 real patients), we assign each EEG window
    a genetic risk score sampled randomly from the XGBoost distribution.
    
    Scores are assigned per-window (NOT per-label) to avoid leaking
    label information through the genetic channel.
    """
    rng = np.random.RandomState(seed)
    
    print(f"\n  Aligning {len(eeg_labels)} EEG windows to genetic risk scores")
    print(f"    Genetic distribution: mean={genetic_risk_scores.mean():.3f}, "
          f"std={genetic_risk_scores.std():.3f}")
    
    # Assign a random genetic risk score to each window independently
    eeg_genetic_scores = rng.choice(
        genetic_risk_scores.flatten(), size=len(eeg_labels), replace=True
    ).reshape(-1, 1)
    
    print(f"    Assigned risk scores to {len(eeg_genetic_scores)} windows")
    print(f"    Assigned range: [{eeg_genetic_scores.min():.3f}, {eeg_genetic_scores.max():.3f}]")
    print(f"    NOTE: Genetic scores are random per-window (no label leakage)")
    
    return eeg_genetic_scores


def plot_training_curves(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(history['train_losses'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(history['val_losses'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history['val_aucs'], 'g-', label='Val AUC', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUC')
    ax2.set_title('Validation AUC')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path / 'training_curves.png'}")


def plot_confusion_matrix(y_true, y_prob, threshold, save_path):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Seizure', 'Seizure'],
                yticklabels=['No Seizure', 'Seizure'],
                ax=ax, annot_kws={'size': 16})
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('True', fontsize=14)
    ax.set_title(f'Confusion Matrix (threshold={threshold:.2f})', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(save_path / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path / 'confusion_matrix.png'}")


def plot_roc_curve(y_true, y_prob, auc_score, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('ROC Curve - Fusion Model', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path / 'roc_curve.png'}")


def plot_pr_curve(y_true, y_prob, ap_score, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, 'b-', linewidth=2, label=f'PR (AP = {ap_score:.3f})')
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title('Precision-Recall Curve - Fusion Model', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'pr_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path / 'pr_curve.png'}")


def plot_attention_distribution(attention_weights, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    unique_vals = np.unique(attention_weights)
    if len(unique_vals) <= 1:
        ax.bar([unique_vals[0]], [len(attention_weights)], width=0.02, 
               edgecolor='black', alpha=0.7)
    else:
        n_bins = min(50, len(unique_vals))
        ax.hist(attention_weights, bins=n_bins, edgecolor='black', alpha=0.7)
    
    ax.axvline(x=np.mean(attention_weights), color='r', linestyle='--', 
               label=f'Mean = {np.mean(attention_weights):.3f}')
    ax.set_xlabel('Attention Weight (EEG contribution)', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('Distribution of Attention Weights', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'attention_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path / 'attention_distribution.png'}")


def plot_score_comparison(y_true, eeg_scores, genetic_scores, fused_scores, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, scores, title in zip(
        axes,
        [eeg_scores, genetic_scores, fused_scores],
        ['EEG Score', 'Genetic Score', 'Fused Score']
    ):
        if len(np.unique(y_true)) > 1:
            ax.hist(scores[y_true == 0], bins=30, alpha=0.5, label='No Seizure', density=True)
            ax.hist(scores[y_true == 1], bins=30, alpha=0.5, label='Seizure', density=True)
        else:
            ax.hist(scores, bins=30, alpha=0.7, density=True)
        ax.set_xlabel('Score')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'score_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path / 'score_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description="Train Fusion Layer")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--genetic-csv', type=str,
                        default='data/processed/genetic_vectors/genetic_training_cohort.csv')
    parser.add_argument('--xgb-model', type=str,
                        default='models/xgboost_genetic/xgboost_genetic_model.pkl')
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = OUTPUT_DIR / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("=" * 60)
    print("Fusion Layer Training Pipeline")
    print("=" * 60)
    print(f"Timestamp: {timestamp}")
    
    # ---- Step 1: Load EEG embeddings ----
    print(f"\n--- Step 1: Loading EEG Embeddings ---")
    if not CACHE_FILE.exists():
        print(f"ERROR: Cache file not found: {CACHE_FILE}")
        print(f"Run: python scripts/extract_eeg_embeddings.py")
        sys.exit(1)
    
    cache = np.load(CACHE_FILE)
    eeg_embeddings = cache['embeddings']
    eeg_labels_raw = cache['labels']

    # CHBMITDataset returns 3-class labels: 0=interictal, 1=preictal, 2=ictal
    # Binarize: 0 → 0 (healthy), 1 or 2 → 1 (seizure risk)
    eeg_labels = (eeg_labels_raw > 0).astype(np.float32)

    n_preictal = int((eeg_labels_raw == 1).sum())
    n_ictal = int((eeg_labels_raw == 2).sum())
    print(f"  Loaded: {eeg_embeddings.shape} embeddings")
    print(f"  Labels: {eeg_labels.shape} ({int(eeg_labels.sum())} positive)")
    print(f"    Preictal (1): {n_preictal}  |  Ictal (2): {n_ictal}  |  Interictal (0): {int(len(eeg_labels) - n_preictal - n_ictal)}")
    
    # ---- Step 2: Load genetic data and compute scores ----
    print(f"\n--- Step 2: Loading Genetic Data ---")
    genetic_scores, genetic_labels = load_genetic_data_and_scores(
        args.genetic_csv, args.xgb_model
    )
    
    # ---- Step 3: Align genetic scores to EEG windows ----
    print(f"\n--- Step 3: Aligning Genetic Scores to EEG ---")
    genetic_scores_aligned = align_genetic_to_eeg(
        eeg_labels, genetic_scores, seed=args.seed
    )
    
    # Use EEG labels (from real data) as ground truth
    labels = eeg_labels
    
    print(f"\n  Final dataset:")
    print(f"    EEG embeddings: {eeg_embeddings.shape}")
    print(f"    Genetic scores: {genetic_scores_aligned.shape}")
    print(f"    Labels: {labels.shape} ({labels.sum()} positive, {labels.mean()*100:.1f}%)")
    
    # ---- Step 4: Split data ----
    print(f"\n--- Step 4: Splitting Data (60/20/20) ---")
    
    X_eeg_temp, X_eeg_test, X_gen_temp, X_gen_test, y_temp, y_test = train_test_split(
        eeg_embeddings, genetic_scores_aligned, labels,
        test_size=0.2, stratify=labels, random_state=args.seed
    )
    
    X_eeg_train, X_eeg_val, X_gen_train, X_gen_val, y_train, y_val = train_test_split(
        X_eeg_temp, X_gen_temp, y_temp,
        test_size=0.25, stratify=y_temp, random_state=args.seed
    )
    
    print(f"  Train: {len(y_train)} ({y_train.sum()} positive)")
    print(f"  Val:   {len(y_val)} ({y_val.sum()} positive)")
    print(f"  Test:  {len(y_test)} ({y_test.sum()} positive)")
    
    # ---- Step 5: Create datasets and train ----
    print(f"\n--- Step 5: Training Fusion Model ---")
    
    train_dataset = FusionDataset(X_eeg_train, X_gen_train, y_train)
    val_dataset = FusionDataset(X_eeg_val, X_gen_val, y_val)
    test_dataset = FusionDataset(X_eeg_test, X_gen_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = AttentionGateFusion(
        eeg_embedding_dim=512,
        genetic_dim=1,
        hidden_dim=128,
        dropout=0.3,
    )
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = FusionTrainer(
        model=model,
        device='cpu',
        learning_rate=args.lr,
        weight_decay=1e-4,
        l1_alpha=0.01,
    )
    
    save_path = OUTPUT_DIR / 'fusion_best.pt'
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_path=str(save_path),
    )
    
    # ---- Step 6: Evaluate ----
    print(f"\n--- Step 6: Evaluating on Test Set ---")
    
    model.load_state_dict(torch.load(save_path, weights_only=False))
    model.eval()
    
    all_preds = []
    all_labels = []
    all_attention = []
    
    with torch.no_grad():
        for eeg_emb, gen_score, labels_batch in test_loader:
            risk_score, alpha = model(eeg_emb, gen_score)
            all_preds.extend(risk_score.numpy().flatten())
            all_labels.extend(labels_batch.numpy().flatten())
            all_attention.extend(alpha.numpy().flatten())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_attention = np.array(all_attention)
    
    # Find optimal threshold
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_preds)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores[:-1])
    optimal_threshold = thresholds[best_idx]
    
    y_pred = (all_preds >= optimal_threshold).astype(int)
    
    auc_score = roc_auc_score(all_labels, all_preds)
    ap_score = average_precision_score(all_labels, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, y_pred).ravel()
    
    test_metrics = {
        'auc': float(auc_score),
        'aucpr': float(ap_score),
        'threshold': float(optimal_threshold),
        'accuracy': float(accuracy_score(all_labels, y_pred)),
        'precision': float(precision_score(all_labels, y_pred, zero_division=0)),
        'recall': float(recall_score(all_labels, y_pred, zero_division=0)),
        'f1': float(f1_score(all_labels, y_pred, zero_division=0)),
        'specificity': float(tn / (tn + fp) if (tn + fp) > 0 else 0),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'n_test': len(all_labels),
        'n_test_seizure': int(all_labels.sum()),
        'mean_attention_eeg': float(all_attention.mean()),
        'mean_attention_genetic': float(1 - all_attention.mean()),
    }
    
    print(f"\n  === Test Metrics ===")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"    {k:>25s}: {v:.4f}")
        else:
            print(f"    {k:>25s}: {v}")
    
    # ---- Step 7: Save results and plots ----
    print(f"\n--- Step 7: Saving Results ---")
    
    torch.save(model.state_dict(), OUTPUT_DIR / 'fusion_final.pt')
    
    results = {
        'timestamp': timestamp,
        'config': vars(args),
        'data': {
            'train_samples': len(y_train),
            'val_samples': len(y_val),
            'test_samples': len(y_test),
            'positive_rate': float(labels.mean()),
            'eeg_embedding_dim': 512,
        },
        'test_metrics': test_metrics,
        'history': history,
    }
    
    with open(OUTPUT_DIR / 'fusion_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Generating plots...")
    plot_training_curves(history, plots_dir)
    plot_confusion_matrix(all_labels, all_preds, optimal_threshold, plots_dir)
    plot_roc_curve(all_labels, all_preds, auc_score, plots_dir)
    plot_pr_curve(all_labels, all_preds, ap_score, plots_dir)
    plot_attention_distribution(all_attention, plots_dir)
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  AUC: {auc_score:.4f}")
    print(f"  AUC-PR: {ap_score:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  Mean attention (EEG): {all_attention.mean():.3f}")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
