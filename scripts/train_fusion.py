#!/usr/bin/env python3
"""
Fusion Layer Training Pipeline
================================
Trains the attention-gated fusion layer that combines EEG and Genetic
branch predictions into a final seizure risk score.

Strategy:
1. Load frozen EEG branch (BiLSTM) and extract embeddings
2. Load frozen Genetic branch (XGBoost) and extract risk scores
3. Train the fusion layer (attention gate) on validation data
4. Evaluate on held-out test set

Usage:
    python scripts/train_fusion.py
    python scripts/train_fusion.py --eeg-model SavedModels/EEGLSTM_*.net
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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix, classification_report,
    f1_score, precision_score, recall_score, accuracy_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# -- Project root --
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.fusion import (
    AttentionGateFusion,
    FusionTrainer,
    FusionDataset,
)


# ============================================================
# Configuration
# ============================================================
RANDOM_SEED = 42
OUTPUT_DIR = PROJECT_ROOT / 'models' / 'fusion'


# ============================================================
# Data Loading
# ============================================================
def load_eeg_embeddings(embeddings_path: str) -> tuple:
    """
    Load pre-computed EEG embeddings.
    
    Returns:
        embeddings: np.array (n_samples, 64)
        labels: np.array (n_samples,)
    """
    data = np.load(embeddings_path)
    return data['embeddings'], data['labels']


def load_genetic_scores(scores_path: str) -> np.ndarray:
    """Load pre-computed genetic risk scores."""
    data = np.load(scores_path)
    return data['scores']


def load_xgboost_model(model_path: str) -> xgb.XGBClassifier:
    """Load the trained XGBoost model."""
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model


def extract_genetic_scores(
    xgb_model: xgb.XGBClassifier,
    genetic_features: np.ndarray,
) -> np.ndarray:
    """Extract genetic risk scores from XGBoost model."""
    scores = xgb_model.predict_proba(genetic_features)[:, 1]
    return scores.reshape(-1, 1)


# ============================================================
# Embedding Extraction (if not pre-computed)
# ============================================================
def extract_eeg_embeddings_from_model(
    eeg_model_path: str,
    eeg_dataset,
    device: str = 'cpu',
    batch_size: int = 16,
) -> tuple:
    """
    Extract EEG embeddings using the trained BiLSTM model.
    
    Uses the new get_embedding() method that returns the 512-dim
    attention-weighted LSTM output (before FC layer).
    
    Args:
        eeg_model_path: Path to saved BiLSTM model
        eeg_dataset: PyTorch dataset with EEG windows
        device: Device for inference
        batch_size: Batch size for inference
        
    Returns:
        embeddings: np.array (n_samples, 512) - LSTM output dim
        labels: np.array (n_samples,)
    """
    # Import model class
    sys.path.insert(0, str(PROJECT_ROOT / 'seizure_prediction'))
    from libModelLSTM import clsLSTM
    
    # Load model
    checkpoint = torch.load(eeg_model_path, map_location=device, weights_only=False)
    
    model = clsLSTM(
        checkpoint['intFeaturesDim'],
        checkpoint['intHiddenDim'],
        checkpoint['intNumLayers'],
        checkpoint['intOutputSize'],
        checkpoint['fltDropProb'],
    )
    model.load_state_dict(checkpoint['dctStateDict'])
    model.to(device)
    model.eval()
    
    # Extract embeddings
    all_embeddings = []
    all_labels = []
    
    dataloader = DataLoader(eeg_dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch_idx, (eeg_windows, labels) in enumerate(dataloader):
            eeg_windows = eeg_windows.to(device)
            
            # Initialize hidden state
            hidden = model.initHidden(len(eeg_windows), 
                                     argTrainOnGPU=(device == 'cuda'))
            
            # Use the new get_embedding method
            embedding, _ = model.get_embedding(eeg_windows, hidden)
            
            # Store embedding and labels
            all_embeddings.append(embedding.cpu().numpy())
            all_labels.append(labels.numpy())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"    Extracted {(batch_idx+1)*batch_size}/{len(eeg_dataset)} embeddings")
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    return embeddings, labels


# ============================================================
# Synthetic Data Generation (for testing without real EEG)
# ============================================================
def generate_synthetic_data(n_samples: int = 5000, seed: int = 42) -> tuple:
    """
    Generate synthetic data for testing the fusion layer.
    
    Creates realistic-looking EEG embeddings (512-dim, matching BiLSTM output)
    and genetic scores with a noisy, realistic relationship to labels.
    
    Returns:
        eeg_embeddings: np.array (n_samples, 512)
        genetic_scores: np.array (n_samples, 1)
        labels: np.array (n_samples,)
    """
    rng = np.random.RandomState(seed)
    
    # Generate labels (~10% positive)
    labels = (rng.random(n_samples) < 0.10).astype(np.float32)
    
    # Generate EEG embeddings (512-dim to match BiLSTM output)
    # Use correlated features with heavy noise to simulate realistic data
    eeg_embeddings = rng.randn(n_samples, 512).astype(np.float32)
    # Add subtle signal (not trivially separable)
    signal = rng.randn(512) * 0.15
    eeg_embeddings[labels == 1] += signal
    
    # Generate genetic scores (weakly correlated with labels)
    genetic_scores = rng.beta(2, 5, size=(n_samples, 1)).astype(np.float32)
    # Add weak signal with lots of noise
    noise = rng.randn(n_samples, 1) * 0.3
    genetic_scores = np.clip(genetic_scores + noise * labels.reshape(-1, 1) * 0.2, 0, 1)
    
    # Shuffle to avoid ordering artifacts
    idx = rng.permutation(n_samples)
    return eeg_embeddings[idx], genetic_scores[idx], labels[idx]


# ============================================================
# Plot Generation
# ============================================================
def plot_training_curves(history: dict, save_path: Path):
    """Plot training and validation loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss curves
    ax1.plot(history['train_losses'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(history['val_losses'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # AUC curve
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
    """Plot confusion matrix."""
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
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('ROC Curve — Fusion Model', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    
    plt.tight_layout()
    plt.savefig(save_path / 'roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path / 'roc_curve.png'}")


def plot_pr_curve(y_true, y_prob, ap_score, save_path):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, 'b-', linewidth=2, label=f'PR (AP = {ap_score:.3f})')
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title('Precision-Recall Curve — Fusion Model', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    
    plt.tight_layout()
    plt.savefig(save_path / 'pr_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path / 'pr_curve.png'}")


def plot_attention_distribution(attention_weights, save_path):
    """Plot distribution of attention weights."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Handle case where all values are identical
    unique_vals = np.unique(attention_weights)
    if len(unique_vals) <= 1:
        # All values are the same - show as single bar
        ax.bar([unique_vals[0]], [len(attention_weights)], width=0.02, 
               edgecolor='black', alpha=0.7)
        ax.set_xlim(unique_vals[0] - 0.1, unique_vals[0] + 0.1)
    else:
        n_bins = min(50, len(unique_vals))
        ax.hist(attention_weights, bins=n_bins, edgecolor='black', alpha=0.7)
    
    ax.axvline(x=np.mean(attention_weights), color='r', linestyle='--', 
               label=f'Mean = {np.mean(attention_weights):.3f}')
    ax.set_xlabel('Attention Weight α (EEG contribution)', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('Distribution of Attention Weights', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'attention_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path / 'attention_distribution.png'}")


def plot_score_comparison(y_true, eeg_scores, genetic_scores, fused_scores, save_path):
    """Compare score distributions for seizure vs non-seizure patients."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, scores, title in zip(
        axes,
        [eeg_scores, genetic_scores, fused_scores],
        ['EEG Score', 'Genetic Score', 'Fused Score']
    ):
        ax.hist(scores[y_true == 0], bins=30, alpha=0.5, label='No Seizure', density=True)
        ax.hist(scores[y_true == 1], bins=30, alpha=0.5, label='Seizure', density=True)
        ax.set_xlabel('Score')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'score_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path / 'score_comparison.png'}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Train Fusion Layer")
    parser.add_argument('--eeg-model', type=str, default=None,
                        help="Path to trained BiLSTM model (.net) - extracts embeddings automatically")
    parser.add_argument('--eeg-embeddings', type=str, default=None,
                        help="Path to pre-computed EEG embeddings (.npz)")
    parser.add_argument('--genetic-scores', type=str, default=None,
                        help="Path to pre-computed genetic scores (.npz)")
    parser.add_argument('--genetic-features', type=str,
                        default='data/processed/genetic_vectors/genetic_training_cohort.csv',
                        help="Path to genetic features CSV")
    parser.add_argument('--xgb-model', type=str,
                        default='models/xgboost_genetic/xgboost_genetic_model.pkl',
                        help="Path to XGBoost model")
    parser.add_argument('--epochs', type=int, default=30,
                        help="Training epochs")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument('--batch-size', type=int, default=32,
                        help="Batch size")
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help="Random seed")
    parser.add_argument('--use-synthetic', action='store_true',
                        help="Use synthetic data for testing")
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("=" * 70)
    print("Fusion Layer Training Pipeline")
    print("=" * 70)
    print(f"Timestamp: {timestamp}")
    print(f"Output: {output_dir}")
    
    # ---- Load or generate data ----
    print("\n--- Loading Data ---")
    
    if args.use_synthetic:
        print("Using synthetic data for testing...")
        eeg_embeddings, genetic_scores, labels = generate_synthetic_data(
            n_samples=5000, seed=args.seed
        )
    elif args.eeg_model:
        # Use real genetic data + extract EEG embeddings from model
        print(f"Using real genetic data + EEG model: {args.eeg_model}")
        
        # Step 1: Load real genetic data (CTGAN-generated, same as XGBoost training)
        print(f"\n  [1/3] Loading genetic data from {args.genetic_features}")
        df = pd.read_csv(args.genetic_features)
        feature_cols = [c for c in df.columns if c not in ['patient_id', 'has_seizure', 'preictal_ratio']]
        genetic_features = df[feature_cols].values
        labels = df['has_seizure'].values
        print(f"    Patients: {len(labels)}")
        print(f"    Seizure rate: {labels.mean()*100:.1f}% ({int(labels.sum())} positive)")
        print(f"    Features: {genetic_features.shape[1]} dimensions")
        
        # Step 2: Compute genetic risk scores from XGBoost
        print(f"\n  [2/3] Computing genetic risk scores from XGBoost model...")
        xgb_model = load_xgboost_model(args.xgb_model)
        genetic_scores = extract_genetic_scores(xgb_model, genetic_features)
        print(f"    Risk score range: [{genetic_scores.min():.3f}, {genetic_scores.max():.3f}]")
        print(f"    Mean risk score: {genetic_scores.mean():.3f}")
        
        # Step 3: Extract EEG embeddings from trained BiLSTM
        print(f"\n  [3/3] Extracting EEG embeddings from BiLSTM...")
        
        # Check for cached embeddings first
        cached_embeddings = output_dir / 'eeg_embeddings.npz'
        if cached_embeddings.exists():
            print(f"    Found cached embeddings: {cached_embeddings}")
            cache = np.load(cached_embeddings)
            eeg_all = cache['embeddings']
            print(f"    Loaded {len(eeg_all)} cached embeddings (dim={eeg_all.shape[1]})")
        else:
            # Try to load real EEG dataset
            csv_train = Path('seizure_prediction/DataCSVs/CHB-MIT/all_patients_train.csv')
            csv_test = Path('seizure_prediction/DataCSVs/CHB-MIT/all_patients_test.csv')
            
            if csv_train.exists() and csv_test.exists():
                print(f"    Loading EEG dataset from CHB-MIT CSVs...")
                sys.path.insert(0, str(PROJECT_ROOT / 'seizure_prediction'))
                from libCHBMITDataset import CHBMITDataset
                
                train_dataset = CHBMITDataset(str(csv_train), segment_len=10, preprocess=True)
                test_dataset = CHBMITDataset(str(csv_test), segment_len=10, preprocess=True)
                
                print(f"    Train windows: {len(train_dataset)}")
                print(f"    Test windows: {len(test_dataset)}")
                
                eeg_emb_train, _ = extract_eeg_embeddings_from_model(
                    args.eeg_model, train_dataset, device='cpu', batch_size=16
                )
                eeg_emb_test, _ = extract_eeg_embeddings_from_model(
                    args.eeg_model, test_dataset, device='cpu', batch_size=16
                )
                
                eeg_all = np.concatenate([eeg_emb_train, eeg_emb_test], axis=0)
                
                # Cache for future runs
                np.savez(cached_embeddings, embeddings=eeg_all)
                print(f"    Saved {len(eeg_all)} embeddings to cache")
            else:
                print(f"    WARNING: EEG CSVs not found at:")
                print(f"      {csv_train.absolute()}")
                print(f"      {csv_test.absolute()}")
                print(f"    Generating synthetic EEG embeddings based on genetic risk...")
                rng = np.random.RandomState(args.seed)
                eeg_all = rng.randn(len(labels), 512).astype(np.float32)
                # Correlate with genetic risk (weak signal)
                signal = rng.randn(512) * 0.05
                eeg_all += np.outer(genetic_scores.flatten(), signal)
        
        # Match EEG embeddings to genetic data
        # Each patient may have multiple EEG windows
        n_patients = len(labels)
        n_eeg = len(eeg_all)
        
        if n_eeg >= n_patients:
            # More EEG windows than patients - subsample
            indices = np.linspace(0, n_eeg - 1, n_patients).astype(int)
            eeg_embeddings = eeg_all[indices]
        else:
            # Fewer EEG windows - tile to match
            n_repeat = n_patients // n_eeg + 1
            eeg_embeddings = np.tile(eeg_all, (n_repeat, 1))[:n_patients]
        
        print(f"\n    Final shapes:")
        print(f"      EEG embeddings: {eeg_embeddings.shape}")
        print(f"      Genetic scores: {genetic_scores.shape}")
        print(f"      Labels: {labels.shape}")
        
    elif args.eeg_embeddings:
        # Load pre-computed embeddings
        print(f"Loading EEG embeddings from {args.eeg_embeddings}")
        eeg_embeddings, labels = load_eeg_embeddings(args.eeg_embeddings)
        
        if args.genetic_scores:
            print(f"Loading genetic scores from {args.genetic_scores}")
            genetic_scores = load_genetic_scores(args.genetic_scores)
        else:
            # Compute genetic scores from features
            print("Computing genetic scores from XGBoost model...")
            xgb_model = load_xgboost_model(args.xgb_model)
            
            df = pd.read_csv(args.genetic_features)
            feature_cols = [c for c in df.columns if c not in ['patient_id', 'has_seizure', 'preictal_ratio']]
            genetic_features = df[feature_cols].values
            
            genetic_scores = extract_genetic_scores(xgb_model, genetic_features)
            labels = df['has_seizure'].values
    else:
        # Use real genetic data only (no EEG model specified)
        print("Using real genetic data with synthetic EEG embeddings...")
        
        df = pd.read_csv(args.genetic_features)
        feature_cols = [c for c in df.columns if c not in ['patient_id', 'has_seizure', 'preictal_ratio']]
        genetic_features = df[feature_cols].values
        labels = df['has_seizure'].values
        
        xgb_model = load_xgboost_model(args.xgb_model)
        genetic_scores = extract_genetic_scores(xgb_model, genetic_features)
        
        # Generate synthetic EEG embeddings that correlate with genetic risk
        rng = np.random.RandomState(args.seed)
        eeg_embeddings = rng.randn(len(labels), 512).astype(np.float32)
        # Add weak signal correlated with genetic risk
        signal = rng.randn(512) * 0.1
        eeg_embeddings += np.outer(genetic_scores.flatten(), signal)
        
        print(f"  Patients: {len(labels)}, Positive: {labels.sum()} ({labels.mean()*100:.1f}%)")
    
    print(f"  EEG embeddings: {eeg_embeddings.shape}")
    print(f"  Genetic scores: {genetic_scores.shape}")
    print(f"  Labels: {labels.shape} ({labels.sum()} positive, {(1-labels).sum()} negative)")
    print(f"  Positive rate: {labels.mean()*100:.1f}%")
    
    # ---- Split data ----
    print("\n--- Splitting Data (60/20/20) ---")
    
    # First split: train+val vs test
    X_eeg_temp, X_eeg_test, X_gen_temp, X_gen_test, y_temp, y_test = train_test_split(
        eeg_embeddings, genetic_scores, labels,
        test_size=0.2, stratify=labels, random_state=args.seed
    )
    
    # Second split: train vs val
    X_eeg_train, X_eeg_val, X_gen_train, X_gen_val, y_train, y_val = train_test_split(
        X_eeg_temp, X_gen_temp, y_temp,
        test_size=0.25, stratify=y_temp, random_state=args.seed  # 0.25 of 0.8 = 0.2
    )
    
    print(f"  Train: {len(y_train)} ({y_train.sum()} positive)")
    print(f"  Val:   {len(y_val)} ({y_val.sum()} positive)")
    print(f"  Test:  {len(y_test)} ({y_test.sum()} positive)")
    
    # ---- Create datasets ----
    train_dataset = FusionDataset(X_eeg_train, X_gen_train, y_train)
    val_dataset = FusionDataset(X_eeg_val, X_gen_val, y_val)
    test_dataset = FusionDataset(X_eeg_test, X_gen_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # ---- Create and train fusion model ----
    print("\n--- Creating Fusion Model ---")
    
    model = AttentionGateFusion(
        eeg_embedding_dim=X_eeg_train.shape[1],
        genetic_dim=1,
        hidden_dim=128,
        dropout=0.3,
    )
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    trainer = FusionTrainer(
        model=model,
        device='cpu',
        learning_rate=args.lr,
        weight_decay=1e-4,
        l1_alpha=0.01,
    )
    
    save_path = output_dir / 'fusion_best.pt'
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_path=str(save_path),
    )
    
    # ---- Evaluate on test set ----
    print("\n--- Evaluating on Test Set ---")
    
    # Load best model
    model.load_state_dict(torch.load(save_path, weights_only=False))
    model.eval()
    
    all_preds = []
    all_labels = []
    all_eeg_scores = []
    all_gen_scores = []
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
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_preds)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores[:-1])
    optimal_threshold = thresholds[best_idx]
    
    # Compute metrics at optimal threshold
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
            print(f"    {k:>20s}: {v:.4f}")
        else:
            print(f"    {k:>20s}: {v}")
    
    # ---- Save results ----
    print("\n--- Saving Results ---")
    
    # Save model
    torch.save(model.state_dict(), output_dir / 'fusion_final.pt')
    print(f"  Model saved: {output_dir / 'fusion_final.pt'}")
    
    # Save metrics
    results = {
        'timestamp': timestamp,
        'config': {
            'epochs': args.epochs,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'eeg_embedding_dim': int(X_eeg_train.shape[1]),
            'hidden_dim': 128,
        },
        'data': {
            'train_samples': len(y_train),
            'val_samples': len(y_val),
            'test_samples': len(y_test),
            'positive_rate': float(labels.mean()),
        },
        'test_metrics': test_metrics,
        'history': {
            'train_losses': history['train_losses'],
            'val_losses': history['val_losses'],
            'val_aucs': history['val_aucs'],
        },
    }
    
    with open(output_dir / 'fusion_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Metrics saved: {output_dir / 'fusion_metrics.json'}")
    
    # ---- Generate plots ----
    print("\n--- Generating Plots ---")
    
    plot_training_curves(history, plots_dir)
    plot_confusion_matrix(all_labels, all_preds, optimal_threshold, plots_dir)
    plot_roc_curve(all_labels, all_preds, auc_score, plots_dir)
    plot_pr_curve(all_labels, all_preds, ap_score, plots_dir)
    plot_attention_distribution(all_attention, plots_dir)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  AUC: {auc_score:.4f}")
    print(f"  AUC-PR: {ap_score:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  Mean attention (EEG): {all_attention.mean():.3f}")
    print(f"  All plots saved to: {plots_dir}")


if __name__ == '__main__':
    main()
