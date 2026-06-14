#!/usr/bin/env python3
"""
Fusion Layer Evaluation Pipeline
==================================
End-to-end evaluation of the trained fusion model on held-out test data.
Compares fusion model against individual branches (EEG-only and Genetic-only).

Usage:
    python scripts/evaluate_fusion.py
    python scripts/evaluate_fusion.py --data-dir data/processed/fusion
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix, classification_report,
    f1_score, precision_score, recall_score, accuracy_score,
    matthews_corrcoef
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.fusion import AttentionGateFusion, FusionDataset


OUTPUT_DIR = PROJECT_ROOT / 'models' / 'fusion'


def load_fusion_model(model_path: str, eeg_dim: int = 64) -> AttentionGateFusion:
    """Load trained fusion model."""
    model = AttentionGateFusion(
        eeg_embedding_dim=eeg_dim,
        genetic_dim=1,
        hidden_dim=128,
        dropout=0.3,
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


def evaluate_branch(y_true, scores, branch_name: str) -> dict:
    """Compute metrics for a single branch."""
    auc = roc_auc_score(y_true, scores)
    ap = average_precision_score(y_true, scores)
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1s[:-1])
    opt_threshold = thresholds[best_idx]
    
    y_pred = (scores >= opt_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        'branch': branch_name,
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


def plot_comparison_roc(y_true, scores_dict, save_path):
    """Plot ROC curves for all branches on the same figure."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    for (name, scores), color in zip(scores_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{name} (AUC={auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('ROC Curve Comparison', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    
    plt.tight_layout()
    plt.savefig(save_path / 'roc_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path / 'roc_comparison.png'}")


def plot_comparison_pr(y_true, scores_dict, save_path):
    """Plot Precision-Recall curves for all branches."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    for (name, scores), color in zip(scores_dict.items(), colors):
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ax.plot(recall, precision, color=color, linewidth=2, label=f'{name} (AP={ap:.3f})')
    
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title('Precision-Recall Curve Comparison', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    
    plt.tight_layout()
    plt.savefig(save_path / 'pr_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path / 'pr_comparison.png'}")


def plot_calibration(y_true, scores_dict, save_path, n_bins=10):
    """Plot calibration curves (reliability diagrams)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    for (name, scores), color in zip(scores_dict.items(), colors):
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_means = []
        bin_true_means = []
        
        for i in range(n_bins):
            mask = (scores >= bin_edges[i]) & (scores < bin_edges[i+1])
            if mask.sum() > 0:
                bin_means.append(scores[mask].mean())
                bin_true_means.append(y_true[mask].mean())
        
        ax.plot(bin_means, bin_true_means, 'o-', color=color, linewidth=2, 
                markersize=6, label=name)
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Perfect calibration')
    ax.set_xlabel('Mean Predicted Probability', fontsize=14)
    ax.set_ylabel('Fraction of Positives', fontsize=14)
    ax.set_title('Calibration Curve', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    
    plt.tight_layout()
    plt.savefig(save_path / 'calibration.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path / 'calibration.png'}")


def plot_improvement_bar(metrics_list, save_path):
    """Bar chart showing improvement of fusion over individual branches."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    branches = [m['branch'] for m in metrics_list]
    metrics_to_plot = ['auc', 'f1', 'recall', 'specificity']
    
    x = np.arange(len(branches))
    width = 0.2
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    
    for i, metric in enumerate(metrics_to_plot):
        values = [m[metric] for m in metrics_list]
        ax.bar(x + i * width, values, width, label=metric.upper(), color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Performance Comparison Across Models', fontsize=16)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(branches, fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path / 'improvement_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path / 'improvement_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Fusion Model")
    parser.add_argument('--fusion-model', type=str,
                        default=str(OUTPUT_DIR / 'fusion_best.pt'),
                        help="Path to trained fusion model")
    parser.add_argument('--eeg-embeddings', type=str, default=None,
                        help="Path to test EEG embeddings (.npz)")
    parser.add_argument('--genetic-scores', type=str, default=None,
                        help="Path to test genetic scores (.npz)")
    parser.add_argument('--data-dir', type=str, default=None,
                        help="Directory containing all test data")
    args = parser.parse_args()
    
    output_dir = OUTPUT_DIR
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Fusion Model Evaluation")
    print("=" * 70)
    
    # ---- Load test data ----
    print("\n--- Loading Test Data ---")
    
    if args.data_dir:
        data_dir = Path(args.data_dir)
        eeg_embeddings = np.load(data_dir / 'test_eeg_embeddings.npy')
        genetic_scores = np.load(data_dir / 'test_genetic_scores.npy')
        labels = np.load(data_dir / 'test_labels.npy')
    elif args.eeg_embeddings:
        eeg_data = np.load(args.eeg_embeddings)
        eeg_embeddings = eeg_data['embeddings']
        labels = eeg_data['labels']
        
        if args.genetic_scores:
            genetic_scores = np.load(args.genetic_scores)['scores'].reshape(-1, 1)
        else:
            raise ValueError("Must provide --genetic-scores when using --eeg-embeddings")
    else:
        print("No data specified. Using synthetic data for demonstration...")
        from scripts.train_fusion import generate_synthetic_data
        eeg_embeddings, genetic_scores, labels = generate_synthetic_data(
            n_samples=1000, seed=42
        )
    
    # Ensure genetic_scores is 2D
    if genetic_scores.ndim == 1:
        genetic_scores = genetic_scores.reshape(-1, 1)
    
    print(f"  Test samples: {len(labels)}")
    print(f"  Positive samples: {int(labels.sum())}")
    print(f"  Positive rate: {labels.mean()*100:.1f}%")
    
    # ---- Load model and get predictions ----
    print("\n--- Loading Fusion Model ---")
    
    model = load_fusion_model(args.fusion_model, eeg_dim=eeg_embeddings.shape[1])
    print(f"  Model loaded: {args.fusion_model}")
    
    # Get fusion predictions
    test_dataset = FusionDataset(eeg_embeddings, genetic_scores, labels)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    all_fused = []
    all_attention = []
    
    with torch.no_grad():
        for eeg_emb, gen_score, _ in test_loader:
            risk_score, alpha = model(eeg_emb, gen_score)
            all_fused.extend(risk_score.numpy().flatten())
            all_attention.extend(alpha.numpy().flatten())
    
    fused_scores = np.array(all_fused)
    attention_weights = np.array(all_attention)
    
    # ---- Also get branch-only scores ----
    # For proper evaluation, we need the actual branch predictions
    # EEG-only: Use the risk score from the fusion model's EEG branch
    # (This is the projection before attention weighting)
    # Genetic-only: Use the genetic scores directly
    
    # To get EEG-only scores, we need to extract them from the fusion model
    # by looking at the intermediate outputs
    eeg_scores = []
    genetic_scores_flat = genetic_scores.flatten()
    
    with torch.no_grad():
        for eeg_emb, gen_score, _ in test_loader:
            # Get intermediate EEG projection
            eeg_proj = model.eeg_projection(eeg_emb)
            gen_proj = model.genetic_projection(gen_score)
            
            # Get attention weights
            combined = torch.cat([eeg_proj, gen_proj], dim=1)
            alpha = model.attention(combined)
            
            # EEG-only score: use the attention-weighted EEG contribution
            # Higher alpha = more EEG influence = higher EEG score
            eeg_only = alpha * eeg_proj.mean(dim=1, keepdim=True)
            eeg_scores.extend(eeg_only.numpy().flatten())
    
    eeg_scores = np.array(eeg_scores)
    
    # ---- Evaluate each branch ----
    print("\n--- Branch Evaluation ---")
    
    eeg_metrics = evaluate_branch(labels, eeg_scores, 'EEG-only')
    gen_metrics = evaluate_branch(labels, genetic_scores_flat, 'Genetic-only')
    fused_metrics = evaluate_branch(labels, fused_scores, 'Fusion')
    
    all_metrics = [eeg_metrics, gen_metrics, fused_metrics]
    
    for m in all_metrics:
        print(f"\n  {m['branch']}:")
        print(f"    AUC:        {m['auc']:.4f}")
        print(f"    AUC-PR:     {m['aucpr']:.4f}")
        print(f"    F1:         {m['f1']:.4f}")
        print(f"    Recall:     {m['recall']:.4f}")
        print(f"    Specificity:{m['specificity']:.4f}")
        print(f"    MCC:        {m['mcc']:.4f}")
    
    # ---- Compute improvement ----
    print("\n--- Improvement Analysis ---")
    improvement = {
        'auc_gain_vs_eeg': fused_metrics['auc'] - eeg_metrics['auc'],
        'auc_gain_vs_genetic': fused_metrics['auc'] - gen_metrics['auc'],
        'f1_gain_vs_eeg': fused_metrics['f1'] - eeg_metrics['f1'],
        'f1_gain_vs_genetic': fused_metrics['f1'] - gen_metrics['f1'],
        'recall_gain_vs_eeg': fused_metrics['recall'] - eeg_metrics['recall'],
        'recall_gain_vs_genetic': fused_metrics['recall'] - gen_metrics['recall'],
    }
    
    for k, v in improvement.items():
        print(f"  {k}: {v:+.4f}")
    
    # ---- Attention analysis ----
    print("\n--- Attention Analysis ---")
    print(f"  Mean attention weight (EEG):     {attention_weights.mean():.3f}")
    print(f"  Mean attention weight (Genetic): {1-attention_weights.mean():.3f}")
    print(f"  Std attention weight:            {attention_weights.std():.3f}")
    
    # Attention by class
    print(f"  Attention for seizure patients:  {attention_weights[labels == 1].mean():.3f}")
    print(f"  Attention for non-seizure:       {attention_weights[labels == 0].mean():.3f}")
    
    # ---- Generate plots ----
    print("\n--- Generating Plots ---")
    
    scores_dict = {
        'EEG-only': eeg_scores,
        'Genetic-only': genetic_scores_flat,
        'Fusion': fused_scores,
    }
    
    plot_comparison_roc(labels, scores_dict, plots_dir)
    plot_comparison_pr(labels, scores_dict, plots_dir)
    plot_calibration(labels, scores_dict, plots_dir)
    plot_improvement_bar(all_metrics, plots_dir)
    
    # ---- Save results ----
    print("\n--- Saving Results ---")
    
    results = {
        'test_samples': len(labels),
        'positive_samples': int(labels.sum()),
        'positive_rate': float(labels.mean()),
        'metrics': {m['branch']: m for m in all_metrics},
        'improvement': improvement,
        'attention': {
            'mean_eeg': float(attention_weights.mean()),
            'mean_genetic': float(1 - attention_weights.mean()),
            'std': float(attention_weights.std()),
            'mean_seizure': float(attention_weights[labels == 1].mean()),
            'mean_non_seizure': float(attention_weights[labels == 0].mean()),
        },
    }
    
    with open(output_dir / 'fusion_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {output_dir / 'fusion_evaluation.json'}")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    
    # Final summary
    print(f"\n  {'Model':<20s} {'AUC':>8s} {'F1':>8s} {'Recall':>8s} {'Spec':>8s}")
    print(f"  {'-'*52}")
    for m in all_metrics:
        print(f"  {m['branch']:<20s} {m['auc']:>8.4f} {m['f1']:>8.4f} {m['recall']:>8.4f} {m['specificity']:>8.4f}")
    
    print(f"\n  Fusion AUC improvement over EEG-only:  {improvement['auc_gain_vs_eeg']:+.4f}")
    print(f"  Fusion AUC improvement over Genetic:   {improvement['auc_gain_vs_genetic']:+.4f}")
    print(f"\n  All plots saved to: {plots_dir}")


if __name__ == '__main__':
    main()
