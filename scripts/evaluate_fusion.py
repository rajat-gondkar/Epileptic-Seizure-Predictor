#!/usr/bin/env python3
"""
Fusion Evaluation: EEG-only vs EEG+Genetic Fusion
===================================================
Compares the trained BiLSTM (EEG-only) against the fusion model
to answer: does adding genetic data help?

EEG-only score: BiLSTM FC layer → softmax → preictal probability
Fusion score:   Fusion model → attention-gated risk score
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

EEG_MODEL_PATH = 'seizure_prediction/SavedModels/EEGLSTM_CHB-MIT_all_patients_train_Epoch-15_TLoss-0.0501_VLoss-0.7880_20260605-152850.net'


def load_data():
    """Load EEG embeddings, compute genetic scores, produce train/test split."""
    cache = np.load(CACHE_FILE)
    eeg_embeddings = cache['embeddings']
    eeg_labels_raw = cache['labels']
    eeg_labels = (eeg_labels_raw > 0).astype(np.float32)

    genetic_csv = 'data/processed/genetic_vectors/genetic_training_cohort.csv'
    xgb_path = 'models/xgboost_genetic/xgboost_genetic_model.pkl'

    df = pd.read_csv(genetic_csv)
    available = [f for f in FEATURE_NAMES if f in df.columns]
    genetic_features = df[available].values.astype(np.float32)

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(xgb_path)
    genetic_risk_scores = xgb_model.predict_proba(genetic_features)[:, 1]

    rng = np.random.RandomState(SEED)
    genetic_scores_aligned = rng.choice(
        genetic_risk_scores, size=len(eeg_labels_raw), replace=True
    ).reshape(-1, 1)

    X_eeg_temp, X_eeg_test, X_gen_temp, X_gen_test, y_temp, y_test = train_test_split(
        eeg_embeddings, genetic_scores_aligned, eeg_labels,
        test_size=0.2, stratify=eeg_labels, random_state=SEED
    )
    X_eeg_train, X_eeg_val, X_gen_train, X_gen_val, y_train, y_val = train_test_split(
        X_eeg_temp, X_gen_temp, y_temp,
        test_size=0.25, stratify=y_temp, random_state=SEED
    )

    return X_eeg_test, X_gen_test, y_test


def compute_eeg_only_scores(X_eeg_test):
    """Get preictal probabilities from the BiLSTM's own FC layer."""
    ckpt = torch.load(EEG_MODEL_PATH, map_location='cpu', weights_only=False)
    fc_w = ckpt['dctStateDict']['FCLayer.weight']  # (3, 512)
    fc_b = ckpt['dctStateDict']['FCLayer.bias']    # (3,)
    with torch.no_grad():
        logits = torch.FloatTensor(X_eeg_test) @ fc_w.T + fc_b
        probs = F.softmax(logits, dim=1)
    return probs[:, 1].numpy()


def compute_fusion_scores(X_eeg_test, X_gen_test):
    """Get fused risk scores from the trained fusion model."""
    model = AttentionGateFusion(eeg_embedding_dim=512, genetic_dim=1, hidden_dim=128, dropout=0.3)
    model.load_state_dict(torch.load(OUTPUT_DIR / 'fusion_best.pt', map_location='cpu', weights_only=False))
    model.eval()
    dataset = FusionDataset(X_eeg_test, X_gen_test, np.zeros(len(X_eeg_test)))
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    scores, alphas = [], []
    with torch.no_grad():
        for eeg_emb, gen_score, _ in loader:
            risk, alpha = model(eeg_emb, gen_score)
            scores.extend(risk.numpy().flatten())
            alphas.extend(alpha.numpy().flatten())
    return np.array(scores), np.array(alphas)


def compute_metrics(y_true, y_prob, name):
    """Compute all metrics for a model."""
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1s[:-1])
    opt_thr = thresholds[best_idx]
    y_pred = (y_prob >= opt_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'name': name, 'auc': auc, 'aucpr': ap, 'threshold': opt_thr,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'mcc': matthews_corrcoef(y_true, y_pred),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Fusion Evaluation: EEG-only vs EEG+Genetic Fusion")
    print("=" * 70)

    X_eeg_test, X_gen_test, y_test = load_data()
    print(f"  Test: {len(y_test)} windows, {int(y_test.sum())} seizure ({y_test.mean()*100:.1f}%)")

    eeg_scores = compute_eeg_only_scores(X_eeg_test)
    fusion_scores, attention_weights = compute_fusion_scores(X_eeg_test, X_gen_test)

    eeg_m = compute_metrics(y_test, eeg_scores, 'EEG-only')
    fuse_m = compute_metrics(y_test, fusion_scores, 'Fusion')

    print(f"\n  {'':20s} {'AUC':>8s} {'AUC-PR':>8s} {'F1':>8s} {'Precision':>10s} {'Recall':>8s} {'MCC':>8s}")
    print(f"  {'-'*74}")
    for m in [eeg_m, fuse_m]:
        print(f"  {m['name']:20s} {m['auc']:>8.4f} {m['aucpr']:>8.4f} {m['f1']:>8.4f} "
              f"{m['precision']:>10.4f} {m['recall']:>8.4f} {m['mcc']:>8.4f}")

    print(f"\n  Fusion improvement over EEG-only:")
    print(f"    AUC:      {fuse_m['auc'] - eeg_m['auc']:+.4f}")
    print(f"    AUC-PR:   {fuse_m['aucpr'] - eeg_m['aucpr']:+.4f}")
    print(f"    F1:       {fuse_m['f1'] - eeg_m['f1']:+.4f}")
    print(f"    Recall:   {fuse_m['recall'] - eeg_m['recall']:+.4f}")
    print(f"    MCC:      {fuse_m['mcc'] - eeg_m['mcc']:+.4f}")

    print(f"\n  Attention: mean={attention_weights.mean():.3f}, std={attention_weights.std():.4f}")

    # ==================== PLOTS ====================
    print("\n--- Generating Plots ---")
    blue, orange = '#2196F3', '#FF9800'

    # 1. ROC
    fig, ax = plt.subplots(figsize=(8, 6))
    for m, color, ls in [(eeg_m, blue, '-'), (fuse_m, orange, '-')]:
        fpr, tpr, _ = roc_curve(y_test, [eeg_scores, fusion_scores][0 if m['name'] == 'EEG-only' else 1])
        ax.plot(fpr, tpr, color=color, linewidth=2.5, linestyle=ls,
                label=f'{m["name"]} (AUC={m["auc"]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.4)
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('ROC Curve: EEG-only vs Fusion', fontsize=16)
    ax.legend(fontsize=13, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'roc_eeg_vs_fusion.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: roc_eeg_vs_fusion.png")

    # 2. PR
    fig, ax = plt.subplots(figsize=(8, 6))
    for sc, name, color in [(eeg_scores, 'EEG-only', blue), (fusion_scores, 'Fusion', orange)]:
        prec, rec, _ = precision_recall_curve(y_test, sc)
        ap = average_precision_score(y_test, sc)
        ax.plot(rec, prec, color=color, linewidth=2.5, label=f'{name} (AP={ap:.3f})')
    baseline = y_test.mean()
    ax.axhline(baseline, color='gray', linestyle='--', linewidth=1, alpha=0.5, label=f'Baseline ({baseline:.3f})')
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title('Precision-Recall: EEG-only vs Fusion', fontsize=16)
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'pr_eeg_vs_fusion.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: pr_eeg_vs_fusion.png")

    # 3. Side-by-side metric bars
    fig, ax = plt.subplots(figsize=(10, 6))
    metric_names = ['AUC', 'AUC-PR', 'F1', 'Precision', 'Recall', 'Specificity', 'MCC']
    eeg_vals = [eeg_m['auc'], eeg_m['aucpr'], eeg_m['f1'], eeg_m['precision'],
                eeg_m['recall'], eeg_m['specificity'], eeg_m['mcc']]
    fuse_vals = [fuse_m['auc'], fuse_m['aucpr'], fuse_m['f1'], fuse_m['precision'],
                 fuse_m['recall'], fuse_m['specificity'], fuse_m['mcc']]
    x = np.arange(len(metric_names))
    width = 0.35
    ax.bar(x - width/2, eeg_vals, width, label='EEG-only', color=blue, alpha=0.85)
    ax.bar(x + width/2, fuse_vals, width, label='Fusion', color=orange, alpha=0.85)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('EEG-only vs Fusion: All Metrics', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=12)
    ax.legend(fontsize=13)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    for i, (ev, fv) in enumerate(zip(eeg_vals, fuse_vals)):
        diff = fv - ev
        color = 'green' if diff > 0 else 'red'
        ax.annotate(f'{diff:+.3f}', xy=(i + width/2, fv), fontsize=8, color=color,
                    ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: metrics_comparison.png")

    # 4. Confusion matrices side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, scores, name, thr in [(axes[0], eeg_scores, 'EEG-only', eeg_m['threshold']),
                                   (axes[1], fusion_scores, 'Fusion', fuse_m['threshold'])]:
        y_pred = (scores >= thr).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Non-seizure', 'Seizure'],
                    yticklabels=['Non-seizure', 'Seizure'])
        ax.set_xlabel('Predicted', fontsize=13)
        ax.set_ylabel('Actual', fontsize=13)
        ax.set_title(f'{name}\n(thr={thr:.3f})', fontsize=14)
    plt.suptitle('Confusion Matrices', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: confusion_matrices.png")

    # 5. Prediction distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, scores, name, color in [(axes[0], eeg_scores, 'EEG-only', blue),
                                     (axes[1], fusion_scores, 'Fusion', orange)]:
        ax.hist(scores[y_test == 0], bins=50, alpha=0.6, color='steelblue', label='Non-seizure', density=True)
        ax.hist(scores[y_test == 1], bins=50, alpha=0.6, color='crimson', label='Seizure', density=True)
        ax.set_xlabel('Predicted Score', fontsize=13)
        ax.set_ylabel('Density', fontsize=13)
        ax.set_title(f'{name} Score Distribution', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'score_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: score_distributions.png")

    # ---- Save JSON ----
    results = {
        'test_samples': len(y_test),
        'positive_samples': int(y_test.sum()),
        'eeg_only': eeg_m, 'fusion': fuse_m,
        'improvement': {
            'auc': fuse_m['auc'] - eeg_m['auc'],
            'aucpr': fuse_m['aucpr'] - eeg_m['aucpr'],
            'f1': fuse_m['f1'] - eeg_m['f1'],
            'recall': fuse_m['recall'] - eeg_m['recall'],
            'mcc': fuse_m['mcc'] - eeg_m['mcc'],
        },
        'attention': {'mean': float(attention_weights.mean()), 'std': float(attention_weights.std())},
    }
    with open(OUTPUT_DIR / 'fusion_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
