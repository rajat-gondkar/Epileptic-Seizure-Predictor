#!/usr/bin/env python3
"""
Step 2: Train LSTM + XGBoost Models
======================================
Trains both branches of the EEG-Genetic Fusion pipeline:
  1. STFT-CNN-BiLSTM with self-attention on STFT spectrograms
  2. XGBoost on 221-dim EEG features

Designed for CUDA GPU (RTX 4050 / T4 / A100 etc).

Usage:
    python 02_train_models.py                    # full training
    python 02_train_models.py --quick-test       # 3 epochs, tiny data
    python 02_train_models.py --lstm-only         # skip XGBoost
    python 02_train_models.py --xgb-only          # skip LSTM
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from scipy import signal
from sklearn.metrics import (
    roc_auc_score, classification_report, roc_curve,
    precision_recall_curve, confusion_matrix, average_precision_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Project root ──
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lstm_eeg import EEGCNNLSTM
from src.models.xgboost_genetic import train_xgboost


def load_config(path=None):
    """Load config.yaml."""
    candidates = [
        path,
        PROJECT_ROOT / "configs" / "config.yaml",
        SCRIPT_DIR / "config.yaml",
    ]
    for cp in candidates:
        if cp and Path(cp).exists():
            with open(cp) as f:
                config = yaml.safe_load(f)
            print(f"Config: {cp}")
            return config
    raise FileNotFoundError("No config.yaml found")


# ============================================================
# Data Loading
# ============================================================
def load_patient_data(feat_dir, patient_id):
    """Load sequences, features, and labels for one patient."""
    paths = {
        k: feat_dir / f"{patient_id}_{k}.npy"
        for k in ["sequences", "features", "labels"]
    }
    if not all(p.exists() for p in paths.values()):
        return None, None, None

    # Use mmap_mode='r' for sequences to avoid loading all into RAM at once
    sequences = np.load(paths["sequences"], mmap_mode="r")
    features  = np.load(paths["features"])
    labels    = np.load(paths["labels"])
    return sequences, features, labels


def load_all_data(config, max_epochs_per_patient=None):
    """
    Load data for all patients with patient-level train/val/test split.

    Split strategy:
        - Test:  last 2 patients (unseen patients)
        - Val:   1 patient
        - Train: remaining patients
    """
    feat_dir = PROJECT_ROOT / config["paths"]["data"]["processed"]["eeg_features"]
    patients = config["dataset"]["chb_mit_patients"]

    print(f"\nData directory: {feat_dir}")
    print(f"Configured patients: {patients}")

    patient_data = {}
    for pat in patients:
        seq, feat, lab = load_patient_data(feat_dir, pat)
        if seq is None:
            print(f"  {pat}: MISSING — skipped")
            continue

        if max_epochs_per_patient and len(lab) > max_epochs_per_patient:
            # Stratified subsample
            rng = np.random.RandomState(42)
            idx_0 = np.where(lab == 0)[0]
            idx_1 = np.where(lab == 1)[0]
            ratio = len(idx_1) / len(lab)
            n_1 = max(int(max_epochs_per_patient * ratio), min(100, len(idx_1)))
            n_0 = max_epochs_per_patient - n_1
            keep_0 = rng.choice(idx_0, min(n_0, len(idx_0)), replace=False)
            keep_1 = rng.choice(idx_1, min(n_1, len(idx_1)), replace=False)
            keep = np.sort(np.concatenate([keep_0, keep_1]))
            seq  = np.array(seq[keep])
            feat = feat[keep]
            lab  = lab[keep]

        patient_data[pat] = {"sequences": seq, "features": feat, "labels": lab}
        n_pre = int((lab == 1).sum())
        print(f"  {pat}: {len(lab):>7,} epochs  ({n_pre:,} preictal, "
              f"{len(lab)-n_pre:,} interictal)")

    if len(patient_data) < 3:
        raise ValueError(f"Need ≥3 patients with data, found {len(patient_data)}")

    # Patient-level split
    all_pats = sorted(patient_data.keys())
    test_pats  = all_pats[-2:]
    val_pats   = [all_pats[-3]]
    train_pats = all_pats[:-3]

    def merge(pats):
        seqs  = [patient_data[p]["sequences"] for p in pats]  # keep as list (memmap-friendly)
        feats = np.concatenate([patient_data[p]["features"] for p in pats])
        labs  = np.concatenate([patient_data[p]["labels"] for p in pats])
        return seqs, feats, labs

    splits = {}
    for name, pats in [("train", train_pats), ("val", val_pats), ("test", test_pats)]:
        s, f, l = merge(pats)
        splits[name] = (s, f, l)
        n_pre = int((l == 1).sum())
        print(f"\n  {name:5s}: {len(l):>8,} epochs  "
              f"(preictal={n_pre:,}, interictal={len(l)-n_pre:,})  "
              f"patients={pats}")

    splits["train_pats"] = train_pats
    splits["val_pats"] = val_pats
    splits["test_pats"] = test_pats
    return splits


# ============================================================
# Dataset that indexes across a list of arrays (memmaps)
# without copying everything into RAM at once.
# ============================================================
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, seqs_list, labels):
        self.seqs = seqs_list
        self.labels = np.asarray(labels)
        self.cumlen = np.cumsum([len(s) for s in seqs_list])
        if self.cumlen[-1] != len(self.labels):
            raise ValueError("Total sequences must match number of labels")

    def __len__(self):
        return int(self.cumlen[-1])

    def __getitem__(self, idx):
        arr_idx = int(np.searchsorted(self.cumlen, idx, side="right"))
        if arr_idx > 0:
            local_idx = idx - self.cumlen[arr_idx - 1]
        else:
            local_idx = idx
        x_raw = np.array(self.seqs[arr_idx][local_idx])  # [T=1280, C=17]

        # ── STFT spectrogram: [C, F, T_stft] ──
        x_np = x_raw.T                                    # [C, T]
        _, _, Zxx = signal.stft(
            x_np, fs=256, nperseg=256, noverlap=192,
            boundary="zeros", padded=True,
        )
        spec = np.abs(Zxx[:, 1:71, :])                    # keep 1–70 Hz
        spec = np.log(spec + 1e-8)                        # log magnitude
        x = torch.from_numpy(spec).float()                # [C, F, T]

        y = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)
        return x, y


# ============================================================
# Focal Loss
# ============================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    Down-weights easy examples (majority class) so the model focuses on hard
    positives and hard negatives.

    FL(pt) = -α_t * (1 - pt)^γ * log(pt)

    Args:
        gamma: focusing parameter (γ=2 is standard)
        alpha: positive-class weight (0.75 means 3:1 pos:neg weighting)
    """
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_t * (1.0 - pt) ** self.gamma * bce
        return loss.mean()


# ============================================================
# LSTM Training
# ============================================================
def _init_final_bias(model, pos_ratio):
    """Initialize fc2.bias so initial prediction ≈ pos_ratio (breaks 0.5 symmetry)."""
    with torch.no_grad():
        bias_val = np.log(pos_ratio / max(1 - pos_ratio, 1e-6))
        model.fc2.bias.fill_(bias_val)


def _log_grad_norms(model, prefix="  [GRAD]"):
    """Print L2 norm of gradients per layer (diagnostic)."""
    lines = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            gnorm = p.grad.norm().item()
            lines.append(f"{name}: {gnorm:.6f}")
    for line in (lines[:3] + ["  ..."] + lines[-3:] if len(lines) > 6 else lines):
        print(prefix, line)


def train_lstm(data, config, device, output_dir):
    """Full LSTM training with early stopping, focal loss and LR warmup."""
    cfg = config["lstm"]
    train_seq, _, train_lab = data["train"]
    val_seq, _, val_lab = data["val"]

    print("\n" + "=" * 60)
    print("LSTM TRAINING")
    print("=" * 60)

    # ── Datasets (keep sequences on disk via memmap) ──
    train_ds = SequenceDataset(train_seq, train_lab)
    val_ds   = SequenceDataset(val_seq, val_lab)

    total_seqs = sum(len(s) for s in train_seq)
    mem_gb = total_seqs * train_seq[0].shape[1] * train_seq[0].shape[2] * 4 / 1e9
    print(f"Train: {total_seqs:,} epochs, ~{mem_gb:.1f} GB if loaded fully")
    print(f"Val:   {len(val_lab):,} epochs")
    print(f"Device: {device}")

    # ── DataLoaders (natural distribution, NO sampler) ──
    use_cuda = device.type == "cuda"
    print(f"CUDA I/O: pin_memory={use_cuda}, non_blocking=True")
    print("  NOTE: CPU at 100% is expected — STFT is computed on CPU with num_workers=0")
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=0, pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
        num_workers=0, pin_memory=use_cuda,
    )

    # ── Model ──
    model = EEGCNNLSTM(
        input_channels=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        embedding_dim=64,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ── Loss: Focal Loss (replaces BCEWithLogitsLoss + pos_weight) ──
    n_neg = int((train_lab == 0).sum())
    n_pos = int((train_lab == 1).sum())
    ratio = n_neg / max(n_pos, 1)
    pos_ratio = n_pos / len(train_lab)
    print(f"Class ratio {ratio:.1f}:1  pos_ratio={pos_ratio:.4f}")
    focal_alpha = cfg.get("focal_alpha", 0.90)
    focal_gamma = cfg.get("focal_gamma", 2.0)
    print(f"Loss: FocalLoss(gamma={focal_gamma}, alpha={focal_alpha})  (NO sampler — natural batches)")
    criterion = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)

    # Break symmetry: init final bias to predict baseline positive rate
    _init_final_bias(model, pos_ratio)
    with torch.no_grad():
        init_pred = torch.sigmoid(model.fc2.bias).item()
    print(f"Initial prediction bias: {init_pred:.4f}")

    # ── Optimiser ──
    base_lr = cfg["learning_rate"]
    optimizer = optim.Adam(
        model.parameters(), lr=base_lr,
        weight_decay=cfg["weight_decay"],
    )
    # ReduceLROnPlateau is used only AFTER warmup
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max",
        patience=cfg["scheduler"]["patience"],
        factor=cfg["scheduler"]["factor"],
    )

    # ── Training loop ──
    best_auc = 0.0
    patience_counter = 0
    max_epochs = cfg["max_epochs"]
    early_stop = cfg["early_stop_patience"]
    history = []
    epoch_start_global = time.time()
    warmup_epochs = 5

    threshold = cfg.get("classification_threshold", 0.20)
    print(f"\n  Classification threshold: {threshold}  (NOT 0.5 — tuned for imbalance)")
    # Header for status monitor
    print(f"\n  {'Epoch':>5} │ {'Train Loss':>10} │ {'Val Loss':>10} │ {'Val AUC':>8} │ "
          f"{'Sens':>6} │ {'Spec':>6} │ {'LR':>10} │ {'Time':>6} │ {'ETA':>8} │ Status")
    print("  " + "─" * 95)

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        # ── Linear LR Warmup (epochs 1–5) ──
        if epoch <= warmup_epochs:
            warmup_factor = epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = base_lr * warmup_factor

        # — Train —
        model.train()
        train_loss = 0.0
        n_batches = 0

        # Augmentation config
        aug_cfg = cfg.get("augmentation", {})
        noise_sigma = aug_cfg.get("gaussian_noise_sigma", 0.0)
        chan_dropout = aug_cfg.get("channel_dropout_prob", 0.0)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}", leave=False)
        for batch_idx, (X_b, y_b) in enumerate(pbar):
            X_b, y_b = X_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)

            # ── Augmentation ──
            if noise_sigma > 0:
                X_b = X_b + torch.randn_like(X_b) * noise_sigma
            if chan_dropout > 0:
                mask = torch.rand(X_b.shape[0], X_b.shape[1], device=device) > chan_dropout
                X_b = X_b * mask.unsqueeze(-1).unsqueeze(-1).float()

            optimizer.zero_grad()
            logits = model(X_b)

            # ── Debug: first batch of first epoch ──
            if epoch == 1 and batch_idx == 0:
                with torch.no_grad():
                    probs = torch.sigmoid(logits).cpu().numpy()
                    print(f"\n  [DEBUG] Batch-0 preds — mean={probs.mean():.4f} std={probs.std():.6f} "
                          f"min={probs.min():.4f} max={probs.max():.4f}")
                    if probs.std() < 1e-6:
                        print("  [WARNING] All predictions identical — model stuck in symmetric min.")

            loss = criterion(logits, y_b)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if epoch == 1 and batch_idx == 0:
                print(f"  [DEBUG] Loss={loss.item():.4f}  Grad norm (clipped): {grad_norm:.4f}")
                if grad_norm < 1e-6:
                    print("  [WARNING] Gradient norm near zero — model not learning!")
                _log_grad_norms(model)
                with torch.no_grad():
                    probs = torch.sigmoid(logits).cpu().numpy()
                    print(f"  [DEBUG] After backward — pred mean={probs.mean():.4f} std={probs.std():.6f}")

            optimizer.step()

            train_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / max(n_batches, 1)

        # — Validate —
        model.eval()
        all_preds, all_labels_list = [], []
        val_loss = 0.0
        n_vb = 0

        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)
                logits = model(X_b)
                val_loss += criterion(logits, y_b).item()
                n_vb += 1
                all_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                all_labels_list.extend(y_b.cpu().numpy().flatten())

        avg_val_loss = val_loss / max(n_vb, 1)
        preds_arr = np.array(all_preds)
        labels_arr = np.array(all_labels_list)

        try:
            val_auc = roc_auc_score(labels_arr, preds_arr)
        except ValueError:
            val_auc = 0.5

        # ── Scheduler step (only after warmup) ──
        if epoch > warmup_epochs:
            plateau_scheduler.step(val_auc)
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        # Calculate sensitivity/specificity at configured threshold
        threshold = cfg.get("classification_threshold", 0.20)
        pred_bin = (preds_arr >= threshold).astype(int)
        sens = (pred_bin[labels_arr == 1] == 1).mean() if (labels_arr == 1).any() else 0
        spec = (pred_bin[labels_arr == 0] == 0).mean() if (labels_arr == 0).any() else 0

        history.append({
            "epoch": epoch, "train_loss": avg_train_loss,
            "val_loss": avg_val_loss, "val_auc": val_auc,
            "val_sens": float(sens), "val_spec": float(spec),
            "lr": lr_now, "time_sec": elapsed,
        })

        # ETA estimate
        elapsed_total = time.time() - epoch_start_global
        avg_epoch_time = elapsed_total / epoch
        remaining_epochs = max_epochs - epoch
        eta_sec = avg_epoch_time * remaining_epochs
        eta_str = f"{int(eta_sec // 60)}m{int(eta_sec % 60):02d}s"

        # Status indicator
        if val_auc > best_auc:
            status = "★ BEST"
        elif patience_counter > 0:
            status = f"↓ {patience_counter}/{early_stop}"
        else:
            status = "  —"

        print(f"  {epoch:5d} │ {avg_train_loss:10.4f} │ {avg_val_loss:10.4f} │ {val_auc:8.4f} │ "
              f"{sens:6.3f} │ {spec:6.3f} │ {lr_now:10.1e} │ {elapsed:5.0f}s │ {eta_str:>8} │ {status}")

        # ── Save latest checkpoint ──
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_auc": val_auc,
            "config": cfg,
        }, output_dir / "lstm_latest.pt")

        # Early stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": best_auc,
                "config": cfg,
            }, output_dir / "lstm_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= early_stop:
                print(f"\n  Early stopping triggered at epoch {epoch} (patience={early_stop})")
                break

    # Save history
    with open(output_dir / "lstm_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Best validation AUC: {best_auc:.4f}")
    print(f"  Best model:  {output_dir / 'lstm_best.pt'}")
    print(f"  Latest model: {output_dir / 'lstm_latest.pt'}")
    print(f"  History:     {output_dir / 'lstm_history.json'}")

    return model, best_auc, history


# ============================================================
# XGBoost Training
# ============================================================
def train_xgb(data, config, output_dir):
    """Train XGBoost on 221-dim EEG feature vectors."""
    import joblib

    print("\n" + "=" * 60)
    print("XGBOOST TRAINING")
    print("=" * 60)

    _, train_feat, train_lab = data["train"]
    _, val_feat, val_lab = data["val"]

    print(f"Train: {train_feat.shape}  ({int((train_lab==1).sum()):,} preictal)")
    print(f"Val:   {val_feat.shape}  ({int((val_lab==1).sum()):,} preictal)")

    model, metrics = train_xgboost(train_feat, train_lab, val_feat, val_lab, config)

    print(f"\nXGBoost Validation Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    joblib.dump(model, output_dir / "xgboost_model.pkl")
    with open(output_dir / "xgboost_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Model saved: {output_dir / 'xgboost_model.pkl'}")
    return model, metrics


# ============================================================
# Test Evaluation
# ============================================================
def evaluate_test(lstm_model, xgb_model, data, device, output_dir, threshold=0.20):
    """Evaluate both models on held-out test patients."""
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    print(f"Classification threshold: {threshold}")

    test_seq, test_feat, test_lab = data["test"]
    print(f"Test: {len(test_lab):,} epochs  ({int((test_lab==1).sum()):,} preictal)")
    print(f"Test patients: {data['test_pats']}")

    results = {}

    # Save test labels for plot generation
    np.save(output_dir / "test_labels.npy", test_lab)

    # ── LSTM ──
    if lstm_model is not None:
        print("\n--- LSTM ---")
        lstm_model.eval()
        test_ds = SequenceDataset(test_seq, test_lab)
        use_cuda = device.type == "cuda"
        test_loader = DataLoader(
            test_ds, batch_size=128, shuffle=False,
            num_workers=0, pin_memory=use_cuda,
        )

        preds = []
        with torch.no_grad():
            for X_b, _ in tqdm(test_loader, desc="LSTM inference"):
                X_b = X_b.to(device, non_blocking=True)
                probs = torch.sigmoid(lstm_model(X_b)).cpu().numpy()
                preds.extend(probs.flatten())

        preds = np.array(preds)
        auc = roc_auc_score(test_lab, preds)
        pred_bin = (preds >= threshold).astype(int)
        report = classification_report(test_lab, pred_bin, output_dict=True, zero_division=0)

        sens = report.get("1.0", report.get("1", {})).get("recall", 0)
        spec = report.get("0.0", report.get("0", {})).get("recall", 0)
        ap = average_precision_score(test_lab, preds)

        print(f"  AUC:         {auc:.4f}")
        print(f"  AP:          {ap:.4f}")
        print(f"  Accuracy:    {report['accuracy']:.4f}")
        print(f"  Sensitivity: {sens:.4f}")
        print(f"  Specificity: {spec:.4f}")

        results["lstm"] = {
            "auc": float(auc), "ap": float(ap),
            "accuracy": float(report["accuracy"]),
            "sensitivity": float(sens), "specificity": float(spec),
            "threshold": threshold,
            "classification_report": report,
        }

        np.save(output_dir / "lstm_test_preds.npy", preds)

    # ── XGBoost ──
    if xgb_model is not None:
        print("\n--- XGBoost ---")
        xgb_preds = xgb_model.predict_proba(test_feat)[:, 1]
        auc = roc_auc_score(test_lab, xgb_preds)
        pred_bin = (xgb_preds >= threshold).astype(int)
        report = classification_report(test_lab, pred_bin, output_dict=True, zero_division=0)

        sens = report.get("1.0", report.get("1", {})).get("recall", 0)
        spec = report.get("0.0", report.get("0", {})).get("recall", 0)
        ap = average_precision_score(test_lab, xgb_preds)

        print(f"  AUC:         {auc:.4f}")
        print(f"  AP:          {ap:.4f}")
        print(f"  Accuracy:    {report['accuracy']:.4f}")
        print(f"  Sensitivity: {sens:.4f}")
        print(f"  Specificity: {spec:.4f}")

        results["xgboost"] = {
            "auc": float(auc), "ap": float(ap),
            "accuracy": float(report["accuracy"]),
            "sensitivity": float(sens), "specificity": float(spec),
            "threshold": threshold,
            "classification_report": report,
        }

        np.save(output_dir / "xgb_test_preds.npy", xgb_preds)

    # Save results
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Train LSTM + XGBoost (Cloud GPU)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--quick-test", action="store_true",
                        help="3 epochs, 2000 epochs/patient — for smoke testing")
    parser.add_argument("--lstm-only", action="store_true")
    parser.add_argument("--xgb-only", action="store_true")
    parser.add_argument("--max-epochs-per-patient", type=int, default=None,
                        help="Cap epochs per patient to limit memory")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.quick_test:
        config["lstm"]["max_epochs"] = 3
        config["lstm"]["batch_size"] = 64
        args.max_epochs_per_patient = 2000
        print("⚡ QUICK TEST MODE: 3 LSTM epochs, 64 batch size, 2000 epochs/patient\n")

    # ── Device ──
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("WARNING: No CUDA GPU detected — training will be very slow on CPU")

    # ── Timestamped output directory ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "models" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # ── Load data ──
    data = load_all_data(config, max_epochs_per_patient=args.max_epochs_per_patient)

    # ── Train ──
    lstm_model = None
    xgb_model = None
    history = []

    if not args.xgb_only:
        lstm_model, best_auc, history = train_lstm(data, config, device, output_dir)

    if not args.lstm_only:
        xgb_model, xgb_metrics = train_xgb(data, config, output_dir)

    # ── Load best LSTM checkpoint for evaluation ──
    if lstm_model is not None:
        ckpt = torch.load(output_dir / "lstm_best.pt", map_location=device,
                          weights_only=False)
        lstm_model.load_state_dict(ckpt["model_state_dict"])
        lstm_model.to(device)

    # ── Test evaluation ──
    threshold = config.get("lstm", {}).get("classification_threshold", 0.20)
    results = evaluate_test(lstm_model, xgb_model, data, device, output_dir, threshold)

    # ── Generate plots ──
    generate_training_plots(history, results, output_dir, threshold)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    if "lstm" in results:
        print(f"  LSTM  — AUC: {results['lstm']['auc']:.4f}  AP: {results['lstm']['ap']:.4f}  "
              f"Sens: {results['lstm']['sensitivity']:.4f}  "
              f"Spec: {results['lstm']['specificity']:.4f}")
    if "xgboost" in results:
        print(f"  XGB   — AUC: {results['xgboost']['auc']:.4f}  AP: {results['xgboost']['ap']:.4f}  "
              f"Sens: {results['xgboost']['sensitivity']:.4f}  "
              f"Spec: {results['xgboost']['specificity']:.4f}")

    print(f"\n  Outputs in: {output_dir}/")
    print(f"    lstm_best.pt          — LSTM model weights")
    print(f"    lstm_history.json     — training curves")
    print(f"    xgboost_model.pkl     — XGBoost model")
    print(f"    test_results.json     — test evaluation")
    print(f"    lstm_test_preds.npy   — LSTM test predictions (for fusion)")
    print(f"    xgb_test_preds.npy    — XGBoost test predictions (for fusion)")
    print(f"    *.png                 — training plots")


# ============================================================
# Plot Generation
# ============================================================
def generate_training_plots(history, results, output_dir, threshold):
    """Generate and save training diagnostic plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARNING] matplotlib not installed; skipping plot generation")
        return

    if not history:
        print("  [WARNING] No training history; skipping plot generation")
        return

    figs = []
    epochs = [h["epoch"] for h in history]

    # 1. Loss curves
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [h["train_loss"] for h in history], label="Train Loss", marker="o", markersize=3)
    ax.plot(epochs, [h["val_loss"] for h in history], label="Val Loss", marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Focal Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    figs.append(("loss_curve.png", fig))

    # 2. Validation AUC
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [h["val_auc"] for h in history], marker="o", markersize=4, color="tab:green")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Random (AUC=0.5)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUC")
    ax.set_title("Validation AUC Over Epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    figs.append(("val_auc_curve.png", fig))

    # 3. Validation Sensitivity / Specificity
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [h.get("val_sens", 0) for h in history], label="Sensitivity", marker="o", markersize=3)
    ax.plot(epochs, [h.get("val_spec", 0) for h in history], label="Specificity", marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Rate")
    ax.set_title(f"Validation Sensitivity / Specificity (threshold={threshold})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    figs.append(("sens_spec_curve.png", fig))

    # 4. Learning Rate schedule
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [h["lr"] for h in history], marker="o", markersize=3, color="tab:orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    figs.append(("lr_schedule.png", fig))

    # 5. Test ROC curves
    fig, ax = plt.subplots(figsize=(7, 7))
    test_labels = np.load(output_dir / "test_labels.npy")
    for model_name, color in [("lstm", "tab:blue"), ("xgboost", "tab:orange")]:
        if model_name in results:
            preds = np.load(output_dir / f"{model_name}_test_preds.npy")
            fpr, tpr, _ = roc_curve(test_labels, preds)
            auc = results[model_name]["auc"]
            ax.plot(fpr, tpr, label=f"{model_name.upper()} (AUC={auc:.3f})", color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Test Set")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    figs.append(("roc_curve.png", fig))

    # 6. Test Precision-Recall curves
    fig, ax = plt.subplots(figsize=(7, 7))
    for model_name, color in [("lstm", "tab:blue"), ("xgboost", "tab:orange")]:
        if model_name in results:
            preds = np.load(output_dir / f"{model_name}_test_preds.npy")
            precision, recall, _ = precision_recall_curve(test_labels, preds)
            ap = results[model_name]["ap"]
            ax.plot(recall, precision, label=f"{model_name.upper()} (AP={ap:.3f})", color=color, linewidth=2)
    # Baseline: random classifier
    baseline = test_labels.mean()
    ax.axhline(baseline, color="red", linestyle="--", alpha=0.5, label=f"Baseline ({baseline:.3f})")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Test Set")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    figs.append(("pr_curve.png", fig))

    # 7. Test Confusion Matrices
    for model_name in ["lstm", "xgboost"]:
        if model_name in results:
            preds = np.load(output_dir / f"{model_name}_test_preds.npy")
            pred_bin = (preds >= threshold).astype(int)
            cm = confusion_matrix(test_labels, pred_bin)
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            ax.set(
                xticks=[0, 1], yticks=[0, 1],
                xticklabels=["Interictal", "Preictal"],
                yticklabels=["Interictal", "Preictal"],
                title=f"Confusion Matrix — {model_name.upper()} (threshold={threshold})",
                ylabel="True label",
                xlabel="Predicted label",
            )
            # Annotate cells
            thresh = cm.max() / 2.0
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, format(cm[i, j], ","),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black",
                            fontsize=12, fontweight="bold")
            figs.append((f"confusion_matrix_{model_name}.png", fig))

    # Save all figures
    for fname, fig in figs:
        fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  Plots saved ({len(figs)} images) → {output_dir}/")


if __name__ == "__main__":
    main()
