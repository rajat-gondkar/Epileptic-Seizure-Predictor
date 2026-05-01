#!/usr/bin/env python3
"""
Step 2: Train LSTM + XGBoost Models
======================================
Trains both branches of the EEG-Genetic Fusion pipeline:
  1. BiLSTM with self-attention on raw EEG sequences
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
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import roc_auc_score, classification_report
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

# ── Project root ──
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lstm_eeg import EEGBiLSTM
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
        x = torch.from_numpy(np.array(self.seqs[arr_idx][local_idx])).float()
        y = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)
        return x, y


# ============================================================
# LSTM Training
# ============================================================
def create_weighted_sampler(labels):
    """WeightedRandomSampler for class imbalance."""
    counts = np.bincount(labels.astype(int))
    weights = 1.0 / counts[labels.astype(int)]
    return WeightedRandomSampler(weights, len(weights), replacement=True)


def train_lstm(data, config, device, output_dir):
    """Full LSTM training with early stopping and gradient accumulation."""
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

    # ── DataLoaders ──
    sampler = create_weighted_sampler(train_lab)

    use_cuda = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], sampler=sampler,
        num_workers=4 if use_cuda else 0, pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
        num_workers=4 if use_cuda else 0, pin_memory=use_cuda,
    )

    # ── Model ──
    model = EEGBiLSTM(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        embedding_dim=64,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ── Loss ──
    # Class imbalance is handled by WeightedRandomSampler (oversampling).
    # No pos_weight needed to avoid double-correction.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
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

    # Header for status monitor
    print(f"\n  {'Epoch':>5} │ {'Train Loss':>10} │ {'Val Loss':>10} │ {'Val AUC':>8} │ "
          f"{'Sens':>6} │ {'Spec':>6} │ {'LR':>10} │ {'Time':>6} │ {'ETA':>8} │ Status")
    print("  " + "─" * 95)

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        # — Train —
        model.train()
        train_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}", leave=False)
        for X_b, y_b in pbar:
            X_b, y_b = X_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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

        scheduler.step(val_auc)
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history.append({
            "epoch": epoch, "train_loss": avg_train_loss,
            "val_loss": avg_val_loss, "val_auc": val_auc,
            "lr": lr_now, "time_sec": elapsed,
        })

        # Calculate sensitivity at threshold=0.5
        pred_bin = (preds_arr >= 0.5).astype(int)
        sens = (pred_bin[labels_arr == 1] == 1).mean() if (labels_arr == 1).any() else 0
        spec = (pred_bin[labels_arr == 0] == 0).mean() if (labels_arr == 0).any() else 0

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
def evaluate_test(lstm_model, xgb_model, data, device, output_dir):
    """Evaluate both models on held-out test patients."""
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    test_seq, test_feat, test_lab = data["test"]
    print(f"Test: {len(test_lab):,} epochs  ({int((test_lab==1).sum()):,} preictal)")
    print(f"Test patients: {data['test_pats']}")

    results = {}

    # ── LSTM ──
    if lstm_model is not None:
        print("\n--- LSTM ---")
        lstm_model.eval()
        test_ds = SequenceDataset(test_seq, test_lab)
        use_cuda = device.type == "cuda"
        test_loader = DataLoader(
            test_ds, batch_size=128, shuffle=False,
            num_workers=4 if use_cuda else 0, pin_memory=use_cuda,
        )

        preds = []
        with torch.no_grad():
            for X_b, _ in tqdm(test_loader, desc="LSTM inference"):
                X_b = X_b.to(device, non_blocking=True)
                probs = torch.sigmoid(lstm_model(X_b)).cpu().numpy()
                preds.extend(probs.flatten())

        preds = np.array(preds)
        auc = roc_auc_score(test_lab, preds)
        pred_bin = (preds >= 0.5).astype(int)
        report = classification_report(test_lab, pred_bin, output_dict=True, zero_division=0)

        sens = report.get("1.0", report.get("1", {})).get("recall", 0)
        spec = report.get("0.0", report.get("0", {})).get("recall", 0)

        print(f"  AUC:         {auc:.4f}")
        print(f"  Accuracy:    {report['accuracy']:.4f}")
        print(f"  Sensitivity: {sens:.4f}")
        print(f"  Specificity: {spec:.4f}")

        results["lstm"] = {
            "auc": float(auc), "accuracy": float(report["accuracy"]),
            "sensitivity": float(sens), "specificity": float(spec),
            "classification_report": report,
        }

        # Save predictions for later fusion
        np.save(output_dir / "lstm_test_preds.npy", preds)

    # ── XGBoost ──
    if xgb_model is not None:
        print("\n--- XGBoost ---")
        xgb_preds = xgb_model.predict_proba(test_feat)[:, 1]
        auc = roc_auc_score(test_lab, xgb_preds)
        pred_bin = (xgb_preds >= 0.5).astype(int)
        report = classification_report(test_lab, pred_bin, output_dict=True, zero_division=0)

        sens = report.get("1.0", report.get("1", {})).get("recall", 0)
        spec = report.get("0.0", report.get("0", {})).get("recall", 0)

        print(f"  AUC:         {auc:.4f}")
        print(f"  Accuracy:    {report['accuracy']:.4f}")
        print(f"  Sensitivity: {sens:.4f}")
        print(f"  Specificity: {spec:.4f}")

        results["xgboost"] = {
            "auc": float(auc), "accuracy": float(report["accuracy"]),
            "sensitivity": float(sens), "specificity": float(spec),
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
        config["lstm"]["batch_size"] = 32
        args.max_epochs_per_patient = 2000
        print("⚡ QUICK TEST MODE: 3 LSTM epochs, 32 batch size, 2000 epochs/patient\n")

    # ── Device ──
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("WARNING: No CUDA GPU detected — training will be very slow on CPU")

    output_dir = PROJECT_ROOT / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    data = load_all_data(config, max_epochs_per_patient=args.max_epochs_per_patient)

    # ── Train ──
    lstm_model = None
    xgb_model = None

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
    results = evaluate_test(lstm_model, xgb_model, data, device, output_dir)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    if "lstm" in results:
        print(f"  LSTM  — AUC: {results['lstm']['auc']:.4f}  "
              f"Sens: {results['lstm']['sensitivity']:.4f}  "
              f"Spec: {results['lstm']['specificity']:.4f}")
    if "xgboost" in results:
        print(f"  XGB   — AUC: {results['xgboost']['auc']:.4f}  "
              f"Sens: {results['xgboost']['sensitivity']:.4f}  "
              f"Spec: {results['xgboost']['specificity']:.4f}")

    print(f"\n  Outputs in: {output_dir}/")
    print(f"    lstm_best.pt          — LSTM model weights")
    print(f"    lstm_history.json     — training curves")
    print(f"    xgboost_model.pkl     — XGBoost model")
    print(f"    test_results.json     — test evaluation")
    print(f"    lstm_test_preds.npy   — LSTM test predictions (for fusion)")
    print(f"    xgb_test_preds.npy    — XGBoost test predictions (for fusion)")


if __name__ == "__main__":
    main()
