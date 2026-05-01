#!/bin/bash
# ============================================================
# EEG-Genetic Fusion — Cloud Training Pipeline
# ============================================================
# Run this ONE script on your cloud GPU instance.
# It handles everything: dependencies → data download → preprocessing → training.
#
# Prerequisites:
#   - Ubuntu/Debian with Python 3.10+ and CUDA
#   - git clone the repo, then cd into it
#
# Usage:
#   cd eeg-genetic-fusion   (or whatever the repo folder is called)
#   bash cloud_training/run_all.sh
#
# For quick smoke test (3 epochs, tiny data):
#   bash cloud_training/run_all.sh --quick-test
#
# To skip data download (if data already present):
#   bash cloud_training/run_all.sh --skip-download
#
# To only train (skip download AND preprocess):
#   bash cloud_training/run_all.sh --train-only
# ============================================================

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "EEG-Genetic Fusion — Cloud Training Pipeline"
echo "============================================================"
echo "Project root: $PROJECT_ROOT"
echo "Time: $(date)"
echo ""

# Parse arguments
QUICK_TEST=""
SKIP_DOWNLOAD=false
TRAIN_ONLY=false

for arg in "$@"; do
    case $arg in
        --quick-test)
            QUICK_TEST="--quick-test"
            echo "⚡ Quick test mode enabled"
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            echo "⏭ Skipping downloads (will still preprocess if needed)"
            ;;
        --train-only)
            TRAIN_ONLY=true
            echo "⏭ Training only (skip download + preprocess)"
            ;;
    esac
done
echo ""

# ── Step 0: Install dependencies ──
echo "============================================================"
echo "STEP 0: Installing dependencies"
echo "============================================================"
pip install -r cloud_training/requirements.txt 2>&1 | tail -5
echo "✅ Dependencies installed"
echo ""

# ── Step 1: Download & Preprocess EEG data ──
if [ "$TRAIN_ONLY" = false ]; then
    echo "============================================================"
    echo "STEP 1: Download & Preprocess EEG data"
    echo "============================================================"

    if [ "$SKIP_DOWNLOAD" = false ]; then
        # Download + preprocess all 8 patients
        python cloud_training/01_download_and_preprocess.py \
            --patients chb01 chb03 chb05 chb06 chb08 chb10 chb16 chb20
    else
        echo "Skipping downloads, using existing files only (--skip-download)"
        python cloud_training/01_download_and_preprocess.py \
            --patients chb01 chb03 chb05 chb06 chb08 chb10 chb16 chb20 \
            --skip-download
    fi

    echo ""
    echo "Verifying data..."
    python -c "
import numpy as np
from pathlib import Path

feat_dir = Path('data/processed/eeg_features')
patients = ['chb01','chb03','chb05','chb06','chb08','chb10','chb16','chb20']
total = 0
ok = True
for pat in patients:
    sp = feat_dir / f'{pat}_sequences.npy'
    fp = feat_dir / f'{pat}_features.npy'
    lp = feat_dir / f'{pat}_labels.npy'
    if not all(p.exists() for p in [sp, fp, lp]):
        print(f'  ❌ {pat}: MISSING')
        ok = False
    else:
        n = len(np.load(lp))
        total += n
        print(f'  ✅ {pat}: {n:,} epochs')
if ok:
    print(f'\n  Total: {total:,} epochs — ALL GOOD')
else:
    print('\n  ⚠ Some patients missing — training may use fewer patients')
"
    echo ""
fi

# ── Step 2: Train models ──
echo "============================================================"
echo "STEP 2: Training LSTM + XGBoost"
echo "============================================================"
echo "Start time: $(date)"
echo ""

python cloud_training/02_train_models.py $QUICK_TEST

echo ""
echo "End time: $(date)"

# ── Summary ──
echo ""
echo "============================================================"
echo "ALL DONE"
echo "============================================================"
echo ""
echo "Download these files from the 'models/' directory:"
echo "  models/lstm_best.pt          — LSTM model weights"
echo "  models/lstm_history.json     — training curves"
echo "  models/xgboost_model.pkl     — XGBoost model"
echo "  models/test_results.json     — test set metrics"
echo "  models/lstm_test_preds.npy   — LSTM predictions (for fusion)"
echo "  models/xgb_test_preds.npy    — XGBoost predictions (for fusion)"
echo ""
echo "To view results:"
echo "  cat models/test_results.json | python -m json.tool"
