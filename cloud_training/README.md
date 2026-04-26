# Cloud Training — EEG-Genetic Fusion

## What this does
Runs the complete training pipeline on a cloud GPU:
1. Downloads CHB-MIT EEG data from PhysioNet (~3 GB selective)
2. Preprocesses EEG → feature extraction (~2–3 hours)
3. Trains BiLSTM on raw EEG sequences (~1–2 hours on RTX 4050)
4. Trains XGBoost on 221-dim feature vectors (~5 minutes)
5. Evaluates both on held-out test patients
6. Saves all model files for download

## Requirements
- **GPU**: NVIDIA RTX 4050 or better (8+ GB VRAM)
- **RAM**: 16+ GB
- **Disk**: 30 GB free
- **Python**: 3.10+
- **CUDA**: 11.8+ with cuDNN

## Usage

### Option A: Full pipeline (recommended)
```bash
# Clone the repo
git clone <your-repo-url> eeg-genetic-fusion
cd eeg-genetic-fusion

# Run everything
bash cloud_training/run_all.sh
```

### Option B: Skip download (if data is already present)
```bash
bash cloud_training/run_all.sh --skip-download
```

### Option C: Training only (data already preprocessed)
```bash
bash cloud_training/run_all.sh --train-only
```

### Option D: Quick smoke test
```bash
bash cloud_training/run_all.sh --quick-test
```

## What to download when done
After training completes, download the `models/` folder:
```
models/
├── lstm_best.pt          # LSTM model weights (~5 MB)
├── lstm_history.json     # Training curves (for plotting)
├── xgboost_model.pkl     # XGBoost model (~2 MB)
├── xgboost_metrics.json  # XGBoost validation metrics
├── test_results.json     # Full test evaluation
├── lstm_test_preds.npy   # LSTM predictions (for fusion layer)
└── xgb_test_preds.npy    # XGBoost predictions (for fusion layer)
```

## Expected training time
| Component | RTX 4050 | T4 (Colab) |
|-----------|----------|------------|
| Data download | 20–40 min | 10–20 min |
| Preprocessing | 2–3 hours | 3–4 hours |
| LSTM training (50 epochs) | 1–2 hours | 2–4 hours |
| XGBoost training | 5 min | 5 min |
| **Total** | **~4–5 hours** | **~6–8 hours** |

## Files in this folder
| File | Purpose |
|------|---------|
| `run_all.sh` | Master script — run this ONE file |
| `01_download_and_preprocess.py` | Downloads EDF files + runs preprocessing |
| `02_train_models.py` | Trains LSTM + XGBoost |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |
