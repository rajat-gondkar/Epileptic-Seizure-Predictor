# To Be Done — EEG-Genetic Fusion for Seizure Prediction

**Last Updated**: June 2026  
**Status**: CTGAN validation set created, ready for fusion training

---

## Current State of Components

| Component | Status | Location | Performance |
|-----------|--------|----------|-------------|
| **EEG Branch** | ✅ DONE | `seizure_prediction/` | BiLSTM+Attention trained |
| **Genetic Branch** | ✅ DONE | `models/xgboost_genetic/` | AUC=0.739 |
| **CTGAN Validation Set** | ✅ DONE | `data/processed/synthetic/` | KS=31.6% (genetic 100%) |
| **Fusion Layer** | ❌ NOT IMPLEMENTED | — | — |
| **Backend API** | ❌ NOT IMPLEMENTED | — | — |
| **Frontend Dashboard** | ❌ NOT IMPLEMENTED | — | — |

---

## What Was Completed in This Session

### 1. CTGAN Synthetic Data Generation (v3)

**File**: `seizure_prediction/generate_ctgan_data.py`

**Approach**: Joint CTGAN training on combined EEG+genetic features with post-generation clipping.

**Results**:
- 279 real samples extracted from EDF files
- 5000 synthetic samples generated
- Genetic features: 100% KS pass rate
- EEG features: 0% KS pass rate (expected with 279 samples)
- All sanity checks pass (no negatives, proper bounds)

**Decision**: Use as validation set only, not for fusion training.

### 2. Data Files Verified

| File | Location | Status |
|------|----------|--------|
| `epilepsy_variants.csv` | `data/raw/clinvar/` | ✅ 13,331 variants |
| `pli_scores.csv` | `data/raw/gnomad/` | ✅ 9 genes |
| `epilepsy_snps.csv` | `data/raw/gwas/` | ✅ 15 SNPs |

### 3. Dependencies Installed (Cloud PC)

```bash
pip install numpy scipy pandas pyyaml pyedflib wfdb scikit-learn matplotlib tqdm psutil pytz sdv ctgan torch
```

---

## What Needs to Be Done Next

### Step 1: Train Fusion Layer ❌ TODO

**Why**: Combine EEG + Genetic branches for better prediction than either alone.

**Approach**:
- Freeze EEG BiLSTM and XGBoost models
- Train attention gate on validation data
- Use real paired data (279 files × EEG + genetic)

**Files needed**:
- `src/training/fusion.py` — Attention-gated fusion model
- `scripts/train_fusion.py` — Training pipeline
- `scripts/evaluate_fusion.py` — Evaluation scripts

**Command**:
```bash
cd /workspace/eeg-genetic-fusion
source venv/bin/activate
python scripts/train_fusion.py \
  --eeg-model seizure_prediction/SavedModels/EEGLSTM_CHB-MIT_*.net \
  --genetic-features data/processed/genetic_vectors/genetic_training_cohort.csv \
  --xgb-model models/xgboost_genetic/xgboost_genetic_model.pkl \
  --epochs 30 --lr 0.001
```

**Estimated time**: 2-3 hours

### Step 2: End-to-End Evaluation ❌ TODO

**Why**: Validate fusion improves over individual branches.

**Approach**:
- Test on held-out 76 EDF files (`all_patients_test.csv`)
- Compare Fusion vs EEG-only vs Genetic-only
- Generate ROC, PR, calibration plots

**Command**:
```bash
python scripts/evaluate_fusion.py \
  --fusion-model models/fusion/fusion_best.pt \
  --data-dir data/processed/fusion
```

### Step 3: FastAPI Backend ❌ TODO

**Why**: REST API for real-time predictions.

**Endpoints**:
- `POST /predict` — Accept EEG window + genetic profile, return risk score
- `WebSocket /predict/stream` — Real-time streaming predictions
- `GET /patients/{id}/profile` — Retrieve genetic profile
- `GET /patients/{id}/history` — Prediction history

**Dependencies**:
```bash
pip install fastapi uvicorn sqlalchemy psycopg2-binary
```

### Step 4: React Dashboard ❌ TODO

**Why**: Real-time clinical monitoring interface.

**Features**:
- Live EEG waveform display
- Risk score gauge (0.0-1.0)
- 4-level alert system (Green/Yellow/Orange/Red)
- Genetic profile card
- Attention weight visualization
- Prediction history chart

---

## Architecture Reference

### Fusion Layer Design

```
                    ┌─────────────────────────────────────┐
                    │         ATTENTION GATE              │
                    │                                     │
EEG Embedding ─────┤  α = σ(W_e·h_eeg + W_g·h_gen + b)  │
    (64-dim)       │                                     │
                    │  P_final = α·P_eeg + (1-α)·P_gen   │
                    │                                     │
Genetic Score ─────┤                                     │
    (1-dim)        └──────────────┬──────────────────────┘
                                  │
                                  ▼
                          Final Risk Score
                             (0.0-1.0)
```

### Mathematical Formulation

Given:
- `h_eeg ∈ ℝ⁶⁴`: EEG embedding from BiLSTM
- `P_genetic ∈ [0,1]`: Genetic risk score from XGBoost

The attention gate computes:
```
α = σ(W_e · h_eeg + W_g · P_genetic + b)
P_final = α · P_eeg + (1 − α) · P_genetic
```

Where:
- `α ∈ [0,1]`: Attention weight (how much to trust EEG vs Genetic)
- `P_final ∈ [0,1]`: Final fused seizure risk score

**Expected**: `α ≈ 0.65-0.75` (EEG contributes ~70%, Genetics ~30%)

---

## Key Decisions Made

| Decision | Rationale |
|----------|-----------|
| Use old XGBoost model (AUC=0.739) | Better than retraining with realistic data (AUC=0.45) |
| XGBoost as feature extractor, not classifier | Continuous risk scores are more useful for fusion |
| CTGAN as validation set only | EEG distributions don't match real data well enough |
| Freeze branches during fusion training | Prevents interference with learned representations |

---

## Timeline Estimate

| Step | Effort | Duration | Status |
|------|--------|----------|--------|
| CTGAN validation set | Medium | 1 day | ✅ DONE |
| Fusion layer implementation | Medium | 1-2 days | ❌ TODO |
| Fusion training + evaluation | Low | 2-3 hours | ❌ TODO |
| FastAPI backend | Medium | 2-3 days | ❌ TODO |
| React dashboard | High | 5-7 days | ❌ TODO |
| **Total remaining** | | **~1 week** | |

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Fusion layer overfits | Poor generalization | Use L1 regularization, dropout |
| EEG embeddings not informative | Fusion doesn't improve | Validate embedding quality first |
| Genetic branch too weak | α ≈ 0 (ignored) | Acceptable — model defaults to EEG-only |
| Computational cost | Slow inference | Optimize with ONNX export |

---

## References

1. **CHB-MIT Dataset**: PhysioNet — https://physionet.org/files/chbmit/1.0.0/
2. **ClinVar**: NCBI — https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz
3. **gnomAD**: gnomAD v2.1.1 — Gene constraint metrics
4. **GWAS Catalog**: Published epilepsy GWAS meta-analyses
5. **XGBoost Documentation**: https://xgboost.readthedocs.io/
6. **PyTorch Documentation**: https://pytorch.org/docs/stable/

---

*This document serves as the roadmap for completing the EEG-Genetic Fusion project. The CTGAN validation set is complete. Next step is fusion layer training on cloud PC.*
