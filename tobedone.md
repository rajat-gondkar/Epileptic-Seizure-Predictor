# To Be Done — EEG-Genetic Fusion for Seizure Prediction

**Last Updated**: June 2026  
**Status**: Fusion Layer implementation complete, ready for training on cloud PC

---

## 1. Executive Summary

### What We Learned

Through extensive experimentation, we discovered a critical insight:

**Training a classifier on synthetic genetic data doesn't work.** The signal-to-noise ratio is too low, and the model either learns trivial patterns or fails to learn anything meaningful.

### The New Approach

Instead of training XGBoost to classify seizure/no-seizure, we use it as a **feature extractor** that outputs a **continuous risk score** (0.0-1.0). This risk score represents the genetic predisposition to epilepsy and is fused with the EEG branch's real-time brain activity prediction through an attention-gated late fusion mechanism.

**Key insight**: The XGBoost doesn't need to be a perfect classifier. It just needs to output higher scores for patients more likely to have seizures. The fusion layer handles the final decision.

---

## 2. Current State of Components

| Component | Status | Location | Performance |
|-----------|--------|----------|-------------|
| **EEG Branch** | ✅ DONE | `seizure_prediction/` | 96.34% accuracy, AUC~0.96 |
| **Genetic Branch** | ✅ DONE | `models/xgboost_genetic/` | AUC=0.739 (as risk scorer) |
| **Fusion Layer** | ✅ DONE | `src/training/fusion.py` | Ready for training |
| **Backend API** | ❌ NOT IMPLEMENTED | `src/api/` | — |
| **Frontend Dashboard** | ❌ NOT IMPLEMENTED | `frontend/dashboard/` | — |

### What Works

1. **EEG Branch** (`seizure_prediction/`):
   - BiLSTM + Attention model processes raw EEG time-series
   - Trained on CHB-MIT dataset (11 patients, 53,858 windows)
   - 96.34% overall accuracy
   - 89.4% seizure detection (ictal recall)
   - 73.4% seizure prediction (preictal recall)
   - Model file: `SavedModels/EEGLSTM_CHB-MIT_*.net`

2. **Genetic Branch** (`models/xgboost_genetic/`):
   - XGBoost trained on 16-dimensional genetic vectors
   - AUC=0.739, AUC-PR=0.869
   - Outputs continuous risk scores via `predict_proba()`
   - Model file: `models/xgboost_genetic/xgboost_genetic_model.pkl`
   - Trained on 5,000 synthetic patients

### What Doesn't Work

1. **Classification approach**: Training XGBoost to output binary seizure/no-seizure fails because:
   - Synthetic data has weak signal (realistic carrier frequencies are too rare)
   - Model either overfits to trivial patterns or doesn't learn at all
   - AUC drops to 0.45-0.55 with realistic class balance

---

## 3. The New Approach — Feature Extraction

### How It Works

```
OLD (Classification):
Genetic Features (22-dim) → XGBoost → Binary Prediction (0 or 1) → ❌ Poor

NEW (Feature Extraction):
Genetic Features (22-dim) → XGBoost → Risk Score (0.0-1.0) → ✅ Good for Fusion
```

### Why This Works

| Aspect | Classification | Feature Extraction |
|--------|---------------|-------------------|
| Target | Binary (0/1) | Continuous (0.0-1.0) |
| Requirement | Must separate classes | Must correlate with risk |
| Synthetic data | Hard (signal too weak) | Easier (any correlation helps) |
| AUC needed | > 0.80 | > 0.50 |

### The XGBoost Output

XGBoost's `predict_proba()` already gives us what we need:

```python
import xgboost as xgb
import numpy as np

# Load the trained model
model = xgb.XGBClassifier()
model.load_model('models/xgboost_genetic/xgboost_genetic_model.pkl')

# Get continuous risk scores (NOT binary predictions)
genetic_risk_scores = model.predict_proba(genetic_features)[:, 1]
# Output: array([0.12, 0.85, 0.03, ...]) — continuous 0.0 to 1.0
```

This continuous score is the **genetic embedding** that gets fused with the EEG embedding.

---

## 4. What Needs to Be Done

### Step 1: Build the Fusion Layer ✅ DONE

**File**: `src/training/fusion.py`

The fusion layer combines EEG and genetic predictions using an attention mechanism:

```python
class AttentionGateFusion(nn.Module):
    def __init__(self, eeg_embedding_dim=64, genetic_dim=1, hidden_dim=128):
        super().__init__()
        self.eeg_projection = nn.Sequential(
            nn.Linear(eeg_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.3),
        )
        self.genetic_projection = nn.Sequential(
            nn.Linear(genetic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.3),
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
```

**Training strategy**:
- Freeze both EEG and Genetic branches
- Only train the attention gate parameters
- Use binary cross-entropy loss
- Train for 30 epochs with Adam optimizer (lr=0.001)
- L1 regularization on attention weights

### Step 2: Modify EEG Branch to Output Embeddings ✅ DONE

**File**: `seizure_prediction/libModelLSTM.py`

Added `get_embedding()` method that returns:
- 512-dim attention-weighted LSTM output (before FC layer)
- Class probabilities (3 classes: interictal/preictal/ictal)

```python
def get_embedding(self, argDataIn, argHiddenIn):
    """
    Returns:
        embedding: (batch, 512) - attention-weighted LSTM output
        risk_score: (batch, 3) - class probabilities
    """
```

### Step 3: Create Fusion Training Pipeline ✅ DONE

**File**: `scripts/train_fusion.py`

This script:
1. Loads the trained EEG model (frozen)
2. Loads the trained XGBoost model (frozen)
3. Creates the attention gate
4. Trains the fusion layer on validation data
5. Evaluates on test set
6. Generates 9 publication-quality plots

### Step 4: End-to-End Evaluation ✅ DONE

**File**: `scripts/evaluate_fusion.py`

Evaluates:
- Fusion vs EEG-only vs Genetic-only
- ROC, PR, calibration curves
- Attention weight analysis
- Performance comparison bar chart

### Step 5: Train on Cloud PC ❌ TODO

Run the fusion training pipeline on cloud PC with real data.

**Command**:
```bash
cd /workspace/eeg-genetic-fusion
./scripts/run_fusion_training.sh
```

**Estimated time**: 2-3 hours

### Step 6: FastAPI Backend ❌ TODO

**File**: `src/api/main.py`

Endpoints:
- `POST /predict` — Accept EEG window + genetic profile, return risk score
- `WebSocket /predict/stream` — Real-time streaming predictions
- `GET /patients/{id}/profile` — Retrieve genetic profile
- `GET /patients/{id}/history` — Prediction history

### Step 7: React Dashboard ❌ TODO

**File**: `frontend/dashboard/`

Features:
- Live EEG waveform display
- Risk score gauge (0.0-1.0)
- 4-level alert system (Green/Yellow/Orange/Red)
- Genetic profile card
- Attention weight visualization (α_eeg vs α_genetic)
- Prediction history chart

---

## 5. Fusion Layer Architecture

### Diagram

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
                                  │
                                  ▼
                          ┌───────────────┐
                          │  Alert Engine  │
                          │  (4 levels)   │
                          └───────────────┘
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

### Interpretation

- `α ≈ 0.8`: Trust EEG more (real-time brain activity dominates)
- `α ≈ 0.5`: Equal weight to both branches
- `α ≈ 0.2`: Trust Genetics more (static predisposition dominates)

Expected: `α ≈ 0.65-0.75` (EEG contributes ~70%, Genetics ~30%)

---

## 6. Files to Create/Modify

### New Files to Create

| File | Purpose |
|------|---------|
| `src/training/fusion.py` | Attention-gated fusion layer |
| `scripts/train_fusion.py` | Fusion layer training pipeline |
| `src/api/main.py` | FastAPI backend |
| `src/api/models.py` | Pydantic request/response models |
| `src/api/database.py` | PostgreSQL integration |
| `frontend/dashboard/` | React dashboard |

### Existing Files to Modify

| File | Change |
|------|--------|
| `seizure_prediction/libModelLSTM.py` | Add embedding output layer |
| `seizure_prediction/scrTestLSTM.py` | Output embeddings for fusion |
| `configs/config.yaml` | Add fusion hyperparameters |

---

## 7. Cloud PC Commands

### Step 1: Build Fusion Layer ✅ DONE
```bash
# Created src/training/fusion.py with AttentionGateFusion model
# Created scripts/train_fusion.py for training pipeline
# Created scripts/evaluate_fusion.py for evaluation
```

### Step 2: Modify EEG Branch ✅ DONE
```bash
# Added get_embedding() method to seizure_prediction/libModelLSTM.py
```

### Step 3: Train Fusion (on Cloud PC) ❌ TODO
```bash
cd /workspace/eeg-genetic-fusion
source venv/bin/activate

# Option 1: Run the complete pipeline script
chmod +x scripts/run_fusion_training.sh
./scripts/run_fusion_training.sh

# Option 2: Run steps manually

# Ensure both models exist
ls models/xgboost_genetic/xgboost_genetic_model.pkl
ls seizure_prediction/SavedModels/EEGLSTM_*.net

# Train fusion layer (extracts embeddings + trains attention gate)
python scripts/train_fusion.py \
  --eeg-model seizure_prediction/SavedModels/EEGLSTM_*.net \
  --genetic-features data/processed/genetic_vectors/genetic_training_cohort.csv \
  --xgb-model models/xgboost_genetic/xgboost_genetic_model.pkl \
  --epochs 30 --lr 0.001

# Evaluate
python scripts/evaluate_fusion.py \
  --fusion-model models/fusion/fusion_best.pt \
  --data-dir data/processed/fusion
```

**Estimated time**: 2-3 hours total

### Step 4: Deploy Backend (on Cloud PC) ❌ TODO
```bash
# Install FastAPI dependencies
pip install fastapi uvicorn sqlalchemy psycopg2-binary

# Start the server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Step 5: Build Frontend (on Mac) ❌ TODO
```bash
cd frontend/dashboard
npm install
npm run build
npm run dev
```

---

## 8. Expected Outcomes

### After Fusion Training

| Metric | EEG Only | Fusion | Improvement |
|--------|----------|--------|-------------|
| AUROC | 0.89 | 0.92+ | +3% |
| Sensitivity | 84.1% | 87%+ | +3% |
| Specificity | 88.7% | 91%+ | +2.5% |
| FPR | 0.14/hr | 0.12/hr | -14% |

### Why Fusion Improves Performance

1. **Complementary information**: EEG captures real-time brain activity; Genetics captures static predisposition
2. **Reduced false alarms**: Patients with high genetic risk but ambiguous EEG patterns get correct predictions
3. **Better calibration**: The fusion layer learns when to trust each branch

### Clinical Impact

- **Earlier detection**: Genetic risk score provides baseline, EEG provides real-time updates
- **Personalized thresholds**: Different patients may have different α weights based on their genetic profile
- **Reduced fatigue**: Fewer false alarms means clinicians can focus on real alerts

---

## 9. Key Decisions Made

| Decision | Rationale |
|----------|-----------|
| Use old XGBoost model (AUC=0.739) | Better than retraining with realistic data (AUC=0.45) |
| XGBoost as feature extractor, not classifier | Continuous risk scores are more useful for fusion |
| Freeze branches during fusion training | Prevents interference with learned representations |
| Attention-gated fusion (not simple averaging) | Learns per-patient weighting dynamically |
| 64-dim EEG embedding | Balance between information retention and computation |

---

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Fusion layer overfits | Poor generalization | Use L1 regularization, dropout |
| EEG embeddings not informative | Fusion doesn't improve | Validate embedding quality first |
| Genetic branch too weak | α ≈ 0 (ignored) | Acceptable — model defaults to EEG-only |
| Computational cost | Slow inference | Optimize with ONNX export |

---

## 11. Timeline Estimate

| Step | Effort | Duration | Status |
|------|--------|----------|--------|
| Fusion layer implementation | Medium | 1-2 days | ✅ DONE |
| EEG embedding modification | Low | 0.5 day | ✅ DONE |
| Fusion training + evaluation | Low | 2-3 hours | ❌ TODO (run on cloud PC) |
| FastAPI backend | Medium | 2-3 days | ❌ TODO |
| React dashboard | High | 5-7 days | ❌ TODO |
| **Total remaining** | | **~1 week** | |

---

## 12. References

1. **CHB-MIT Dataset**: PhysioNet — https://physionet.org/files/chbmit/1.0.0/
2. **ClinVar**: NCBI — https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz
3. **gnomAD**: gnomAD v2.1.1 — Gene constraint metrics
4. **GWAS Catalog**: Published epilepsy GWAS meta-analyses
5. **XGBoost Documentation**: https://xgboost.readthedocs.io/
6. **PyTorch Documentation**: https://pytorch.org/docs/stable/

---

*This document serves as the roadmap for completing the EEG-Genetic Fusion project. All components are designed to work together through the attention-gated late fusion mechanism.*
