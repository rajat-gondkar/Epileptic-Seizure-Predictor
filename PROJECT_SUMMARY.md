# Project Summary Report
## AI-Based EEG + Genetic Marker Fusion for Epileptic Seizure Prediction

**Institution**: RVCE IDP  
**Team**: Rajat G A, Srijeeta Ghosh, Aamir Ibrahim, Misab Abdul Raheem  
**Guide**: Prof. Deepika P

---

## 1. What We Built

A multimodal machine learning system that predicts epileptic seizures by combining **brainwave (EEG) data** with **genetic risk markers**. The system outputs a personalized seizure risk score between 0 and 1, mapped to a 4-level clinical alert system (Low → Critical).

### Why this matters
- Epilepsy affects ~50 million people worldwide.
- Current seizure prediction relies heavily on EEG alone, missing the biological predisposition encoded in a patient’s genetics.
- By fusing both data types, the model aims to improve sensitivity (catching more true pre-seizure states) while keeping false alarms low.

---

## 2. System Architecture

The system uses a **late-fusion** design — two independent models make predictions, and a final fusion layer weighs them dynamically per patient:

1. **EEG Branch (BiLSTM)**: Reads raw 5-second EEG windows and learns temporal patterns that precede seizures.
2. **Genetic Branch (XGBoost)**: Uses a 12-dimensional genetic profile to estimate baseline seizure susceptibility.
3. **Fusion Layer**: An attention gate learns how much to trust the EEG signal vs. the genetic profile for each patient, producing the final risk score.

---

## 3. Datasets Used

### EEG Data — CHB-MIT Scalp EEG Database
- **Source**: PhysioNet (publicly available)
- **Patients processed**: 8 pediatric patients (chb01, chb03, chb05, chb06, chb08, chb10, chb16, chb20)
- **Total EDF files**: 190 recordings
- **Files with seizures**: 59
- **Sampling rate**: 256 Hz
- **Total extracted epochs**: ~333,600 five-second windows
  - ~301,000 interictal (normal)
  - ~32,600 preictal (within 30 minutes before a seizure)

### Genetic Databases
Because the CHB-MIT dataset does not include real genetic information, we built simulated genetic profiles using population-level data from three public sources:

| Database | What it provides | Used for |
|----------|------------------|----------|
| **ClinVar** | 13,331 pathogenic variants across 9 epilepsy genes | Mutation flags |
| **gnomAD** | Gene constraint scores (pLI) | Measuring how intolerant a gene is to harmful variants |
| **GWAS Catalog** | 15 epilepsy-linked SNPs with effect sizes | Polygenic Risk Score (PRS) |

---

## 4. EEG Preprocessing Pipeline

Raw EEG signals are noisy. We apply a standardized cleaning pipeline before training:

1. **Channel selection**: Pick 17 standard 10-20 bipolar channels; drop duplicates.
2. **Filtering**:
   - Bandpass 0.5–70 Hz (keeps clinically relevant brainwaves)
   - Notch 60 Hz (removes electrical mains noise)
   - Common Average Reference (re-centers all channels)
3. **Artifact rejection**: Discard epochs where any channel exceeds 500 µV (muscle movement, electrode pops).
4. **Epoching**: Cut the continuous recording into overlapping 5-second windows (1-second stride).
5. **Labelling**:
   - **Interictal** = 0 (normal, far from seizures)
   - **Preictal** = 1 (within 30 minutes before seizure onset)
   - **Excluded** = -1 (ictal/postictal/artifacts)

**Class imbalance**: ~9.2 interictal epochs for every 1 preictal epoch. This is handled by weighted sampling during training.

---

## 5. Feature Extraction

For the XGBoost branch, we extract **13 features per channel** from every clean epoch:

- 5 frequency band powers (Delta, Theta, Alpha, Beta, Gamma)
- Alpha/Beta and Theta/Alpha ratios
- Spike rate (abnormal peak detection)
- Signal variance
- Hjorth parameters (activity, mobility, complexity)
- Sample entropy (signal complexity / irregularity)

**Total feature vector**: 13 features × 17 channels = **221 dimensions** per epoch.

The LSTM branch does **not** use hand-crafted features — it learns directly from the raw time-series signal.

---

## 6. Genetic Feature Engineering

Each patient gets a simulated 12-point genetic vector:

- **9 binary mutation flags** (one per gene: SCN1A, SCN8A, KCNQ2, SCN2A, KCNT1, DEPDC5, PCDH19, GRIN2A, GABRA1)
- **2 pLI scores** for SCN1A and SCN8A (how intolerant the gene is to damage)
- **1 Polygenic Risk Score** (standardized across the cohort)

These flags are sampled from realistic population carrier frequencies (~0.3% to 1.5% per gene), and the PRS is computed from GWAS effect sizes.

---

## 7. Model Details

### 7.1 EEG Branch — Bidirectional LSTM
- **Input**: Raw EEG window [1280 time-points × 17 channels]
- **Layers**: 2-layer BiLSTM (hidden size 128) + self-attention + fully connected layers
- **Parameters**: ~759,000
- **Training**: 50 epochs max, early stopping on validation AUC, class-weighted loss to handle imbalance

### 7.2 Genetic Branch — XGBoost
- **Input**: 12-dim genetic vector
- **Algorithm**: Gradient-boosted trees
- **Training**: GPU-accelerated, early stopping after 20 rounds of no improvement
- **Output**: Genetic seizure probability

### 7.3 Attention-Based Fusion Layer
- Combines the EEG embedding (64-dim) and genetic embedding (64-dim) into a 128-dim vector.
- An attention gate learns patient-specific weights for each branch.
- Final output: a single risk score P_final ∈ [0, 1].
- **Loss**: Binary cross-entropy + L1 regularization on attention weights to prevent over-reliance on one branch.

---

## 8. Risk Score & Alert System

| Level | Color | Risk Range | Clinical Meaning |
|-------|-------|------------|------------------|
| 1 | Green | 0.00 – 0.25 | Low risk — routine monitoring |
| 2 | Yellow | 0.25 – 0.50 | Moderate risk — increase monitoring |
| 3 | Orange | 0.50 – 0.75 | High risk — prepare for possible seizure |
| 4 | Red | 0.75 – 1.00 | Critical — seizure imminent; alert caregiver |

---

## 9. Synthetic Data Generation (CTGAN)

Because real patient data is limited, we trained a **CTGAN** to generate 1,000 realistic synthetic patient records. This is useful for augmenting training data and privacy-safe sharing.

- **Input to CTGAN**: 190 per-file EEG summaries (mean/std of features) combined with genetic profiles.
- **Biological constraints enforced after generation**:
  - Mutation flags clipped to {0, 1}
  - pLI scores fixed to known constants
  - PRS clipped to ±5 standard deviations
  - Band powers forced non-negative
  - Low genetic risk patients (no mutations + PRS < −1) forced to seizure-free

**Validation**: The synthetic data closely matches real data on seizure ratio (26.3% real vs. 29.2% synthetic) and mutation rates. Some statistical tests (KS) remain limited due to the small real sample size (190 rows), but mode collapse and scaling issues from earlier versions were fixed.

---

## 10. Presentation & Visualization

We built a lightweight, browser-based dashboard (`presentation/index.html`) to showcase results. It includes:

- **Side-by-side EEG comparisons**: Raw vs. preprocessed signals for 5 clinically relevant segments (normal, eye blink, muscle artifact, pre-ictal, ictal).
- **Genetic heatmap**: Mutation status across 8 patients × 9 genes.
- **PRS bar chart**: Per-patient polygenic risk scores.
- **CTGAN results**: Distribution histograms (real vs. synthetic), seizure label proportions, and reconstructed synthetic waveforms.

No frameworks required — pure HTML/CSS/JS, served locally with Python’s built-in HTTP server.

---

## 11. Cloud Training Setup

To train on a GPU machine (e.g., cloud VM with NVIDIA RTX 4050), we created a one-click pipeline in `cloud_training/`:

| Script | What it does |
|--------|--------------|
| `run_all.sh` | Master script — installs dependencies, downloads data, preprocesses, trains |
| `01_download_and_preprocess.py` | Selectively downloads only seizure + limited interictal files from PhysioNet, then runs preprocessing |
| `02_train_models.py` | Trains both BiLSTM and XGBoost, evaluates on held-out test patients, saves all model files |

**Smart behaviors**:
- Skips downloads if EDF files already exist.
- Skips preprocessing if `.npy` files already exist.
- Uses memory-mapped loading so large datasets don’t crash RAM.

**Expected training time** (RTX 4050):
- Data download: 20–40 min
- Preprocessing: 2–3 hours
- LSTM training: 1–2 hours
- XGBoost training: ~5 minutes
- **Total**: ~4–5 hours end-to-end.

---

## 12. Current Status & What’s Next

### Completed
- Full preprocessing pipeline for 8 patients (333k+ epochs)
- Feature extraction (221-dim vectors) and genetic profile generation
- CTGAN synthetic data generator with biological constraints
- Interactive presentation frontend
- Cloud training scripts with CUDA support
- LSTM and XGBoost architecture implementation

### In Progress / Upcoming
- Train and evaluate LSTM + XGBoost on the full 8-patient dataset
- Train the attention-based fusion layer using outputs from both branches
- Final evaluation against target metrics (AUROC > 0.90, sensitivity > 85%, specificity > 90%)
- Deployment-ready API and frontend integration

---

## 13. Target Performance

| Metric | Target |
|--------|--------|
| AUROC | > 0.90 |
| Sensitivity (at 10% false positive rate) | > 85% |
| Specificity | > 90% |
| False Prediction Rate | < 0.15 per hour |
| Prediction horizon | 30 minutes before onset |

---

## 14. File Structure (Simplified)

```
eeg-genetic-fusion/
├── src/                          # Core Python code
│   ├── data_pipeline/            # EEG preprocessing + genetic features
│   └── models/                   # LSTM, XGBoost, CTGAN, Fusion
├── scripts/                      # Helper scripts (download, preprocess)
├── cloud_training/               # One-click cloud GPU training
├── configs/config.yaml           # All hyperparameters
├── data/
│   ├── raw/chb-mit/              # EEG recordings (EDF files)
│   ├── raw/clinvar/              # Genetic variants
│   ├── raw/gnomad/               # Gene constraint scores
│   ├── raw/gwas/                 # GWAS SNPs
│   └── processed/                # Preprocessed .npy files + synthetic data
├── presentation/                 # Interactive HTML dashboard
└── models/                       # Saved model weights (generated after training)
```

---

## 15. Key Takeaways

- We are building a **multimodal seizure predictor** that fuses real-time EEG patterns with a patient’s genetic risk profile.
- The system handles the notoriously imbalanced nature of seizure data (9:1 normal vs. pre-seizure) through weighted training and careful labelling.
- Because CHB-MIT lacks genetic data, we simulated realistic profiles from public population databases — clearly flagged as simulated in all outputs.
- A fully automated cloud training pipeline was built so the entire workflow (download → preprocess → train → evaluate) can run on any GPU machine with a single command.

---

*For detailed technical parameters, architecture diagrams, and implementation specifics, refer to the companion document:* **`TECHNICAL_REFERENCE.md`**.
