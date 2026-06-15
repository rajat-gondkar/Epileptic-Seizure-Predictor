# Progress So Far — EEG-Genetic Fusion for Seizure Prediction

## Project Goal
Build a multimodal AI system that fuses real-time EEG brain signals with patient-specific genetic biomarkers to predict epileptic seizures before they happen. The system has two branches:
- **EEG Branch**: BiLSTM+Attention model that processes raw EEG time-series
- **Genetic Branch**: XGBoost model that processes 16-dimensional genetic feature vectors
- **Fusion Layer**: Combines both modalities for final seizure prediction

---

## What Was Done (Chronological)

### Phase 1: Old Pipeline Cleanup

**Deleted `cloud_training/` directory** — This was the original STFT-CNN-BiLSTM approach which was inefficient and poorly structured. Removed:
- `src/models/lstm_eeg.py`
- `src/models/xgboost_genetic.py`
- `src/models/ctgan_synthetic.py`
- `src/data_pipeline/eeg_preprocessing.py`
- `src/data_pipeline/data_loader.py`
- `scripts/selective_download_and_preprocess.py`
- `scripts/run_acquisition.py`

**Why**: The old CTGAN implementation (`src/models/ctgan_synthetic.py`) had major issues:
- Only 190 real rows used for training
- Generated 1000 synthetic samples (too few)
- Poor KS test pass rate: 12.9%
- High mean correlation difference: 0.358
- Only used log10+StandardScaler preprocessing
- Poor correlation preservation

### Phase 2: Understanding the Data Pipeline

Reviewed `seizure_prediction/setup_chbmit.py` to understand how CHB-MIT data is organized:
- EDF files stored at: `Data/CHB-MIT/chbXX/chbXX_YY.edf`
- Annotation files at: `Data/CHB-MIT/chbXX/chbXX_YY.edf.annotation.txt`
- CSV manifests at: `DataCSVs/CHB-MIT/all_patients_train.csv` (279 files)
- Test CSV at: `DataCSVs/CHB-MIT/all_patients_test.csv` (76 files)
- 11 patients: chb01, chb02, chb03, chb04, chb07, chb09, chb11, chb14, chb15, chb18, chb24
- Window-level distribution: Interictal 96.0%, Preictal 3.8%, Ictal 0.2%
- 53,858 total windows across all patients

Reviewed `seizure_prediction/libDataIO.py` for EDF reading:
- Uses `pyedflib` for reading EDF files
- `fnReadEDFUsingPyEDFLib()` reads channels, sampling rate, data
- `fnReadCHBMITAnnoTxt()` reads annotation files for seizure times
- `fnBreakCHBMITSegment()` segments EEG into interictal/preictal/ictal zones
- 3-zone model: interictal (0) → preictal (1) → gap (0) → ictal (2)
- Preictal window: 30 min, Prediction horizon: 5 min

Reviewed `seizure_prediction/libCHBMITDataset.py` for preprocessing:
- Lazy-loading PyTorch Dataset
- Bandpass filter: 0.5-70 Hz
- Resample to 128 Hz
- Z-score normalization per channel
- 19 common EEG channels

### Phase 3: Genetic Data Review

**ClinVar** (`data/raw/clinvar/epilepsy_variants.csv`):
- 13,331 pathogenic/likely pathogenic variants
- Genes covered: SCN1A (4975), KCNQ2 (1839), SCN2A (1692), PCDH19 (1034), SCN8A (1019), GRIN2A (1009), DEPDC5 (975), KCNT1 (488), GABRA1 (300)

**gnomAD** (`data/raw/gnomad/pli_scores.csv`):
- pLI (probability of loss-of-function intolerance) scores for 9 epilepsy genes
- High pLI (>0.9): SCN1A (1.0), SCN8A (1.0), SCN2A (1.0), KCNQ2 (0.99998), GRIN2A (0.99998), PCDH19 (0.99975), GABRA1 (0.91489)
- Low pLI: DEPDC5 (0.1165), KCNT1 (2.76e-05)

**GWAS** (`data/raw/gwas/epilepsy_snps.csv`):
- 15 SNPs with odds ratios and risk allele frequencies
- Strongest effects: rs11890028 (SCN1A, OR=1.33), rs72823592 (KCNT1, OR=1.42)
- Risk allele frequencies range: 0.08 to 0.47

### Phase 4: Existing XGBoost Model Review

Reviewed `models/xgboost_genetic/`:
- Already trained XGBoost classifier
- AUC = 0.739, AUC-PR = 0.869
- Trained on 5000 synthetic patients (16-dim vectors)
- Uses `scripts/generate_synthetic_genetic_patients.py` for data generation

### Phase 5: New CTGAN Script Creation

Created `seizure_prediction/generate_ctgan_data.py` — completely rewritten CTGAN pipeline.

**Key improvements over old version**:
1. Processes all 279 EDF files (old: only 190 rows)
2. Generates 5000 synthetic samples (old: only 1000)
3. Uses compact ~20 features instead of 48 per-channel features
4. Reads genetic data from `data/raw/` folder
5. Includes biological constraints
6. Proper validation with KS tests and correlation analysis

**Pipeline steps**:
1. Load EDF file list from CSV
2. Extract EEG features from each EDF using pyedflib
3. Parse annotation files for seizure labels
4. Build 16-dim genetic profiles per patient
5. Augment real data for CTGAN training
6. Train CTGAN with SDV library
7. Generate synthetic samples
8. Validate quality metrics

### Phase 6: Bug Fixes

1. **NumPy 2.0+ compatibility**: `np.trapz` removed in NumPy 2.0+, added alias `np.trapz = np.trapezoid`

2. **JSON serialization error**: `numpy.bool_` not JSON serializable, fixed with `bool(p_value > 0.05)`

3. **Only 55 files extracted**: Code had `edf_paths[:5]` limiting to 5 files per patient, removed the limit

4. **48 features too many**: Reduced to ~20 compact features by averaging across channels instead of per-channel features

5. **CTGAN overfitting**: Reduced network size (256→128), batch size (500→100), increased epochs (300→500)

### Phase 7: CTGAN v2 — Separate EEG and Genetic

**Problem**: CTGAN treated genetic features (binary flags, bounded ratios) as continuous, producing negative mutations, negative ratios, and varying pLI scores.

**Solution**: Separate CTGAN (EEG) from direct sampling (genetic).

**Result**: 
- KS Pass Rate: 0% — CTGAN still couldn't learn EEG distributions
- Genetic features were perfect
- But cross-modal relationships destroyed (random pairing)

### Phase 8: CTGAN v3 — Joint Training

**Approach**: Train CTGAN on COMBINED EEG+genetic data (37 columns) to preserve cross-modal relationships. Clip bounded features after generation.

**Final Results** (from `synthetic/validation_report.json`):

| Metric | Value | Verdict |
|--------|-------|---------|
| KS Pass Rate | 31.6% (12/38) | Partial |
| Genetic features | 12/12 pass | Perfect |
| EEG features | 0/19 pass | Bad |
| Sanity checks | All pass | Perfect |
| Correlation diff | 0.235 | High |
| Correlation corr | 0.453 | Low |

**What's working**:
- All genetic features pass KS tests
- No negative mutations, proper bounds
- pLI scores constant at 1.0
- Binary features are binary

**What's NOT working**:
- EEG features have KS stats 0.13-0.38 (should be <0.05)
- Only 279 real samples → CTGAN can't learn complex EEG distributions
- Correlation structure not preserved well

**Decision**: Use synthetic data as **validation set only**, not for fusion training.

---

## Current Project Structure

```
eeg-genetic-fusion/
├── seizure_prediction/                    # NEW PIPELINE (canonical)
│   ├── generate_ctgan_data.py            # NEW: CTGAN synthetic data (v3)
│   ├── setup_chbmit.py                   # CHB-MIT data setup
│   ├── libCHBMITDataset.py              # PyTorch Dataset (lazy loading)
│   ├── libDataIO.py                      # EDF reading, annotation parsing
│   ├── libModelLSTM.py                   # BiLSTM+Attention model
│   ├── libUtils.py                       # Utility functions
│   ├── scrTrainLSTM.py                   # LSTM training script
│   ├── scrTestLSTM.py                    # LSTM testing script
│   ├── Data/CHB-MIT/                     # Raw EDF files (on cloud PC)
│   ├── DataCSVs/CHB-MIT/                # CSV file lists
│   │   ├── all_patients_train.csv       # 279 training EDF paths
│   │   └── all_patients_test.csv        # 76 test EDF paths
│   ├── SavedModels/                      # Trained LSTM models
│   └── Results/                          # Test results
│
├── src/data_pipeline/
│   ├── genetic_feature_engineering.py    # 16-dim vector construction
│   └── prs_computation.py               # Polygenic Risk Score calculation
│
├── scripts/
│   ├── generate_synthetic_genetic_patients.py  # 5000 synthetic patients
│   ├── acquire_genetic_data.py                 # ClinVar/gnomAD/GWAS download
│   └── select_features.py                      # Feature selection
│
├── data/raw/
│   ├── clinvar/
│   │   ├── epilepsy_variants.csv         # 13,331 pathogenic variants
│   │   └── variant_summary.txt.gz       # Raw ClinVar data
│   ├── gnomad/
│   │   ├── pli_scores.csv               # 9 genes pLI scores
│   │   └── gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz
│   └── gwas/
│       └── epilepsy_snps.csv            # 15 GWAS SNPs with effect sizes
│
├── data/processed/
│   ├── eeg_features/                     # Extracted EEG features
│   ├── genetic_vectors/
│   │   └── genetic_training_cohort.csv   # 5000 synthetic genetic patients
│   └── synthetic/
│       ├── real_eeg_genetic_features.csv # Real EEG+genetic (279 rows)
│       ├── synthetic_eeg_genetic_data.csv # CTGAN synthetic (5000 rows)
│       └── validation_report.json        # Quality metrics
│
├── synthetic/                            # Copy of results for review
│   ├── real_eeg_genetic_features.csv
│   ├── synthetic_eeg_genetic_data.csv
│   └── validation_report.json
│
├── models/
│   └── xgboost_genetic/                  # Trained XGBoost (AUC=0.739)
│
├── configs/
│   └── config.yaml                       # Master configuration
│
├── requirements_full.txt                 # Complete dependencies
├── progress_so_far.md                    # This file
└── tobedone.md                           # Roadmap
```

---

## 16-Dimensional Genetic Vector Layout

```
Index  Feature                      Description
-----  -------------------------    --------------------------------
[0]    SCN1A_mutation                1.0 × binary flag (weight=1.00)
[1]    SCN8A_mutation                1.0 × binary flag (weight=0.85)
[2]    KCNQ2_mutation                1.0 × binary flag (weight=0.95)
[3]    SCN2A_mutation                1.0 × binary flag (weight=0.90)
[4]    KCNT1_mutation                1.0 × binary flag (weight=0.80)
[5]    DEPDC5_mutation               1.0 × binary flag (weight=0.65)
[6]    PCDH19_mutation               1.0 × binary flag (weight=0.40)
[7]    GRIN2A_mutation               1.0 × binary flag (weight=0.55)
[8]    GABRA1_mutation               1.0 × binary flag (weight=0.50)
[9]    SCN1A_pLI                     pLI score (1.0 for SCN1A)
[10]   SCN8A_pLI                     pLI score (1.0 for SCN8A)
[11]   PRS                           Polygenic Risk Score (standardized)
[12]   mutation_burden               Sum of weighted mutations
[13]   ion_channel_burden            Sum for ion channel genes only
[14]   tier1_flag                    1 if any Tier-1 gene carrier
[15]   SCN1A_severity_proxy          pLI × mutation flag
```

---

## Gene Risk Weights (Literature-Backed)

| Gene    | Risk Weight | Carrier Freq | pLI     | Role |
|---------|-------------|--------------|---------|------|
| SCN1A   | 1.00        | 10%          | 1.000   | Sodium channel, most common epilepsy gene |
| KCNQ2   | 0.95        | 8%           | 0.99998 | Potassium channel, neonatal seizures |
| SCN2A   | 0.90        | 7%           | 1.000   | Sodium channel, DEE |
| SCN8A   | 0.85        | 6%           | 1.000   | Sodium channel, DEE |
| KCNT1   | 0.80        | 5%           | 0.00003 | Potassium channel, focal epilepsy |
| DEPDC5  | 0.65        | 5%           | 0.117   | mTOR pathway, focal epilepsy |
| GRIN2A  | 0.55        | 4%           | 0.99998 | Glutamate receptor, focal epilepsy |
| GABRA1  | 0.50        | 3%           | 0.915   | GABA receptor, absence epilepsy |
| PCDH19  | 0.40        | 4%           | 0.99975 | Protocadherin, DEE (females only) |

---

## Cloud PC Setup Commands

```bash
cd /workspace/eeg-genetic-fusion
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install numpy scipy pandas pyyaml pyedflib wfdb scikit-learn matplotlib tqdm psutil pytz sdv ctgan torch
python seizure_prediction/generate_ctgan_data.py --n-synthetic 5000 --seed 42
```

---

## CTGAN Synthetic Data — Final Status

**Purpose**: Validation set only (not for fusion training)

**What it's good for**:
- Checking models don't overfit to training data quirks
- Testing inference pipeline end-to-end
- Quick iteration without touching held-out real test set

**What it's NOT good for**:
- Fusion layer training (EEG distributions don't match real data)
- Replacing real test set (76 EDF files in `all_patients_test.csv`)

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `seizure_prediction/generate_ctgan_data.py` | CTGAN synthetic data generation (v3) |
| `seizure_prediction/libDataIO.py` | EDF reading, annotation parsing |
| `seizure_prediction/libModelLSTM.py` | BiLSTM+Attention model |
| `src/data_pipeline/genetic_feature_engineering.py` | 16-dim vector construction |
| `scripts/generate_synthetic_genetic_patients.py` | Synthetic genetic patient generation |
| `data/raw/gnomad/pli_scores.csv` | pLI scores for 9 genes |
| `data/raw/gwas/epilepsy_snps.csv` | GWAS SNP weights for PRS |
| `configs/config.yaml` | All project configuration |

---

*Last updated: June 2026 — CTGAN v3 Complete (Validation Set Only)*
