# Technical Reference Document
## AI-Based EEG + Genetic Marker Fusion for Epileptic Seizure Prediction

**Project**: RVCE IDP  
**Team**: Rajat G A, Srijeeta Ghosh, Aamir Ibrahim, Misab Abdul Raheem  
**Guide**: Prof. Deepika P

---

## 1. System Architecture

The system is a **multimodal late-fusion neural architecture** combining two independent branches:

- **EEG Branch**: Bidirectional LSTM with self-attention, trained on raw EEG time-series
- **Genetic Branch**: XGBoost gradient-boosted classifier, trained on a 12-dimensional genetic feature vector
- **Fusion Layer**: Attention-gated late fusion that dynamically weights both branches per patient to produce a final personalised seizure risk score

The output is a continuous risk score P_final ∈ [0, 1] mapped to a 4-level clinical alert system.

---

## 2. Datasets

### 2.1 CHB-MIT Scalp EEG Database (Primary EEG Dataset)

- **Source**: PhysioNet — https://physionet.org/content/chbmit/1.0.0/
- **Format**: European Data Format (EDF)
- **Patients used**: chb01, chb03, chb05 (full download) + chb06, chb08, chb10, chb16, chb20 (selective download)
- **Sampling rate**: 256 Hz
- **Recording duration per file**: 1 hour (3600 seconds)
- **Total EDF files downloaded**: 190 (chb01: 42, chb03: 38, chb05: 39, chb06: 15, chb08: 13, chb10: 15, chb16: 14, chb20: 14)
- **Channel montage**: Bipolar 10-20 system (17 channels after deduplication)
- **Annotated EEG files with seizures**: 59 files across 8 patients
  - chb01: 7 seizure files
  - chb03: 7 seizure files
  - chb05: 5 seizure files

**Seizure annotations (from per-patient summary files):**

| Patient | File | Seizure Start (s) | Seizure End (s) | Duration (s) |
|---------|------|--------------------|-----------------|--------------|
| chb01 | chb01_03.edf | 2996 | 3036 | 40 |
| chb01 | chb01_04.edf | 1467 | 1494 | 27 |
| chb01 | chb01_15.edf | 1732 | 1772 | 40 |
| chb01 | chb01_16.edf | 1015 | 1066 | 51 |
| chb01 | chb01_18.edf | 1720 | 1810 | 90 |
| chb01 | chb01_21.edf | 327 | 420 | 93 |
| chb01 | chb01_26.edf | 1862 | 1963 | 101 |
| chb03 | chb03_01.edf | 362 | 414 | 52 |
| chb03 | chb03_02.edf | 731 | 796 | 65 |
| chb03 | chb03_03.edf | 432 | 501 | 69 |
| chb03 | chb03_04.edf | 2162 | 2214 | 52 |
| chb03 | chb03_34.edf | 1982 | 2029 | 47 |
| chb03 | chb03_35.edf | 2592 | 2656 | 64 |
| chb03 | chb03_36.edf | 1725 | 1778 | 53 |
| chb05 | chb05_06.edf | 417 | 532 | 115 |
| chb05 | chb05_13.edf | 1086 | 1196 | 110 |
| chb05 | chb05_16.edf | 2317 | 2413 | 96 |
| chb05 | chb05_17.edf | 2451 | 2571 | 120 |
| chb05 | chb05_22.edf | 2348 | 2465 | 117 |

### 2.2 ClinVar (Genetic Variant Database)

- **Source**: NCBI — https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz
- **Download size**: 373 MB (compressed)
- **Filter applied**: 9 epilepsy-relevant genes, Pathogenic or Likely pathogenic classification only
- **Output**: `data/raw/clinvar/epilepsy_variants.csv` — 13,331 variants

**Pathogenic variant counts per gene:**

| Gene | Pathogenic Variants |
|------|-------------------|
| SCN1A | 4,975 |
| KCNQ2 | 1,839 |
| SCN2A | 1,692 |
| PCDH19 | 1,034 |
| SCN8A | 1,019 |
| GRIN2A | 1,009 |
| DEPDC5 | 975 |
| KCNT1 | 488 |
| GABRA1 | 300 |
| **Total** | **13,331** |

### 2.3 gnomAD Gene Constraint (pLI Scores)

- **Source**: gs://gcp-public-data--gnomad/release/2.1.1/constraint/gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz
- **Metric used**: pLI (probability of loss-of-function intolerance) — values close to 1.0 indicate the gene is highly intolerant to protein-truncating variants

**pLI scores for target genes (from gnomAD v2.1.1):**

| Gene | pLI | o/e LoF | o/e LoF upper CI | Observed LoF | Expected LoF |
|------|-----|---------|-------------------|--------------|--------------|
| SCN1A | 1.00000 | 0.02260 | 0.071 | 2 | 88.487 |
| SCN2A | 1.00000 | 0.06049 | 0.127 | 5 | 82.663 |
| SCN8A | 1.00000 | 0.06195 | 0.130 | 5 | 80.704 |
| KCNQ2 | 0.99998 | 0.05028 | 0.158 | 2 | 39.774 |
| GRIN2A | 0.99998 | 0.08196 | 0.188 | 4 | 48.805 |
| PCDH19 | 0.99975 | 0.00000 | 0.126 | 0 | 23.729 |
| GABRA1 | 0.91489 | 0.14198 | 0.367 | 3 | 21.129 |
| DEPDC5 | 0.11650 | 0.23539 | 0.331 | 24 | 101.960 |
| KCNT1 | 0.00003 | 0.32431 | 0.471 | 20 | 61.670 |

### 2.4 GWAS Catalog (Epilepsy SNP Weights for PRS)

- **Source**: Published epilepsy GWAS meta-analyses (literature-curated, GWAS Catalog API unavailable)
- **SNPs used**: 15 genome-wide significant epilepsy-associated SNPs
- **Effect size measure**: Odds Ratio (OR)

**Epilepsy GWAS SNPs used for Polygenic Risk Score:**

| SNP ID | Chr | Gene | OR | p-value | Trait | RAF |
|--------|-----|------|----|---------|-------|-----|
| rs6732655 | 2 | SCN1A | 1.25 | 2.4×10⁻¹⁵ | generalised epilepsy | 0.37 |
| rs55670523 | 2 | SCN1A | 1.31 | 1.2×10⁻¹² | generalised epilepsy | 0.28 |
| rs13020210 | 2 | SCN1A | 1.27 | 8.9×10⁻¹³ | epilepsy | 0.29 |
| rs11890028 | 2 | SCN1A | 1.33 | 1.5×10⁻¹⁴ | febrile seizure | 0.31 |
| rs2947349 | 2 | SCN1A/SCN2A | 1.22 | 5.6×10⁻¹¹ | epilepsy | 0.35 |
| rs1556832 | 2 | SCN2A | 1.18 | 3.1×10⁻¹⁰ | focal epilepsy | 0.42 |
| rs12987787 | 2 | SCN2A | 1.16 | 2.3×10⁻⁸ | generalised epilepsy | 0.41 |
| rs72823592 | 9 | KCNT1 | 1.42 | 4.5×10⁻⁹ | focal epilepsy | 0.12 |
| rs1034114 | 12 | SCN8A | 1.19 | 7.2×10⁻⁹ | epilepsy | 0.39 |
| rs7163093 | 15 | CHRNA7 | 1.12 | 9.3×10⁻⁹ | generalised epilepsy | 0.44 |
| rs117503424 | 16 | GRIN2A | 1.35 | 2.1×10⁻⁸ | focal epilepsy | 0.08 |
| rs2292096 | 5 | GABRA1 | 1.28 | 3.7×10⁻¹⁰ | absence epilepsy | 0.22 |
| rs28498976 | 4 | PCDH7 | 1.15 | 1.8×10⁻⁸ | generalised epilepsy | 0.47 |
| rs4839797 | 20 | KCNQ2 | 1.24 | 6.8×10⁻⁹ | neonatal seizure | 0.18 |
| rs2241085 | 22 | DEPDC5 | 1.17 | 4.2×10⁻⁸ | focal epilepsy | 0.33 |

*RAF = Risk Allele Frequency*

---

## 3. EEG Preprocessing Pipeline

**Library**: MNE-Python v1.12.0  
**Implementation**: `src/data_pipeline/eeg_preprocessing.py`

### 3.1 Channel Selection

- CHB-MIT EDFs contain 23 raw channels including duplicates and non-EEG channels
- Duplicate handling: MNE renames duplicate channels (e.g. `T8-P8` → `T8-P8-0`, `T8-P8-1`); the second occurrence is dropped
- 17–18 bipolar channels matching the standard 10-20 system are selected:

```
FP1-F7, F7-T7, T7-P7, P7-O1,
FP1-F3, F3-C3, C3-P3, P3-O1,
FP2-F4, F4-C4, C4-P4, P4-O2,
FP2-F8, F8-T8, T8-P8, P8-O2,
FZ-CZ, CZ-PZ
```

*Note: chb01 and chb03 yield 17 channels (T8-P8 duplicate dropped), giving feature vectors of size 17×13 = 221.*

### 3.2 Signal Filtering

| Step | Method | Parameters |
|------|--------|-----------|
| Bandpass | IIR Butterworth (zero-phase, via MNE filtfilt) | Order 4, 0.5–70 Hz |
| Notch | IIR notch | 60 Hz (US power line frequency) |
| Re-reference | Common Average Reference (CAR) | All selected channels averaged |

### 3.3 Artifact Rejection

- **Method**: Peak-to-peak (PTP) amplitude threshold per channel
- **Threshold**: 500 µV (5×10⁻⁴ V)
- **Decision**: Epoch rejected if any single channel exceeds threshold
- *Rationale for 500 µV*: The commonly cited 150 µV threshold is appropriate for clinical settings with artifact-free recordings. CHB-MIT data, acquired during natural conditions with movement, requires a more permissive threshold. 500 µV is standard in published seizure prediction literature for this dataset.

### 3.4 Epoching

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epoch length | 5 seconds | Standard for seizure prediction; captures slow oscillations |
| Epoch stride | 1 second | Overlapping windows for temporal continuity |
| Samples per epoch | 1280 (at 256 Hz × 5 s) | Covers all frequency bands of interest |
| Preictal window | 30 minutes before seizure onset | Clinical seizure prediction horizon (SPH) |
| Postictal exclusion | 30 minutes after seizure offset | Postictal state distinct from resting baseline |

### 3.5 Epoch Labelling

| Label | Class | Condition |
|-------|-------|-----------|
| 0 | Interictal | No seizure within ±30 min; passes artifact check |
| 1 | Preictal | Within 30 min prior to seizure onset; passes artifact check |
| -1 | Excluded | Ictal period OR postictal window OR artifact rejected |

### 3.6 Processed Dataset Statistics

| Patient | Total Epochs | Interictal | Preictal | Download type |
|---------|-------------|------------|----------|---------------|
| chb01 | 95,699 | 86,837 | 8,862 | Full (42 files) |
| chb03 | 87,780 | 80,519 | 7,261 | Full (38 files) |
| chb05 | 49,919 | 47,107 | 2,812 | Full (39 files) |
| chb06 | 17,472 | 16,216 | 1,256 | Selective (15 files) |
| chb08 | 14,662 | 12,883 | 1,779 | Selective (13 files) |
| chb10 | 16,642 | 14,839 | 1,803 | Selective (15 files) |
| chb16 | 17,547 | 15,261 | 2,286 | Selective (13/14 files†) |
| chb20 | 33,894 | 27,385 | 6,509 | Selective (14 files) |
| **Total** | **333,615** | **301,047** | **32,568** | — |

†chb16_18.edf (18 channels) dropped by channel harmonisation; all others have 17 channels.

**Overall class imbalance ratio (interictal : preictal): ~9.2 : 1**  
Positive class weighting applied during model training to compensate.

**Channel harmonisation fix**: `process_patient()` now detects mixed channel counts across files within a patient, keeps the dominant count (17), and drops outlier-channel-count files before concatenation. Triggered on chb16.

---

## 4. Feature Extraction

**Implementation**: `src/data_pipeline/eeg_preprocessing.py` — `EEGFeatureExtractor`

For each epoch, 13 features are extracted per channel:

| Index | Feature | Method |
|-------|---------|--------|
| 0 | Delta band power (0.5–4 Hz) | Welch PSD, trapezoidal integration |
| 1 | Theta band power (4–8 Hz) | Welch PSD, trapezoidal integration |
| 2 | Alpha band power (8–12 Hz) | Welch PSD, trapezoidal integration |
| 3 | Beta band power (12–30 Hz) | Welch PSD, trapezoidal integration |
| 4 | Gamma band power (30–70 Hz) | Welch PSD, trapezoidal integration |
| 5 | Alpha/Beta ratio | α-power / β-power |
| 6 | Theta/Alpha ratio | θ-power / α-power |
| 7 | Interictal spike rate (peaks/sec) | Peaks > (mean + 3×SD), scipy.find_peaks |
| 8 | Signal variance | np.var |
| 9 | Hjorth Activity | Variance of signal |
| 10 | Hjorth Mobility | √(var(dx/dt) / var(x)) |
| 11 | Hjorth Complexity | Mobility(dx/dt) / Mobility(x) |
| 12 | Sample Entropy | Template matching, m=2, r=0.2×SD, downsampled to 256 pts |

**Welch PSD parameters**: nperseg = min(epoch_length, 256) = 256 samples; noverlap = 128 samples

**Total feature vector dimension**: 13 features × 17 channels = **221 dimensions** (for patients with 17 channels after deduplication; 234 for 18-channel patients)

---

## 5. Genetic Feature Engineering

**Implementation**: `src/data_pipeline/genetic_feature_engineering.py`, `src/data_pipeline/prs_computation.py`

### 5.1 12-Dimensional Genetic Feature Vector

| Index | Feature | Type | Source |
|-------|---------|------|--------|
| 0 | SCN1A_mutation_flag | Binary {0,1} | ClinVar |
| 1 | SCN8A_mutation_flag | Binary {0,1} | ClinVar |
| 2 | KCNQ2_mutation_flag | Binary {0,1} | ClinVar |
| 3 | SCN2A_mutation_flag | Binary {0,1} | ClinVar |
| 4 | KCNT1_mutation_flag | Binary {0,1} | ClinVar |
| 5 | DEPDC5_mutation_flag | Binary {0,1} | ClinVar |
| 6 | PCDH19_mutation_flag | Binary {0,1} | ClinVar |
| 7 | GRIN2A_mutation_flag | Binary {0,1} | ClinVar |
| 8 | GABRA1_mutation_flag | Binary {0,1} | ClinVar |
| 9 | SCN1A_pLI_score | Continuous [0,1] | gnomAD v2.1.1 |
| 10 | SCN8A_pLI_score | Continuous [0,1] | gnomAD v2.1.1 |
| 11 | Polygenic Risk Score | Continuous (standardised) | GWAS Catalog (15 SNPs) |

### 5.2 Simulated Genetic Profiles (CHB-MIT patients)

Since CHB-MIT provides no real genetic data, profiles are simulated using:

**Mutation flags** — sampled from population-level carrier frequencies:

| Gene | Carrier Frequency Used |
|------|----------------------|
| SCN1A | 1.5% |
| SCN8A | 0.8% |
| KCNQ2 | 1.0% |
| SCN2A | 0.7% |
| KCNT1 | 0.4% |
| DEPDC5 | 0.5% |
| PCDH19 | 0.6% |
| GRIN2A | 0.5% |
| GABRA1 | 0.3% |

**pLI scores** — fixed constants from gnomAD (not simulated; gene-level population constraint)

**Polygenic Risk Score (PRS)**:
- Formula: PRS = Σᵢ βᵢ × gᵢ, where βᵢ = ln(ORᵢ) and gᵢ ∈ {0, 1, 2} is the genotype dosage
- Genotypes sampled from Hardy-Weinberg equilibrium using GWAS risk allele frequencies
- Cohort-level standardisation applied: PRS_std = (PRS − μ) / σ
- Simulated PRS distribution (10,000 simulations): mean=0.0, SD=1.0, range=[−3.14, +4.11]

**Simulated genetic profiles (8 patients used for training):**

| Patient | SCN1A | SCN8A | KCNQ2 | SCN2A | KCNT1 | DEPDC5 | PCDH19 | GRIN2A | GABRA1 | SCN1A_pLI | SCN8A_pLI | PRS |
|---------|-------|-------|-------|-------|-------|--------|--------|--------|--------|-----------|-----------|-----|
| chb01 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 | +1.023 |
| chb03 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 | +0.339 |
| chb05 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 | +0.092 |
| chb06 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 | +1.006 |
| chb08 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 | — |
| chb10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 | — |
| chb16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 | — |
| chb20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 | — |

---

## 6. Model Architecture

### 6.1 EEG Branch — STFT-CNN-Bidirectional LSTM

| Component | Specification |
|-----------|--------------|
| Input | STFT magnitude spectrogram [batch, C=17, F=70, T_stft=21] |
| **STFT parameters** | nperseg=256, noverlap=192, fs=256 Hz → 70 freq bins (1–70 Hz), ~21 time frames |
| Preprocessing | Log-scaled magnitude: `log(|STFT| + 1e-8)` |
| **CNN Front-End** | **3× Conv2D** (kernel=(3,3), stride=(2,1)) with BatchNorm + ReLU |
|   Conv1 | 17 → 32 channels, freq 70 → **35**, time preserved |
|   Conv2 | 32 → 64 channels, freq 35 → **18**, time preserved |
|   Conv3 | 64 → **128** channels, freq 18 → **9**, time preserved |
| Pool | AdaptiveAvgPool2d((1, None)) → [batch, 128, 1, T] → squeeze → [batch, T, 128] |
| Layer 1 | Bidirectional LSTM; hidden_size=128; **num_layers=1**; **dropout=0.2** |
| Effective hidden dim | 256 (bidirectional) |
| Layer 2 | Self-attention over LSTM output timesteps |
| Layer 3 | Global average pooling |
| Layer 4 | FC(256→64) + ReLU + **Dropout(0.2)** |
| Output | FC(64→1) + Sigmoid → P_seizure ∈ [0,1] |

**Training hyperparameters (final, v4):**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimiser | Adam | — |
| **Learning rate** | **1×10⁻⁴** | Reduced from 1e-3; standard for LSTM on long sequences |
| Weight decay | 1×10⁻⁴ | — |
| **Batch size** | **64** | Better gradient estimates with 1280-timestep sequences |
| Max epochs | 50 | — |
| **Early stopping patience** | **15** (on validation AUC) | More patience needed with lower LR |
| **LR warmup** | **Linear, 5 epochs** | Prevents early-epoch collapse before gradients stabilize |
| LR scheduler (post-warmup) | ReduceLROnPlateau (patience=5, factor=0.5) | — |
| **Loss** | **`FocalLoss(gamma=2.0, alpha=0.75)`** | Replaces BCE + pos_weight; down-weights easy negatives automatically |
| Sampler | **None** (natural batches) | Focal Loss handles imbalance internally; no sampler needed |
| Bias init | `_init_final_bias(pos_ratio)` | Final layer bias initialized so initial prediction ≈ dataset positive rate (~8 %), breaking symmetry |
| **Gradient clipping** | **Max norm = 1.0** | Tighter clip prevents spikes; grad norms at batch 0 are already small (~0.6) |
| Augmentation | Gaussian noise σ=0.01; channel dropout p=0.1 | Active in training loop |
| Validation strategy | Patient-level leave-one-out cross-validation | Train on first 5 patients, val on 1, test on last 2 |
| Data loading | Memory-mapped `.npy` via `SequenceDataset` | Streams batches from disk without loading all ~27 GB into RAM |
| Checkpoints saved | `lstm_best.pt` + `lstm_latest.pt` | Best (by val AUC) and most recent epoch only |
| Live monitor | Terminal table: Epoch / Train Loss / Val Loss / Val AUC / Sens / Spec / LR / Time / ETA / Status | Updated every epoch |
| First-batch diagnostics | Prints pred mean/std/min/max, grad norm, per-layer gradient norms | Debug only; confirms model is not stuck |

**XGBoost branch**:

| Parameter | Value |
|-----------|-------|
| Input | 221-dim EEG feature vector per epoch (XGBoost is currently trained on EEG features only; genetic fusion happens in the fusion layer) |
| n_estimators | 300 |
| max_depth | 4 |
| learning_rate | 0.05 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| **scale_pos_weight** | **9** | Matches actual ~11:1 class ratio |
| eval_metric | AUC |
| early_stopping_rounds | 20 |
| GPU | CUDA (`device='cuda'`) via histogram method |
| Output | P_genetic ∈ [0,1] |

**XGBoost 2.0+ API compatibility:**
XGBoost ≥ 2.0 removed `early_stopping_rounds` from `.fit()`. The training script uses a `try/except` fallback:
```python
try:
    model.fit(..., early_stopping_rounds=20, ...)
except TypeError:
    # XGBoost 2.0+ moved early stopping to callbacks
    callbacks = [xgb.callback.EarlyStopping(rounds=20, save_best=True)]
    model.fit(..., callbacks=callbacks, ...)
```

---

## 12. Cloud Training Pipeline

**Location**: `cloud_training/`  
**Target environment**: Ubuntu VM with NVIDIA RTX 4050 (6 GB VRAM), CUDA 11.8+, Python 3.10+  
**Entry point**: `bash cloud_training/run_all.sh`

### 12.1 Scripts

| Script | Purpose |
|--------|---------|
| `run_all.sh` | Master shell script; auto-detects `python3`/`python`, installs deps, runs steps 01 and 02, pipes all output to `models/training_log.txt` via `tee` |
| `01_download_and_preprocess.py` | Thin wrapper forwarding to `scripts/selective_download_and_preprocess.py` |
| `02_train_models.py` | Trains BiLSTM and XGBoost with live terminal status monitor |
| `requirements.txt` | All pip dependencies pinned for reproducibility |

### 12.2 Shell Script Features

**Python auto-detection:**
Ubuntu/Debian systems often lack a `python` command. `run_all.sh` now probes for `python3` first, then `python`, and fails gracefully with an error message if neither is found.

**Argument handling:**
| Flag | Behaviour |
|------|-----------|
| (none) | Full pipeline: deps → download → preprocess → train |
| `--skip-download` | Skips EDF downloads but still runs preprocessing if `.npy` files are missing |
| `--train-only` | Skips download AND preprocessing; jumps straight to Step 2 |
| `--quick-test` | 3 LSTM epochs, 2000 epochs/patient cap — smoke test |
| `--xgboost` | Skips LSTM training; runs only XGBoost (use with `--train-only`) |

**Pipeline progress tracker:**
The shell script prints `[1/3]`, `[2/3]`, `[3/3]` step headers with elapsed times and a final total duration.

**Log persistence:**
All stdout/stderr from the training script is tee'd to `models/training_log.txt` automatically. Structured metrics are also saved to `models/lstm_history.json`, `models/xgboost_metrics.json`, and `models/test_results.json`.

### 12.3 Model Training Specifications (Final)

**BiLSTM branch** (`02_train_models.py`):

| Parameter | Value | Notes |
|-----------|-------|-------|
| Architecture | **STFT-CNN-BiLSTM**; 3× Conv2D on spectrograms + 1-layer BiLSTM | STFT (70 freq × ~21 time frames) → 2D CNN (freq pool) → BiLSTM. Replaces raw-waveform 1D CNN. |
| Attention | Self-attention over all timesteps | — |
| Total parameters | ~759,000 | — |
| Device | CUDA auto-detect; CPU fallback | MPS explicitly avoided (Apple Silicon bottleneck) |
| Data loading | `SequenceDataset` with `mmap_mode='r'`, **`num_workers=0`** | Raw sequences streamed from disk; STFT computed on-the-fly in `__getitem__`. Workers disabled to prevent `/dev/shm` exhaustion on cloud VMs. |
| Batch size | **64** | Better gradient estimates with long sequences |
| Optimiser | Adam, **lr=1e-4**, weight_decay=1e-4 | Lower LR prevents overshooting |
| **Loss** | **`FocalLoss(gamma=2.0, alpha=0.75)`** | Replaces BCE + pos_weight; see §6.1 evolution |
| Sampler | **None** (natural batches, shuffle=True) | Focal Loss handles imbalance internally |
| Bias init | Final layer bias → `ln(pos_ratio / (1−pos_ratio))` | Breaks 0.5-symmetry trap |
| **LR warmup** | **Linear, 5 epochs** | Prevents early-epoch collapse |
| Early stopping | **Patience=15** on validation AUC | More patience needed with lower LR |
| LR scheduler (post-warmup) | ReduceLROnPlateau (patience=5, factor=0.5) | — |
| **Gradient clipping** | **Max norm = 1.0** | Tighter clip prevents gradient spikes |
| Augmentation | Gaussian noise σ=0.01; channel dropout p=0.1 | Active in training loop |
| Checkpoints | `lstm_best.pt` + `lstm_latest.pt` | Best by val AUC, and most recent epoch |
| Live monitor | Terminal table: Epoch / Train Loss / Val Loss / Val AUC / Sens / Spec / LR / Time / ETA / Status | Updated every epoch |
| First-batch diagnostics | Prints pred mean/std/min/max, grad norm, per-layer gradient norms | Debug only; confirms model is not stuck |

**XGBoost branch**:

| Parameter | Value |
|-----------|-------|
| Input | 221-dim feature vector per epoch |
| GPU | `device='cuda'` (CUDA-accelerated histogram method) |
| n_estimators | 300 max |
| Early stopping | 20 rounds (XGBoost 2.0+ uses `xgb.callback.EarlyStopping`) |
| eval_metric | AUC |
| scale_pos_weight | 9 |
| Output | `models/xgboost_model.pkl` |

### 12.4 Download Strategy

`scripts/selective_download_and_preprocess.py` downloads only seizure files + up to 8 interictal files per patient. It now supports `--skip-download` to use existing EDFs only (useful when copying the full project folder to another machine).

### 12.5 Outputs After Training

```
models/
├── lstm_best.pt              # Best LSTM checkpoint (by validation AUC)
├── lstm_latest.pt            # Last epoch checkpoint (survival net)
├── lstm_history.json         # Per-epoch loss, AUC, LR, time
├── xgboost_model.pkl         # Trained XGBoost model
├── xgboost_metrics.json      # Validation AUC/accuracy/precision/recall/F1
├── test_results.json         # Test-set evaluation (LSTM + XGBoost)
├── lstm_test_preds.npy       # LSTM predictions on held-out test patients
├── xgb_test_preds.npy        # XGBoost predictions on held-out test patients
└── training_log.txt          # Full stdout/stderr from the run
```

These outputs are the inputs for the Attention Fusion Layer.

---

## 13. Known Issues and Design Decisions

| Issue | Resolution |
|-------|-----------|
| CHB-MIT has no real genetic data | Profiles simulated from population carrier frequencies and GWAS SNPs — flagged clearly in all outputs |
| CTGAN KS pass rate low (12.9%) | Fundamental sample size limitation (190 rows). Mode collapse and band power precision issues from v1 are fixed. Documented as known limitation. |
| chb16_18.edf channel mismatch (18 ch vs 17) | Automatically dropped by channel harmonisation in `process_patient()` |
| MPS (Apple Silicon) bottleneck | Cloud training script hard-excludes MPS and uses CUDA only |
| Canvas hidden on page load | CTGAN synthetic canvas deferred with `requestAnimationFrame + setTimeout(50ms)` |
| Preprocessing looks similar to raw | Expected behaviour — pipeline is conservative by design. Fixed visualization using shared µV/pixel scale across both panels. |
| EEG downsampled in presentation JSON | 2:1 stride (256 Hz → 128 Hz effective) applied at export only, to reduce file size. Original data unchanged. |
| **LSTM trivial minimum trap** (loss stuck at 0.693, AUC=0.5000) | `WeightedRandomSampler(1/count)` + unweighted `BCEWithLogitsLoss` creates a symmetric minimum at p=0.5. Fixed by using **FocalLoss(gamma=2, alpha=0.75)** with natural batches + bias initialization to dataset positive rate. See §6.1 for evolution. |
| **Double class-imbalance correction** | Original code used both `WeightedRandomSampler` AND `pos_weight≈10` simultaneously. Fixed to use **Focal Loss alone** (no sampler, no pos_weight). |
| **VRAM exhaustion on RTX 4050 6 GB** | Batch size 64 + self-attention over 1280 timesteps ≈ 6–7 GB. Mitigated by using 1 LSTM layer instead of 2, plus memory-mapped `SequenceDataset`. |
| **DataLoader shared-memory crash** (`RuntimeError: unable to allocate shared memory`) | Default `num_workers=4` on CUDA causes `/dev/shm` exhaustion during multi-worker batch collation. Fixed by setting **`num_workers=0`** for train, validation, and test DataLoaders. |
| **XGBoost 2.0+ API break** | `early_stopping_rounds` and `use_label_encoder` were removed in different XGBoost versions. Fixed by inspecting `fit()` signature at runtime to choose the correct parameter (`early_stopping_rounds`, `callbacks`, or none), and by removing the deprecated `use_label_encoder` argument. |
| **`python` command missing on Ubuntu** | `run_all.sh` now auto-detects `python3` then `python`, and fails gracefully if neither exists. |
| **Augmentation defined but never applied** | `config.yaml` listed Gaussian noise and channel dropout, but the training loop never executed them. Fixed — augmentation is now active in the batch loop. |
| **Per-epoch checkpoints bloating disk** | Originally saved every epoch. Fixed to keep only `lstm_best.pt` and `lstm_latest.pt`. |
| **No persistent training log** | Terminal output was ephemeral. Fixed — `run_all.sh` now pipes all output to `models/training_log.txt` via `tee`. |
| **LSTM all-0/all-1 oscillation** (AUC≈0.5, Sens/Spec flip-flopping) | `pos_weight=3` was too weak for 11:1 imbalance; model oscillated between predicting all-negative and all-positive. Fixed by switching to **FocalLoss** and removing the sampler. |
| **LR 1e-3 too aggressive** | Caused overshooting before model found useful gradient signal. Fixed by reducing to **1e-4** and adding **5-epoch linear warmup**. |
| **Raw 1280-step LSTM unstable** | Feeding raw 1280 timesteps directly into BiLSTM causes gradient vanishing across the long sequence. Fixed by adding a **1D CNN front-end** (3 conv layers, kernel=5, stride=2) that reduces sequence length 1280 → 160 before the LSTM sees it. |
| **LSTM failing to learn generalizable features** (val AUC ~0.50–0.60, collapsing to 0.50) | Raw waveform input lacks the time-frequency structure that BiLSTMs need for EEG pattern recognition. Fixed by switching input to **STFT magnitude spectrograms** (70 freq bins × ~21 time frames) and replacing the 1D CNN with a **2D CNN** (Conv2d, stride=(2,1)) that compresses frequency while preserving temporal resolution for the LSTM. |
| **Gradient clip 5.0 too loose** | Allowed gradient spikes that destabilized training. Fixed by tightening to **1.0**. |

