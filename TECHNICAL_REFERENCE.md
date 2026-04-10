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
- **Patients used**: chb01, chb03, chb05 (subset of 24 total patients)
- **Sampling rate**: 256 Hz
- **Recording duration per file**: 1 hour (3600 seconds)
- **Total EDF files downloaded**: 119 (chb01: 42, chb03: 38, chb05: 39)
- **Channel montage**: Bipolar 10-20 system (17–18 channels per patient after deduplication)
- **Annotated EEG files with seizures**: 19 files across 3 patients
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

| Patient | Total Epochs | Interictal | Preictal | Feature Shape | Sequence Shape |
|---------|-------------|------------|----------|---------------|----------------|
| chb01 | 95,699 | 86,837 | 8,862 | (95699, 221) | (95699, 1280, 17) |
| chb03 | 87,780 | 80,519 | 7,261 | (87780, 221) | (87780, 1280, 17) |
| chb05 | 49,919 | 47,107 | 2,812 | (49919, 221) | (49919, 1280, 17) |
| **Total** | **233,398** | **214,463** | **18,935** | — | — |

**Overall class imbalance ratio (interictal : preictal): ~11.3 : 1**  
Positive class weighting will be applied during model training to compensate.

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

**Simulated genetic profiles (current 5 patients):**

| Patient | SCN1A | SCN8A | KCNQ2 | SCN2A | KCNT1 | DEPDC5 | PCDH19 | GRIN2A | GABRA1 | SCN1A_pLI | SCN8A_pLI | PRS |
|---------|-------|-------|-------|-------|-------|--------|--------|--------|--------|-----------|-----------|-----|
| chb01 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 | +1.023 |
| chb02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 | −1.669 |
| chb03 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 | +0.339 |
| chb04 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 | +0.215 |
| chb05 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 | +0.092 |

---

## 6. Model Architecture (Planned — Not Yet Trained)

### 6.1 EEG Branch — Bidirectional LSTM

| Component | Specification |
|-----------|--------------|
| Input | Raw EEG sequence [batch, T=1280, C=17] |
| Layer 1 | Bidirectional LSTM; hidden_size=128; num_layers=2; dropout=0.3 |
| Effective hidden dim | 256 (bidirectional) |
| Layer 2 | Self-attention over LSTM output timesteps |
| Layer 3 | Global average pooling |
| Layer 4 | FC(256→64) + ReLU + Dropout(0.3) |
| Output | FC(64→1) + Sigmoid → P_seizure ∈ [0,1] |

**Training hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Optimiser | Adam |
| Learning rate | 1×10⁻³ |
| Weight decay | 1×10⁻⁴ |
| Batch size | 64 |
| Max epochs | 50 |
| Early stopping patience | 10 (on validation AUC) |
| LR scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Loss | Binary Cross-Entropy with positive class weight |
| Positive class weight | n_interictal / n_preictal (≈10×) |
| Validation strategy | Patient-level leave-one-out cross-validation |
| Augmentation | Gaussian noise (σ=0.01); random channel zero-out (p=0.1) |

### 6.2 Genetic Branch — XGBoost

| Parameter | Value |
|-----------|-------|
| Input | 12-dim genetic feature vector |
| n_estimators | 300 |
| max_depth | 4 |
| learning_rate | 0.05 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| scale_pos_weight | 5 |
| eval_metric | AUC |
| early_stopping_rounds | 20 |
| Output | P_genetic ∈ [0,1] |

### 6.3 Attention-Based Fusion Layer

| Component | Specification |
|-----------|--------------|
| Input 1 | eeg_embedding [batch, 64] — penultimate LSTM output |
| Input 2 | genetic_vector [batch, 12] — raw genetic features |
| Projection | FC(12→64) + ReLU → genetic_embedding [batch, 64] |
| Concatenation | combined = [eeg_emb \| gen_emb] → [batch, 128] |
| Attention gate | FC(128→2) + Softmax → [α_eeg, α_genetic] (sums to 1) |
| Fusion | P_final = α_eeg × sigmoid(FC(eeg_emb)) + α_genetic × sigmoid(FC(gen_emb)) |
| Output | P_final ∈ [0,1] |

**Fusion training hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Epochs | 30 |
| Batch size | 32 |
| Learning rate | 1×10⁻³ |
| Loss | BCE + L1 regularisation on attention weights (λ=0.01) |
| LSTM/XGBoost | Frozen during fusion training |

### 6.4 CTGAN Synthetic Data Generator

| Parameter | Value |
|-----------|-------|
| Input | Real patient records (genetic features + EEG summary statistics) |
| Generator dimensions | (256, 256) |
| Discriminator dimensions | (256, 256) |
| Training epochs | 300 |
| Batch size | 500 |
| Synthetic samples | 1,000 |
| Discrete columns | 9 mutation flags + label |

**Post-generation biological constraints enforced:**
- Mutation flags clipped to {0, 1}
- pLI scores frozen to gnomAD values (not re-generated)
- PRS clipped to [−5, +5] SD
- EEG power values clipped to non-negative
- If all 9 mutation flags = 0 AND PRS < 0 → label forced to 'low_risk'
- If SCN1A=1 OR SCN8A=1 OR KCNQ2=1 → label must not be 'low_risk'

---

## 7. Risk Score and Alert System

| Alert Level | Label | Colour | P_final Range | Clinical Action |
|-------------|-------|--------|--------------|-----------------|
| 1 | Low Risk | Green | [0.00, 0.25) | Monitor normally |
| 2 | Moderate Risk | Yellow | [0.25, 0.50) | Increase monitoring frequency |
| 3 | High Risk | Orange | [0.50, 0.75) | Prepare for possible seizure |
| 4 | Critical | Red | [0.75, 1.00] | Seizure imminent — alert caregiver |

---

## 8. Evaluation Metrics (Target Performance)

| Metric | Target |
|--------|--------|
| AUROC | > 0.90 |
| Sensitivity at FPR = 0.10 | > 85% |
| Specificity | > 90% |
| False Prediction Rate | < 0.15 / hour |
| Seizure Prediction Horizon (SPH) | 30 minutes before onset |
| Mean prediction lead time | > 20 minutes |

**Baseline comparisons:**
- Unimodal LSTM (EEG only)
- Unimodal XGBoost (genetic only)
- Simple average fusion: (P_eeg + P_genetic) / 2
- **Proposed**: Attention-based fusion (expected best)

**Prior work benchmarks:**
- Tsiouris et al. 2018 (LSTM on CHB-MIT): 99.6% accuracy
- Zhu et al. 2024 (Multidimensional Transformer + RNN): 98.24% sensitivity, 97.27% specificity

---

## 9. Software Stack

| Component | Library / Version |
|-----------|-----------------|
| EEG processing | MNE-Python v1.12.0 |
| EDF reading | pyedflib v0.1.42 |
| EEG download | wfdb v4.3.1 |
| Deep learning | PyTorch ≥ 2.0 |
| Classical ML | XGBoost ≥ 1.7 |
| Synthetic data | CTGAN / SDV |
| Explainability | SHAP |
| Scientific computing | NumPy 2.4.4, SciPy 1.17.1, pandas 3.0.2 |
| API | FastAPI + Uvicorn |
| Database ORM | SQLAlchemy, psycopg2 (PostgreSQL 15) |
| Frontend (planned) | React 18, Vite, Recharts, Socket.io |
| Python | 3.12.12 |

---

## 10. File Outputs

| File | Location | Description |
|------|----------|-------------|
| `seizure_annotations.csv` | `data/raw/chb-mit/` | 19 seizures, 4 columns |
| `epilepsy_variants.csv` | `data/raw/clinvar/` | 13,331 pathogenic variants |
| `pli_scores.csv` | `data/raw/gnomad/` | pLI + LoF metrics for 9 genes |
| `epilepsy_snps.csv` | `data/raw/gwas/` | 15 GWAS SNPs with OR and RAF |
| `genetic_profiles.csv` | `data/processed/genetic_vectors/` | 5 patients × 12 features |
| `chb01_sequences.npy` | `data/processed/eeg_features/` | (95699, 1280, 17) float32 |
| `chb01_features.npy` | `data/processed/eeg_features/` | (95699, 221) float32 |
| `chb01_labels.npy` | `data/processed/eeg_features/` | (95699,) int8 |
| `chb03_sequences.npy` | `data/processed/eeg_features/` | (87780, 1280, 17) float32 |
| `chb03_features.npy` | `data/processed/eeg_features/` | (87780, 221) float32 |
| `chb03_labels.npy` | `data/processed/eeg_features/` | (87780,) int8 |
| `config.yaml` | `configs/` | All hyperparameters and paths |
