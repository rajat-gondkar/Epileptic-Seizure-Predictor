# EEG + Genetic Marker Fusion for Epileptic Seizure Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch" />
  <img src="https://img.shields.io/badge/MNE--Python-1.12-green" />
  <img src="https://img.shields.io/badge/XGBoost-1.7%2B-brightgreen" />
  <img src="https://img.shields.io/badge/Status-In%20Development-yellow" />
</p>

A multimodal AI system that fuses **real-time EEG signals** with **patient-specific genetic biomarkers** to deliver personalised, early epileptic seizure prediction. Built as part of the RVCE Interdisciplinary Design Project (IDP).

---

## Overview

Epilepsy affects ~50 million people worldwide. Existing seizure prediction systems treat all patients identically, ignoring the significant genetic variability in epilepsy aetiology. This project addresses that gap by:

1. **EEG Branch** — A Bidirectional LSTM with self-attention trained on the CHB-MIT scalp EEG database (256 Hz, 5-second sliding windows, 30-minute preictal horizon)
2. **Genetic Branch** — An XGBoost classifier on a 12-dimensional genetic vector: 9 gene mutation flags (SCN1A, SCN8A, KCNQ2, SCN2A, KCNT1, DEPDC5, PCDH19, GRIN2A, GABRA1), 2 gnomAD pLI scores, and a Polygenic Risk Score from 15 GWAS SNPs
3. **Fusion Layer** — An attention-gated late fusion that dynamically weights both branches per patient to produce a final risk score
4. **Synthetic Data** — CTGAN to augment the paired EEG-genetic training set with population-constraint awareness
5. **Clinical Dashboard** — React.js frontend with a 4-level real-time alert engine

---

## System Architecture

```
                    ┌─────────────────────────┐
                    │      Patient Input        │
                    │   EEG + Genetic Profile   │
                    └───────────┬─────────────┘
                                │
              ┌─────────────────┴─────────────────┐
              ▼                                   ▼
  ┌───────────────────────┐         ┌─────────────────────────┐
  │     EEG Branch         │         │     Genetic Branch       │
  │                        │         │                          │
  │  Input: [B, 1280, 17]  │         │  Input: [B, 12]          │
  │  BiLSTM (hidden=128×2) │         │  XGBoost (300 trees)     │
  │  Self-Attention        │         │                          │
  │  FC(256→64) + ReLU     │         │  Output: P_genetic       │
  │  Output: P_eeg         │         └──────────┬──────────────┘
  └───────────┬───────────┘                     │
              │                                 │
              └─────────────────┬───────────────┘
                                ▼
                   ┌────────────────────────┐
                   │  Attention-Gated Fusion │
                   │                        │
                   │  α_eeg + α_genetic = 1 │
                   │  (learned per patient)  │
                   └────────────┬───────────┘
                                ▼
                   ┌────────────────────────┐
                   │   Risk Score P ∈ [0,1]  │
                   │   4-Level Alert System  │
                   └────────────────────────┘
```

---

## Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Environment Setup & Data Acquisition | ✅ Complete |
| 2 | EEG Preprocessing & Feature Extraction | 🔄 In Progress |
| 3 | Genetic Feature Engineering | ✅ Complete |
| 4 | Synthetic Data Generation (CTGAN) | 📋 Planned |
| 5 | Individual Model Training (LSTM + XGBoost) | 📋 Planned |
| 6 | Attention-Based Fusion Training | 📋 Planned |
| 7 | Evaluation & Benchmarking | 📋 Planned |
| 8 | FastAPI Backend | 📋 Planned |
| 9 | React.js Dashboard | 📋 Planned |
| 10 | Integration & Testing | 📋 Planned |

---

## Datasets

| Dataset | Source | Purpose | Size |
|---------|--------|---------|------|
| [CHB-MIT EEG](https://physionet.org/content/chbmit/1.0.0/) | PhysioNet | Training EEG branch | ~40 GB (subset: ~15 GB) |
| [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) | NCBI | Gene mutation flags | 13,331 pathogenic variants |
| [gnomAD v2.1.1](https://gnomad.broadinstitute.org/) | Broad Institute | Gene constraint (pLI) | 9 genes |
| [GWAS Catalog](https://www.ebi.ac.uk/gwas/) | EBI / Literature | Polygenic Risk Score | 15 epilepsy SNPs |

**EEG patients used:** chb01, chb03, chb05 (119 EDF files, 19 annotated seizures, 30-min preictal windows)

**Target genes:** SCN1A · SCN8A · KCNQ2 · SCN2A · KCNT1 · DEPDC5 · PCDH19 · GRIN2A · GABRA1

---

## Preprocessing Pipeline

```
EDF File (CHB-MIT, 256 Hz)
  │
  ├─ Channel selection: 17–18 bipolar 10-20 channels
  ├─ Bandpass filter: 0.5–70 Hz (4th-order Butterworth, zero-phase)
  ├─ Notch filter: 60 Hz
  ├─ Re-reference: Common Average Reference (CAR)
  ├─ Artifact rejection: peak-to-peak > 500 µV → excluded
  └─ Sliding epochs: 5 s window, 1 s stride
       ├─ Label 0: Interictal (baseline)
       ├─ Label 1: Preictal (≤ 30 min before seizure onset)
       └─ Label -1: Ictal / postictal → excluded from training
```

**Feature extraction (per epoch):** 13 features × 17 channels = **221-dimensional vector**
- Band powers: δ, θ, α, β, γ (Welch PSD)
- Ratios: α/β, θ/α
- Hjorth parameters: Activity, Mobility, Complexity
- Spike rate (peaks > 3 SD / sec)
- Sample Entropy (m=2, r=0.2×SD)

**Processed data (chb01 + chb03):**
- chb01: 95,699 valid epochs (86,837 interictal | 8,862 preictal)
- chb03: 87,780 valid epochs (80,519 interictal | 7,261 preictal)

---

## Genetic Feature Vector (12 dimensions)

```python
[
  SCN1A_mut, SCN8A_mut, KCNQ2_mut,   # Binary mutation flags from ClinVar
  SCN2A_mut, KCNT1_mut, DEPDC5_mut,  # (ClinVar: Pathogenic / Likely pathogenic)
  PCDH19_mut, GRIN2A_mut, GABRA1_mut,
  SCN1A_pLI,          # gnomAD v2.1.1 pLI score (1.0 = highly intolerant)
  SCN8A_pLI,          # gnomAD v2.1.1 pLI score
  PRS_standardised,   # Polygenic Risk Score from 15 GWAS SNPs, zscore-normalised
]
```

**PRS formula:** `PRS = Σ ln(OR_i) × genotype_i`  where genotype_i ∈ {0, 1, 2} (Hardy-Weinberg sampled from GWAS risk allele frequencies)

---

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/<your-username>/eeg-genetic-fusion.git
cd eeg-genetic-fusion
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Download datasets
python scripts/acquire_genetic_data.py        # Downloads ClinVar, gnomAD, GWAS
python scripts/acquire_chbmit.py \
    --patients chb01 chb03 chb05 \
    --data-dir data/raw/chb-mit               # ~15 GB; can resume if interrupted

# 3. EEG preprocessing
python src/data_pipeline/eeg_preprocessing.py --all
# Outputs: data/processed/eeg_features/<patient>_{sequences,features,labels}.npy

# 4. Genetic feature engineering
python src/data_pipeline/genetic_feature_engineering.py
# Output: data/processed/genetic_vectors/genetic_profiles.csv

# 5. (Optional) Start PostgreSQL via Docker
docker-compose up -d postgres

# 6. FastAPI backend (once models are trained)
uvicorn src.api.main:app --reload --port 8000
```

> **Note:** EDF files (~40 MB each) and processed `.npy` arrays (up to 7 GB each) are excluded from this repository. Download them locally using the acquisition scripts above.

---

## Directory Structure

```
eeg-genetic-fusion/
│
├── configs/
│   └── config.yaml                   # All hyperparameters, paths, band definitions
│
├── data/
│   ├── raw/
│   │   ├── chb-mit/                  # EDF files — NOT tracked (see .gitignore)
│   │   │   └── seizure_annotations.csv  # ✅ Tracked — 19 seizure ground truths
│   │   ├── clinvar/
│   │   │   └── epilepsy_variants.csv    # ✅ Tracked — 13,331 pathogenic variants
│   │   ├── gnomad/
│   │   │   └── pli_scores.csv           # ✅ Tracked — pLI for 9 genes
│   │   └── gwas/
│   │       └── epilepsy_snps.csv        # ✅ Tracked — 15 GWAS SNPs
│   ├── processed/                    # .npy arrays — NOT tracked (regenerate locally)
│   └── splits/
│
├── src/
│   ├── data_pipeline/
│   │   ├── eeg_preprocessing.py      # MNE pipeline, epoching, 221-dim features
│   │   ├── data_loader.py            # PyTorch Dataset + patient-level LOO-CV
│   │   ├── genetic_feature_engineering.py  # 12-dim genetic vector construction
│   │   └── prs_computation.py        # PRS computation (HWE simulation + standardisation)
│   ├── models/                       # BiLSTM, XGBoost, CTGAN, Fusion (planned)
│   ├── training/                     # Training scripts (planned)
│   ├── evaluation/                   # Metrics, SHAP explainability (planned)
│   └── api/                          # FastAPI backend (planned)
│
├── scripts/
│   ├── acquire_chbmit.py             # CHB-MIT downloader with resume support
│   ├── acquire_genetic_data.py       # ClinVar + gnomAD + GWAS downloader
│   └── run_acquisition.py            # Master orchestrator
│
├── notebooks/                        # EDA notebooks (planned)
├── models/                           # Saved weights — NOT tracked
├── frontend/dashboard/               # React.js dashboard (planned)
├── tests/
│
├── TECHNICAL_REFERENCE.md            # Full technical spec for paper writing
├── requirements.txt
├── docker-compose.yml
└── .gitignore
```

---

## Target Performance

| Metric | Target |
|--------|--------|
| AUROC | > 0.90 |
| Sensitivity (at FPR = 10%) | > 85% |
| Specificity | > 90% |
| False Prediction Rate | < 0.15 / hour |
| Seizure Prediction Horizon | 30 minutes |

**Baselines from literature:**
- Tsiouris et al. 2018 (LSTM, CHB-MIT): 99.6% accuracy
- Zhu et al. 2024 (Multidimensional Transformer): 98.24% sensitivity, 97.27% specificity

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| EEG Signal Processing | MNE-Python 1.12, SciPy |
| Deep Learning | PyTorch ≥ 2.0 |
| Classical ML | XGBoost ≥ 1.7 |
| Synthetic Data | CTGAN / SDV |
| Explainability | SHAP |
| Backend API | FastAPI + Uvicorn |
| Database | PostgreSQL 15 + SQLAlchemy |
| Frontend | React 18, Vite, Recharts, Socket.io |
| Containerisation | Docker Compose |
| Python | 3.12.12 |

---

## Team

| Name | Role |
|------|------|
| Rajat G A | Project Lead, EEG Pipeline, Model Architecture |
| Srijeeta Ghosh | Genetic Feature Engineering, Data Analysis |
| Aamir Ibrahim | Model Training, Evaluation |
| Misab Abdul Raheem | Frontend Dashboard, API Integration |

**Guide:** Prof. Deepika P  
**Institution:** RV College of Engineering (RVCE), Bengaluru

---

## License

This project is for academic and research purposes. The CHB-MIT dataset is provided under the PhysioNet Credentialed Health Data License and requires a signed data use agreement before access.
