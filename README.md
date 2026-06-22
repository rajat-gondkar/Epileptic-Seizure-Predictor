# EEG + Genetic Marker Fusion for Epileptic Seizure Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch" />
  <img src="https://img.shields.io/badge/XGBoost-1.7%2B-brightgreen" />
  <img src="https://img.shields.io/badge/React-18-61dafb?logo=react" />
  <img src="https://img.shields.io/badge/Models-Trained%20%26%20Evaluated-success" />
</p>

A multimodal AI system that fuses **EEG signals** with **patient-specific genetic biomarkers** through a learned attention gate to deliver personalised epileptic seizure prediction.

**Project:** RVCE Innovative Design Project (IDP)
**Team:** Rajat G A · Srijeeta Ghosh · Aamir Ibrahim · Misab Abdul Raheem
**Guide:** Prof. Deepika P

> For the full technical specification (every hyperparameter, dataset table, and result), see **[`TECHNICAL_REFERENCE.md`](./TECHNICAL_REFERENCE.md)**.

---

## Overview

Epilepsy affects ~50 million people worldwide, yet most seizure-prediction systems treat all patients identically and ignore the genetic variability in epilepsy aetiology. This project addresses that gap with a **late-fusion multimodal architecture**:

1. **EEG Branch** — A Bidirectional LSTM with attention trained on the CHB-MIT scalp EEG database. It operates on raw preprocessed 30-second windows (no hand-crafted feature extraction) and learns temporal patterns directly. Produces a 512-dim embedding plus a 3-class prediction (interictal / preictal / ictal).
2. **Genetic Branch** — An XGBoost classifier on a **22-dimensional** genetic feature vector: 9 weighted gene-mutation flags, 7 gnomAD pLI scores, a Polygenic Risk Score from 15 GWAS SNPs, and 5 engineered interaction/burden features.
3. **Fusion Layer** — An attention-gated late fusion that learns a per-patient weight α blending the EEG embedding with the genetic risk score into a final risk `P_final ∈ [0,1]`.
4. **Synthetic Data (CTGAN)** — Generates a constrained synthetic cohort, used strictly as a **validation benchmark** (not as fusion training data).
5. **Clinical Alert** — `P_final` maps to a 4-level alert system (Low / Moderate / High / Critical).
6. **Showcase Dashboard** — A reactive React + Vite dashboard that visualises the entire pipeline and includes an interactive fusion simulator.

---

## System Architecture

```
                    ┌─────────────────────────┐
                    │      Patient Input       │
                    │   EEG + Genetic Profile  │
                    └───────────┬──────────────┘
                                │
              ┌─────────────────┴──────────────────┐
              ▼                                     ▼
  ┌───────────────────────────┐       ┌──────────────────────────┐
  │       EEG Branch           │       │      Genetic Branch       │
  │                            │       │                           │
  │  Input: [B, 3840, 19]      │       │  Input: [B, 22]           │
  │  BiLSTM (hidden=256, 2L)   │       │  XGBoost (300 trees,      │
  │  + Attention over time     │       │           max_depth=3)    │
  │  → 512-dim embedding       │       │                           │
  │  → 3-class head            │       │  Output: P_genetic        │
  └───────────┬───────────────┘       └────────────┬─────────────┘
              │ 512-dim embedding                   │ risk score
              └─────────────────┬───────────────────┘
                                ▼
                   ┌─────────────────────────┐
                   │  Attention-Gated Fusion  │
                   │  α = σ(W_e·h_e+W_g·h_g+b) │
                   │  P_final = α·EEG+(1−α)·Gen│
                   └────────────┬─────────────┘
                                ▼
                   ┌─────────────────────────┐
                   │   Risk Score P ∈ [0,1]   │
                   │   4-Level Alert System   │
                   └─────────────────────────┘
```

---

## Project Status

| Component | Description | Status |
|-----------|-------------|--------|
| Data Acquisition | CHB-MIT + ClinVar + gnomAD + GWAS | ✅ Complete |
| EEG Preprocessing | Bandpass, resample, z-score (per-window, lazy) | ✅ Complete |
| Genetic Feature Engineering | 22-dim vector + PRS | ✅ Complete |
| Synthetic Data (CTGAN) | 5,000 validation records, KS-tested | ✅ Complete |
| EEG Branch (BiLSTM + Attention) | Trained, 96.34% test accuracy | ✅ Complete |
| Genetic Branch (XGBoost) | Trained, AUC 0.739 / AUC-PR 0.869 | ✅ Complete |
| Fusion Layer | Trained, AUC 0.888 (+14.4% over EEG-only) | ✅ Complete |
| Showcase Dashboard (React) | Reactive pipeline + fusion simulator + live demo | ✅ Complete |
| Backend API (FastAPI) | Real-time BiLSTM inference on uploaded EDF files | ✅ Complete |

---

## Datasets

| Dataset | Source | Purpose | Size |
|---------|--------|---------|------|
| [CHB-MIT EEG](https://physionet.org/content/chbmit/1.0.0/) | PhysioNet | Training the EEG branch | ~40–45 GB |
| [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) | NCBI | Gene mutation flags | 13,331 pathogenic variants |
| [gnomAD v2.1.1](https://gnomad.broadinstitute.org/) | Broad Institute | Gene constraint (pLI) | 9 genes |
| [GWAS Catalog](https://www.ebi.ac.uk/gwas/) | EBI / Literature | Polygenic Risk Score | 15 epilepsy SNPs |

**EEG patients used (11):** chb01, chb02, chb03, chb04, chb07, chb09, chb11, chb14, chb15, chb18, chb24
— **355 EDF files**, 68 with seizure annotations, 256 Hz (downsampled to 128 Hz).

**Target genes (9):** SCN1A · SCN8A · KCNQ2 · SCN2A · KCNT1 · DEPDC5 · PCDH19 · GRIN2A · GABRA1

---

## EEG Preprocessing Pipeline

Preprocessing is applied **per-window** in `seizure_prediction/libCHBMITDataset.py` (lazy loading — only the needed 30-second slice is read from disk, keeping RAM < 3 GB). The model trains on the **raw preprocessed signal**; no spectral/Hjorth feature extraction is performed.

```
EDF File (CHB-MIT, 256 Hz)
  │
  ├─ EDF read: only the required 30-second window (PyEDFlib)
  ├─ Channel selection: 19 common bipolar 10-20 channels
  ├─ Bandpass filter: 0.5–45 Hz (4th-order Butterworth)
  ├─ Resample: 256 Hz → 128 Hz
  ├─ Z-score normalisation: per channel (zero mean, unit variance)
  └─ Transpose: (channels, time) → (time, channels)
       → Input tensor: (batch=16, time=3840, channels=19)
```

**Three-zone labeling** per seizure: interictal (0) · preictal (1, 25-min window) · 5-min prediction-horizon gap (0) · ictal (2).

**Window class distribution (53,858 windows):** Interictal 96.0% · Preictal 3.8% · Ictal 0.2%.

---

## Genetic Feature Vector (22 dimensions)

```python
[
  # [0-8]  Weighted mutation flags (0 or gene risk weight)
  SCN1A_mutation, SCN8A_mutation, KCNQ2_mutation, SCN2A_mutation,
  KCNT1_mutation, DEPDC5_mutation, PCDH19_mutation, GRIN2A_mutation, GABRA1_mutation,
  # [9-15] gnomAD pLI scores
  SCN1A_pLI, SCN8A_pLI, KCNQ2_pLI, SCN2A_pLI, GRIN2A_pLI, PCDH19_pLI, GABRA1_pLI,
  # [16]   Polygenic Risk Score (standardised)
  polygenic_risk_score,
  # [17-21] Engineered interaction / aggregate features
  sodium_channel_interaction, potassium_channel_interaction,
  receptor_interaction, prs_tier1_interaction, mutation_burden,
]
```

**PRS formula:** `PRS = Σ ln(OR_i) × genotype_i`, where `genotype_i ∈ {0,1,2}` is Hardy-Weinberg sampled from GWAS risk-allele frequencies, then z-score standardised.

**Cohort:** The genetic branch is trained on **10,000 fully synthetic patients** generated from population-level carrier frequencies. CHB-MIT provides no genetic sequencing, so no real patient genetics are used. See `scripts/generate_synthetic_genetic_patients.py`.

---

## Results

### EEG Branch (BiLSTM + Attention) — 53,858 windows
| Metric | Value |
|--------|-------|
| Overall accuracy | **96.34%** |
| Ictal recall (seizure detection) | 89.4% |
| Preictal recall (prediction) | 73.4% |
| Interictal specificity | 97.3% |

### Genetic Branch (XGBoost) — 10,000 synthetic patients
| Metric | Value |
|--------|-------|
| AUC | 0.739 |
| AUC-PR | 0.869 |
| Precision | 0.867 |
| Recall | 0.553 |

### Fusion Layer — EEG-only vs Fusion (13,508 test windows)
| Metric | EEG-only | Fusion | Δ |
|--------|----------|--------|---|
| AUC | 0.744 | **0.888** | +14.4% |
| AUC-PR | 0.255 | **0.443** | +18.7% |
| F1 | 0.352 | **0.456** | +10.4% |
| Recall | 0.320 | **0.413** | +9.3% |
| MCC | 0.327 | **0.437** | +11.0% |

Fusion catches 55 more seizure windows while reducing false positives by 61. The learned attention settled near α ≈ 0.50 because the synthetic genetic scores have very low variance; the gains flow through the risk head, which also receives the raw genetic score directly.

### CTGAN Validation
KS pass rate 31.6% overall (genetic columns 100%, EEG columns 0%) — used as a validation benchmark only.

---

## Quick Start

```bash
# 1. Clone and set up the Python environment
git clone https://github.com/<your-username>/eeg-genetic-fusion.git
cd eeg-genetic-fusion
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Acquire datasets
python scripts/acquire_genetic_data.py          # ClinVar + gnomAD + GWAS
python scripts/acquire_chbmit.py                 # CHB-MIT EDF files (large; resumable)

# 3. Genetic pipeline
python scripts/generate_synthetic_genetic_patients.py --n-patients 10000
python scripts/train_xgboost_genetic.py          # → models/xgboost_genetic/

# 4. EEG branch (trained inside seizure_prediction/)
cd seizure_prediction
bash runTrainLSTM.sh                             # or: python scrTrainLSTM.py ...
cd ..

# 5. Fusion pipeline
python scripts/extract_eeg_embeddings.py         # cache 512-dim embeddings (~2.5h)
python scripts/train_fusion_clean.py             # train the attention-gated fusion
python scripts/evaluate_fusion.py                # EEG-only vs fusion comparison
```

### Frontend dashboard

```bash
cd frontend/dashboard
npm install
npm run dev        # http://localhost:5173
npm run build      # production build → dist/
```

The dashboard is fully static and reads pre-extracted results from `frontend/dashboard/src/data/*.json` — no backend required for the showcase sections.

### Live inference API (for the upload-an-EDF demo)

```bash
# from the project root
./venv/bin/python -m uvicorn src.api.main:app --port 8000
# or: bash scripts/run_inference_api.sh
```

Loads the real all-patients BiLSTM and exposes `POST /api/predict` (upload a
`.edf`, get per-window interictal/preictal/ictal predictions). The dashboard's
**Live Demo** section auto-detects it at `http://localhost:8000`; if it's
offline it falls back to a labelled simulated playback.

> **Note:** EDF files and large processed arrays are excluded from the repository (see `.gitignore`). Regenerate them locally with the acquisition scripts above.

---

## Directory Structure

```
eeg-genetic-fusion/
│
├── configs/
│   └── config.yaml                   # Hyperparameters, paths, band definitions
│
├── data/
│   ├── raw/                          # EDF + genomic source files (mostly untracked)
│   │   ├── chb-mit/                  # EDF files — NOT tracked
│   │   ├── clinvar/epilepsy_variants.csv     # 13,331 pathogenic variants
│   │   ├── gnomad/pli_scores.csv             # pLI scores
│   │   └── gwas/epilepsy_snps.csv            # 15 GWAS SNPs
│   ├── processed/
│   │   ├── genetic_vectors/          # genetic_profiles.csv, training_cohort.csv
│   │   └── synthetic/                # CTGAN model + synthetic records + report
│   └── splits/
│
├── seizure_prediction/               # EEG branch (BiLSTM + attention)
│   ├── libCHBMITDataset.py           # Lazy dataset: filter, resample, z-score
│   ├── libDataIO.py                  # EDF reading, 3-zone preictal labeling
│   ├── libModelLSTM.py               # BiLSTM + attention (clsLSTM, get_embedding)
│   ├── scrTrainLSTM.py / scrTestLSTM.py
│   └── setup_chbmit.py               # Downloader + annotation converter
│
├── src/
│   ├── data_pipeline/
│   │   ├── genetic_feature_engineering.py   # 22-dim genetic vector
│   │   └── prs_computation.py               # Polygenic Risk Score
│   ├── training/
│   │   └── fusion.py                 # AttentionGateFusion + trainer + inference
│   ├── models/                       # (scaffold)
│   ├── evaluation/                   # (scaffold)
│   └── api/                          # FastAPI inference backend
│       ├── main.py                   # endpoints: /api/health, /model-info, /predict
│       └── inference.py              # EDF preprocessing + real BiLSTM inference
│
├── scripts/
│   ├── acquire_chbmit.py             # CHB-MIT downloader
│   ├── acquire_genetic_data.py       # ClinVar + gnomAD + GWAS downloader
│   ├── generate_synthetic_genetic_patients.py
│   ├── train_xgboost_genetic.py
│   ├── extract_eeg_embeddings.py     # one-time embedding extraction
│   ├── train_fusion_clean.py         # fusion training
│   └── evaluate_fusion.py            # EEG-only vs fusion comparison
│
├── models/
│   └── xgboost_genetic/              # Trained XGBoost model, metrics, plots
│
├── fusion results/                   # Trained fusion model + metrics + plots
│   ├── fusion_best.pt / fusion_final.pt
│   ├── fusion_evaluation.json        # EEG-only vs fusion
│   ├── fusion_metrics.json           # training history + test metrics
│   └── plots/
│
├── frontend/dashboard/               # React + Vite showcase dashboard
│   ├── src/components/               # Hero, Datasets, Fusion simulator, etc.
│   └── src/data/                     # Static result JSON powering the UI
│
├── TECHNICAL_REFERENCE.md            # Full technical spec
├── requirements.txt
├── docker-compose.yml
└── .gitignore
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| EEG Signal Processing | PyEDFlib, SciPy |
| Deep Learning | PyTorch ≥ 2.0 |
| Classical ML | XGBoost ≥ 1.7 |
| Synthetic Data | CTGAN / SDV |
| Explainability | SHAP |
| Frontend | React 18, Vite, Tailwind CSS, Recharts, Framer Motion |
| Backend API | FastAPI + Uvicorn |
| Containerisation | Docker Compose |
| Python | 3.12 |

---

## License

This project is for academic and research purposes. The CHB-MIT dataset is provided under the PhysioNet Credentialed Health Data License and requires a signed data use agreement before access.
