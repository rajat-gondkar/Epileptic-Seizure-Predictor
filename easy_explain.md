# EEG-Genetic Fusion — Easy Project Summary

**What this project does:** We are building an AI system that predicts epileptic seizures before they happen by combining two types of patient data — brain recordings (EEG) and genetic risk markers. The idea is that seizures leave early warning signals in both brain waves and DNA, and fusing them together should give more reliable predictions than using either one alone.

---

## 1. Data We Collected

We gathered data from four different sources and combined them into one dataset.

| Source | What it is | Size | Why we need it |
|--------|-----------|------|----------------|
| **CHB-MIT Scalp EEG** | Raw brainwave recordings from epilepsy patients | 190 files, 8 patients (chb01–chb20) | This is the main signal we learn from |
| **ClinVar** | Database of known disease-causing gene mutations | 13,331 pathogenic variants across 9 epilepsy genes | Tells us which specific gene mutations matter |
| **gnomAD** | Population-level gene constraint scores | pLI scores for all 9 target genes | Tells us how intolerant each gene is to damage |
| **GWAS Catalog** | Genome-wide association study results | 15 epilepsy-linked SNPs with effect sizes | Used to compute a Polygenic Risk Score (PRS) |

**CHB-MIT breakdown:**
- Full download: chb01 (42 files), chb03 (38 files), chb05 (39 files)
- Selective download: chb06, chb08, chb10, chb16, chb20
- 59 of the 190 files contain at least one annotated seizure
- Sampling rate: 256 Hz (256 samples per second)

---

## 2. EEG Preprocessing

Raw EEG is messy — it contains eye blinks, muscle noise, power-line hum, and random movement. We clean it step by step before the model sees it.

| Step | What we did | Why |
|------|-------------|-----|
| **Channel selection** | Picked 17 standard 10-20 bipolar channels | Removes duplicates and non-EEG channels |
| **Bandpass filter** | 4th-order Butterworth, 0.5–70 Hz | Keeps brain-relevant frequencies; removes slow drift and high noise |
| **Notch filter** | 60 Hz | Removes US power-line electrical interference |
| **Re-reference** | Common Average Reference (CAR) | Makes all channels comparable |
| **Artifact rejection** | Reject epochs where any channel exceeds ±500 µV | Removes movement and equipment artifacts |
| **Epoching** | 5-second windows, 1-second stride | Creates training examples; overlapping windows preserve time continuity |

**Labeling:**
- **Interictal** (label = 0): Normal brain state, far from any seizure
- **Preictal** (label = 1): Within 30 minutes *before* a seizure starts — this is what we want to predict
- **Excluded** (label = -1): During seizure, post-seizure recovery, or artifact-corrupted

**Dataset stats after preprocessing:**

| Patient | Total Epochs | Interictal | Preictal |
|---------|-------------|------------|----------|
| chb01 | 95,699 | 86,837 | 8,862 |
| chb03 | 87,780 | 80,519 | 7,261 |
| chb05 | 49,919 | 47,107 | 2,812 |
| chb06 | 17,472 | 16,216 | 1,256 |
| chb08 | 14,662 | 12,883 | 1,779 |
| chb10 | 16,642 | 14,839 | 1,803 |
| chb16 | 17,547 | 15,261 | 2,286 |
| chb20 | 33,894 | 27,385 | 6,509 |
| **Total** | **333,615** | **301,047** | **32,568** |

Class imbalance: roughly **9 interictal epochs for every 1 preictal epoch**. The model training accounts for this.

---

## 3. Genetic Feature Engineering

**The 16-dimensional genetic vector per patient:**

| Index | Feature | Type | Source |
|-------|---------|------|--------|
| 0–8 | **Weighted mutation flags** (SCN1A, SCN8A, KCNQ2, SCN2A, KCNT1, DEPDC5, PCDH19, GRIN2A, GABRA1) | 0 or risk weight | ClinVar carrier frequencies × literature risk weights |
| 9 | SCN1A pLI score | 0–1 | gnomAD (how intolerant the gene is to damage) |
| 10 | SCN8A pLI score | 0–1 | gnomAD |
| 11 | Polygenic Risk Score (PRS) | Standardised continuous | GWAS effect sizes × simulated genotypes |
| 12 | **Mutation burden score** | Continuous | Sum of all weighted mutation flags |
| 13 | **Ion channel gene burden** | Continuous | Sum of ion-channel gene mutations only |
| 14 | **Tier-1 carrier flag** | Binary | 1 if any Tier-1 mutation present |
| 15 | **SCN1A severity proxy** | 0 or 1.0 | pLI × SCN1A mutation flag |

**Example simulated profiles:**
- chb03 carries an SCN1A mutation (most common epilepsy gene)
- chb01 has a high PRS (+1.02)
- pLI scores are fixed constants from gnomAD (SCN1A = 1.0, meaning completely intolerant to loss-of-function)

---

## 4. CTGAN — Synthetic Data Generation

**Why we did this:** Real patient data is limited (only 190 EDF files). Deep learning models need more examples to generalise well. We use **CTGAN** (Conditional Tabular GAN) to generate **1,000 realistic synthetic patient records** from the 190 real ones.

**How it works:**
1. Aggregate all epochs per EDF file into a single summary row (mean & std of each feature)
2. Attach the 16 genetic features for that patient
3. Log-transform tiny power values (~10⁻¹⁰) to prevent precision loss
4. Z-score normalise everything
5. Train CTGAN for 300 epochs (generator: 256×256, discriminator: 256×256)
6. Generate 1,000 synthetic rows
7. Apply biological constraints: mutations must be 0 or 1, pLI stays fixed, PRS clipped to realistic bounds

**Validation results:**

| Metric | Value | What it means |
|--------|-------|---------------|
| **Real records** | 190 | Original patient-file summaries |
| **Synthetic records** | 1,000 | Generated by CTGAN |
| **KS Test Pass Rate** | 24 / 31 (77.4%) | Statistical similarity test |
| **Mean Correlation Diff** | 0.160 | Difference in feature correlations between real and synthetic data; lower is better |
| **Seizure label preservation** | Real: 26.3% seizure → Synthetic: 29.2 | CTGAN preserves the class imbalance |

**What this tells us:** CTGAN captures the broad structure of the data but struggles with fine-grained statistical matching because 190 real examples is very small for a GAN. The synthetic data is still useful for augmenting training, but the model must be validated carefully on real held-out patients.

---

## 5. Model Training

### 5.1 XGBoost Branch (Genetic Model)

XGBoost is a gradient-boosted decision tree model trained **exclusively on the 16-dimensional genetic feature vector** using an **all-synthetic patient cohort**. It learns which combinations of gene mutations, pLI scores, PRS values, and engineered burden scores predict seizure risk.

**Training setup:**

| Parameter | Value | Reason |
|-----------|-------|--------|
| Input | 16-dim genetic vector per patient | 9 weighted mutations + 2 pLI + 1 PRS + 4 engineered features |
| Training data | **5,000 all-synthetic patients** | Generated from population frequencies with literature risk weights |
| Estimators | 300 (max) | Low-dim data converges faster |
| Max depth | 3 | Prevents overfitting on 16 features |
| Min child weight | 4 | Forces 4+ samples per leaf |
| Learning rate | 0.03 | Conservative updates |
| Subsample | 0.75 | Row sampling for regularisation |
| Colsample bytree | 0.80 | Feature sampling per tree |
| Reg alpha (L1) | 0.5 | Feature selection on sparse data |
| Reg lambda (L2) | 2.0 | Smooths synthetic noise |
| Scale pos weight | computed fresh | Matches actual genetic label distribution |
| Primary metric | AUC-PR | Better than ROC-AUC for imbalanced data |
| Device | **CPU** | Mac-compatible; no GPU required |
| Early stopping | 30 rounds | Stops if AUC-PR plateaus |

**Validation strategy:**
1. **Train/val/test split** (70% / 15% / 15%) with stratification
2. **5-fold stratified cross-validation** on the full cohort

**Engineered features:**
- **Mutation burden score:** Sum of all risk-weighted mutation flags
- **Ion channel burden:** Sum of ion-channel gene mutations only
- **Tier-1 carrier flag:** Binary — does patient carry any Tier-1 mutation?
- **SCN1A severity proxy:** pLI × SCN1A flag

**SHAP analysis** verifies the model learned real biology:
- Expected top features: mutation burden, SCN1A, PRS, KCNQ2
- Red flag if PCDH19 or GABRA1 rank near top

**Actual Results (trained on 5,000 synthetic patients):**

| Metric | Value | What it means |
|--------|-------|---------------|
| **AUC** | 0.739 | Clearly learns a real genetic signal (above random 0.5) |
| **AUC-PR** | 0.869 | **Strong** — excellent ranking of seizure vs non-seizure patients |
| **Precision** | 0.867 | **Excellent** — when it flags high-risk, it's right 87% of the time |
| **Recall** | 0.553 | Moderate — catches about half of all true seizure patients |
| **F1** | 0.675 | Good balance between precision and recall |
| **Specificity** | 0.808 | Good — correctly identifies 81% of low-risk patients |

**Model profile:** Conservative but accurate. It rarely gives false alarms (high precision), but misses some true cases (moderate recall). This is clinically reasonable for a screening tool.


XGBoost's patient-level risk score feeds into the final fusion layer alongside the LSTM's epoch-level predictions.

### 5.2 EEG Branch — BiLSTM with STFT-CNN (In Progress). !!!!DONT LEARN THIS!!!!

This is the **deep learning branch** that learns directly from raw EEG waveforms. It is currently under active development.

**Architecture:**

| Layer | What it does |
|-------|-------------|
| **STFT** | Converts 1D EEG waveforms into 2D time-frequency spectrograms (70 frequency bins × ~21 time frames) |
| **2D CNN** | 3 convolutional layers that learn spatial patterns across frequency and time |
| **BiLSTM** | Bidirectional LSTM that reads the sequence forward and backward to capture temporal dynamics |
| **Self-Attention** | Learns which time steps are most important for seizure prediction |
| **FC + Sigmoid** | Outputs a probability P(seizure) ∈ [0, 1] |

**Training configuration:**

| Parameter | Value |
|-----------|-------|
| Optimiser | Adam, learning rate 1×10⁻⁴ |
| Loss | Focal Loss (gamma=2.0, alpha=0.90) |
| Batch size | 64 |
| Dropout | 0.5 |
| LR warmup | 5 epochs |
| Early stopping | 15 epochs (on validation AUC) |
| Gradient clipping | Max norm 1.0 |
| Augmentation | Gaussian noise + channel dropout |

**Why Focal Loss?** Normal loss functions get overwhelmed when one class is 9× more common than the other. Focal Loss forces the model to pay extra attention to the rare preictal examples so it doesn't just learn to predict "no seizure" every time.

**Why STFT instead of raw waveforms?** Raw 1D signals are extremely long (1,280 samples per epoch) and hard for LSTMs to process. STFT converts them into compact 2D images where seizure patterns (like theta slowing or spike bursts) become visible as blobs and ridges — much easier for CNNs to detect.

**Current status:** The architecture and training pipeline are fully implemented. Cloud training scripts are ready. Full end-to-end training on all 8 patients is the next step.

---

## 6. Fusion Layer (Planned)

After both branches are trained, an **attention-gated late fusion** layer dynamically weights the LSTM and XGBoost predictions per patient to produce a final risk score:

```
P_final = attention_weight × P_LSTM + (1 − attention_weight) × P_XGBoost
```

This score is mapped to a 4-level clinical alert system:
- **Green (0–0.25):** Low risk
- **Yellow (0.25–0.50):** Moderate risk
- **Orange (0.50–0.75):** High risk
- **Red (0.75+):** Critical — seizure likely imminent

---

## 7. Summary of What Exists vs. What's Next

| Component | Status |
|-----------|--------|
| Data downloading (CHB-MIT, ClinVar, gnomAD, GWAS) | ✅ Complete |
| EEG preprocessing pipeline | ✅ Complete (333K epochs) |
| Feature extraction (13 features × 17 channels) | ✅ Complete |
| Genetic feature engineering (16-dim vectors) | ✅ Complete |
| CTGAN synthetic data generation | ✅ Complete (190 → 1,000) |
| XGBoost genetic training | ✅ Complete — AUC 0.739, AUC-PR 0.869 |
| LSTM + STFT-CNN training | 🔄 In progress |
| Attention fusion layer | ⏳ Pending (after LSTM converges) |
| Real-time seizure prediction demo | ⏳ Future work |
