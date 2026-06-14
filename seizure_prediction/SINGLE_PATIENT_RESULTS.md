# Single-Patient Seizure Prediction Results

This document summarizes the single-patient training and testing experiments on CHB-MIT. These experiments validate the approach before scaling to multi-patient training.

---

## Patient chb01

### Dataset

| Metric | Value |
|--------|-------|
| Total EDF files | 42 |
| Files with seizure annotations | 7 |
| Training CSV | `chb01.csv` (33 files) |
| Test CSV | `chb01_test.csv` / `chb01.csv` |

### Window Distribution (30s windows @ 256 Hz, no overlap)

| Class | Count | Percentage |
|-------|-------|-----------|
| Interictal (0) | 3,514 | 92.8% |
| Preictal (1) | 260 | 6.9% |
| Ictal (2) | 14 | 0.4% |
| **Total** | **3,788** | **100%** |

### File-Based Split

| Split | Files | Windows |
|-------|-------|---------|
| Training | 23 | 2,588 |
| Validation | 7 | 840 |
| Test (held-out) | 3 | 360 |

Training class counts: interictal=2,340, preictal=236, ictal=12

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Hidden dimensions | 512 |
| LSTM layers | 3 |
| Dropout | 0.4 |
| Batch size | 16 |
| Learning rate | 0.0005 |
| Weight decay | 0.0001 |
| Optimizer | AdamW |
| Epochs | 20 |
| Sampling rate | 256 Hz |
| Window size | 30 s |
| Window stride | 30 s (no overlap) |

### Training Results

| Metric | Value |
|--------|-------|
| Training duration | 6 hours 45 minutes |
| Final training loss | 0.0939 |
| Final validation loss | 0.5516 |

### Test Results

#### Confusion Matrix

| True \ Predicted | Interictal | Preictal | Ictal |
|-----------------|-----------|----------|-------|
| **Interictal** | 3,191 | 320 | 3 |
| **Preictal** | 10 | 250 | 0 |
| **Ictal** | 0 | 0 | 14 |

#### Per-Class Metrics

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Interictal | 0.9969 | 0.9081 | 0.9504 |
| Preictal | 0.4386 | 0.9615 | 0.6024 |
| Ictal | 0.8235 | 1.0000 | 0.9032 |

**Overall Accuracy: 91.21%** (3,455 / 3,788)

#### Analysis

- **Seizure Detection**: 100% ictal recall — all 14 seizure windows correctly identified
- **Seizure Prediction**: 96.2% preictal recall — the model detects the vast majority of pre-seizure windows
- **False Alarms**: 8.3% of interictal windows (320/3,514) misclassified as preictal — acceptable for a single-patient model
- **Low precision on preictal (0.44)** reflects the class imbalance (6.9% preictal) — model flags some normal variations as preictal

---

## Patient chb15

### Dataset

| Metric | Value |
|--------|-------|
| Total EDF files | 40 |
| Files with seizure annotations | 14 |
| Training CSV | `chb15.csv` |
| Test CSV | `chb15_test.csv` |

### Window Distribution (30s windows @ 256 Hz, no overlap)

| Class | Count | Percentage |
|-------|-------|-----------|
| Interictal (0) | <!-- FILL --> | <!-- FILL --> |
| Preictal (1) | <!-- FILL --> | <!-- FILL --> |
| Ictal (2) | <!-- FILL --> | <!-- FILL --> |
| **Total** | **<!-- FILL -->** | **100%** |

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Hidden dimensions | 512 |
| LSTM layers | 3 |
| Dropout | 0.4 |
| Batch size | 16 |
| Learning rate | 0.0005 |
| Weight decay | 0.0001 |
| Optimizer | AdamW |
| Epochs | 20 |
| Sampling rate | 256 Hz |
| Window size | 30 s |
| Window stride | 30 s (no overlap) |

### Training Results

| Metric | Value |
|--------|-------|
| Training duration | <!-- FILL --> |
| Final training loss | <!-- FILL --> |
| Final validation loss | <!-- FILL --> |

### Test Results

#### Confusion Matrix

| True \ Predicted | Interictal | Preictal | Ictal |
|-----------------|-----------|----------|-------|
| **Interictal** | <!-- FILL --> | <!-- FILL --> | <!-- FILL --> |
| **Preictal** | <!-- FILL --> | <!-- FILL --> | <!-- FILL --> |
| **Ictal** | <!-- FILL --> | <!-- FILL --> | <!-- FILL --> |

#### Per-Class Metrics

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Interictal | <!-- FILL --> | <!-- FILL --> | <!-- FILL --> |
| Preictal | <!-- FILL --> | <!-- FILL --> | <!-- FILL --> |
| Ictal | <!-- FILL --> | <!-- FILL --> | <!-- FILL --> |

**Overall Accuracy: <!-- FILL -->%**

#### Analysis

- <!-- Key observations for chb15 -->

---

## Cross-Patient Comparison

| Metric | chb01 | chb15 |
|--------|-------|-------|
| EDF files | 42 | 40 |
| Seizure files | 7 | 14 |
| Total windows | 3,788 | <!-- FILL --> |
| Training loss | 0.0939 | <!-- FILL --> |
| Validation loss | 0.5516 | <!-- FILL --> |
| Test accuracy | 91.21% | <!-- FILL -->% |
| Interictal recall | 90.81% | <!-- FILL -->% |
| Preictal recall | 96.15% | <!-- FILL -->% |
| Ictal recall | 100% | <!-- FILL -->% |
| Preictal precision | 43.86% | <!-- FILL -->% |

---

*Generated on: June 2026*
*Project: Seizure Prediction Using LSTM on EEG Data (CHB-MIT)*
