# Seizure Prediction Model Analysis

## What's Actually Wrong (Diagnosis First)

Your logs tell the full story. Epoch 1: AUC = 0.500, sensitivity = 0, specificity = 1. Epoch 2: AUC = 0.488, sensitivity = 1, specificity = 0. The model is **oscillating between predicting all-negative and all-positive** — it hasn't learned anything. This is a classic unstable training collapse caused by a combination of issues hitting simultaneously.

The core problems:

- **pos_weight=3.0 is too low for an 11:1 ratio.** With 11:1 imbalance, you need `pos_weight` closer to 8–11, not 3. The `WeightedRandomSampler` and `pos_weight` are fighting each other — sampler pushes toward 25% positive in batches, but the loss weight still under-penalizes misses.
- **LR of 1e-3 is too aggressive for a 759k-param model on raw EEG sequences of length 1280.** The model is overshooting in early epochs before it finds any useful gradient signal.
- **Batch size of 32 with 1280-timestep sequences** means the LSTM is doing enormous forward passes with very noisy gradient estimates. This compounds the oscillation.
- **Your val set (16,642 epochs) is from a single patient** — if that patient has a very different seizure morphology, AUC on it is meaningless as a training signal.

---

## Recommended Parameters (EEG BiLSTM)

| Parameter | Current | Recommended | Reason |
|---|---|---|---|
| `pos_weight` | 3.0 | **9.0–11.0** | Match actual class ratio; remove sampler conflict |
| `WeightedRandomSampler` | 1/√count | **Remove it** | Use pos_weight alone. Sampler + high weight double-counts and causes instability |
| Learning rate | 1e-3 | **1e-4** | Standard for LSTM on long sequences |
| LR warmup | None | **Linear warmup 5 epochs** | Prevents early-epoch collapse before gradients stabilize |
| Batch size | 32 | **64** | Better gradient estimates with long sequences; more stable loss signal |
| Hidden size | 128 (256 eff) | **Keep, but reduce to 1 layer first** | Debug with simpler model; add layer back once training is stable |
| Dropout | 0.3+0.3 | **0.2+0.2** | You have no convergence yet — don't regularize a model that hasn't learned |
| Gradient clip | 5.0 | **1.0** | Grad norms at batch 0 are already small (0.63); clip tighter to prevent spikes |
| BCEWithLogitsLoss | pos_weight=3 | **Focal Loss (γ=2, α=0.75)** | Class imbalance consistently causes majority class to dominate standard BCE; Focal Loss directly down-weights easy negatives |
| Input sequence | 1280 raw samples | **Extract features (band power, Hjorth, etc.) → feed 221-dim per window** | Raw 1280-step LSTM is extremely hard to train; see below |
| Early stopping patience | 10 | **15** | With 1e-4 LR you need more epochs to see movement |
| Bias init | ln(pos_ratio) | **Keep** | This is correct, don't change |

---

## The More Important Suggestion: Consider a CNN-LSTM Front-End

The single biggest problem architecturally is feeding **raw 1280 time steps directly** into the BiLSTM. This is why training is unstable — gradients vanish across 1280 steps even with LSTM gates. State-of-the-art CHB-MIT models use 3D CNN front-ends to extract spatial-temporal features before feeding into BiLSTM, and CNN+LSTM pipelines on CHB-MIT consistently achieve higher sensitivity than LSTM alone on raw signals.

**Practical fix (minimal code change):** Add a **1D CNN front-end** (3 conv layers, kernel=5, stride=2) before the BiLSTM. This reduces sequence length from 1280 → ~160 before the LSTM sees it. Gradient flow improves dramatically.

---

## XGBoost Genetic Branch (Quick Assessment)

Your XGB config is reasonable but `scale_pos_weight=5` is again too low for your actual ratio (~9–11). Set it to **9**. `max_depth=4` is conservative (good for generalization on a 12-D genetic vector, but you're currently running it on 221-D EEG features — that's a different regime). `n_estimators=300` with `early_stopping=20` is fine.

The bigger issue: **you're training XGBoost on EEG features (221-D), not genetic features (12-D).** Once you move to the genetic branch proper with 12-D input, `max_depth=3` and `n_estimators=100–150` will be sufficient. Running a 300-tree depth-4 forest on 12 features will overfit badly.