# XGBoost Retraining on Genetic Data — Iterative Guide
### EEG-Genetic Fusion Project | Genetic Branch Redesign

> **Context:** The architecture has been clarified — EEG epochs go to the BiLSTM branch, and the **12-dimensional genetic vector goes exclusively to XGBoost**. The two outputs are later merged via an attention-gated fusion layer. This document covers everything you need to retrain XGBoost correctly on the genetic data alone.

---

## Part 1 — Why the Original XGBoost Setup Was Wrong

The original XGBoost was trained on the **221-dimensional EEG feature vector** (13 features × 17 channels). That was the wrong input. XGBoost should only ever see the **12-dimensional genetic vector** per patient. The fix is not just swapping the input — it changes everything about how you should configure and evaluate the model.

| Aspect | Original (Wrong) | Corrected |
|--------|-----------------|-----------|
| Input | 221-dim EEG features per epoch | 12-dim genetic vector per patient |
| Sample count | ~333,000 epochs | 8 real patients + 1,000 CTGAN synthetic rows |
| Signal type | Temporal, continuous | Tabular, mostly binary + 3 continuous |
| Tree depth | 4 (suited for EEG features) | 2–3 (low-dim tabular, reduce overfitting) |
| Prediction granularity | Per 5-sec epoch | Per patient (then broadcast to all their epochs) |

---

## Part 2 — Gene Risk Tier List (Literature-Backed)

This is the most important section for feature engineering. Not all 9 gene flags in your vector carry equal seizure risk. The literature is clear on a rough tier ordering. Use this to add **weighted mutation scores** instead of raw binary flags.

### Tier 1 — Highest Risk (Severe DEE, drug-resistant)

| Gene | pLI | Why it matters | Source |
|------|-----|----------------|--------|
| **SCN1A** | 1.00 | Most frequent monogenic epilepsy cause; Dravet syndrome. Pathogenic variants found in 15/339 patients in clinical panels — the single largest contributor. | Brunklaus et al. 2022; Thomas et al. 2019 |
| **KCNQ2** | 0.97 | Second most common; neonatal DEE with tonic seizures and suppression-burst EEG. 10/339 patients in clinical panels. | MLe-KCNQ2 (Frontiers, 2024); Thomas et al. 2019 |
| **STXBP1** | 1.00 | 5th most common epilepsy gene overall; epilepsy in 95% of carriers; accounts for 10–15% of early-infantile DEE. Severe phenotype. | EpiPred preprint 2025; HGMD top-10 |
| **SCN2A** | 0.99 | Overlaps with SCN1A phenotype; early-onset seizures, autism comorbidity. 6/339 patients in clinical panels. | Thomas et al. 2019 |

### Tier 2 — High Risk (Significant, more variable severity)

| Gene | pLI | Why it matters |
|------|-----|----------------|
| **SCN8A** | 0.99 | Early-onset epileptic encephalopathy; gain-of-function variants → hyperexcitability. 5/339 in clinical panels. |
| **CDKL5** | 1.00 | X-linked; epileptic spasms; 6/339 patients. More relevant for females. |
| **KCNT1** | 0.96 | Severe focal epilepsy of infancy (SFEI); gain-of-function → massive K⁺ current leak. |

### Tier 3 — Moderate Risk (Milder or more focal epilepsies)

| Gene | pLI | Why it matters |
|------|-----|----------------|
| **DEPDC5** | 0.96 | Focal epilepsy (FFEVF); autosomal dominant; seizures can be drug-responsive. |
| **GRIN2A** | 0.99 | Sleep-related epilepsy (CSWS, LKS); often adolescent-onset, milder. |
| **GABRA1** | 0.84 | Juvenile myoclonic epilepsy; idiopathic generalised; generally better prognosis. |
| **PCDH19** | 0.00 | X-linked, female-limited; cluster seizures in early childhood; male carriers unaffected. |

### Recommended: Replace Binary Flags with Weighted Scores

Instead of index 0–8 being raw `{0, 1}`, assign literature-derived risk weights:

```python
GENE_RISK_WEIGHTS = {
    'SCN1A':  1.00,   # Tier 1
    'KCNQ2':  0.95,   # Tier 1
    'STXBP1': 0.93,   # Tier 1
    'SCN2A':  0.90,   # Tier 1
    'SCN8A':  0.85,   # Tier 2
    'CDKL5':  0.82,   # Tier 2
    'KCNT1':  0.80,   # Tier 2
    'DEPDC5': 0.65,   # Tier 3
    'GRIN2A': 0.55,   # Tier 3
    'GABRA1': 0.50,   # Tier 3
    'PCDH19': 0.40,   # Tier 3 (female-limited)
}

# Weighted mutation flag: 0 if no mutation, weight if mutation present
genetic_vec[i] = GENE_RISK_WEIGHTS[gene] * binary_flag[i]
```

This moves mutation flags from 0/1 to 0/0.40–1.00, giving XGBoost a continuous gradient to split on instead of two discrete steps.

---

## Part 3 — Feature Engineering Additions

Your current 12-dim vector is functional but sparse. These additions are all derivable from your existing data with no new external sources.

### 3.1 Aggregate Mutation Burden Score

```python
# Sum of risk-weighted mutation flags across all 9 genes
mutation_burden = sum(GENE_RISK_WEIGHTS[g] * flag[g] for g in genes)
# Range: 0.0 (no mutations) to ~7.4 (all 9 mutations)
```

This single feature captures cumulative genetic load and tends to be one of the top SHAP contributors in epilepsy ML papers (TLE diagnostic model, PMC 2025).

### 3.2 Ion Channel Gene Burden

```python
# Only ion channel genes — these are the most directly seizure-relevant
ion_channel_genes = ['SCN1A', 'SCN2A', 'SCN8A', 'KCNQ2', 'KCNT1', 'GRIN2A']
ion_channel_burden = sum(GENE_RISK_WEIGHTS[g] * flag[g] for g in ion_channel_genes)
```

Ion channel dysfunction is the primary mechanism of seizure generation. Separating it from synaptic/transcriptional genes (STXBP1, DEPDC5, PCDH19) lets XGBoost learn different risk profiles.

### 3.3 Tier-1 Carrier Flag

```python
# Binary: does this patient carry ANY Tier-1 mutation?
tier1_flag = int(any(flag[g] == 1 for g in ['SCN1A', 'KCNQ2', 'STXBP1', 'SCN2A']))
```

Clinical panels consistently show that Tier-1 carriers need different treatment pathways. This is a strong discriminating feature.

### 3.4 SCN1A-Specific Severity Proxy

Since SCN1A has the best-characterised prediction model (Brunklaus et al. 2022, Broad Institute), you can encode a severity proxy:

```python
# pLI × mutation_flag gives partial credit even without the variant
scn1a_risk_proxy = pLI_SCN1A * flag_SCN1A  # 0 or 1.0
```

### Updated Vector (16 dimensions)

| Index | Feature | Type |
|-------|---------|------|
| 0–8 | Weighted mutation flags (9 genes) | 0 or risk weight |
| 9 | SCN1A pLI | Continuous 0–1 |
| 10 | SCN8A pLI | Continuous 0–1 |
| 11 | PRS | Standardised continuous |
| 12 | Mutation burden score | Continuous 0–7.4 |
| 13 | Ion channel gene burden | Continuous 0–4.6 |
| 14 | Tier-1 carrier flag | Binary |
| 15 | SCN1A severity proxy | 0 or 1.0 |

---

## Part 4 — XGBoost Hyperparameter Recommendations

The previous config was tuned for 221-dimensional EEG features. With a 16-dimensional genetic vector and ~1,000–1,200 samples (real + synthetic), the entire tuning philosophy changes.

### 4.1 Core Config Changes

| Parameter | Previous | Recommended | Reason |
|-----------|----------|-------------|--------|
| `n_estimators` | 300 | 100–200 (with early stop) | Low-dim data converges faster; more trees = overfitting |
| `max_depth` | 4 | **2–3** | 16 features don't need deep trees; depth 2–3 prevents memorising synthetic data patterns |
| `learning_rate` | 0.05 | **0.01–0.05** | Lower LR with more trees if AUC plateaus |
| `min_child_weight` | default (1) | **3–5** | Forces each leaf to have 3–5 samples; critical with small n |
| `subsample` | default (1.0) | **0.7–0.8** | Row subsampling reduces overfitting on 1,000-row dataset |
| `colsample_bytree` | default (1.0) | **0.7–0.9** | Feature subsampling per tree; with only 16 features, 0.7 still uses 11+ per tree |
| `reg_alpha` (L1) | 0 | **0.1–1.0** | L1 regularisation performs feature selection on low-dim sparse genetic data |
| `reg_lambda` (L2) | 1 | **1.0–5.0** | Increase L2 to smooth out CTGAN noise |
| `scale_pos_weight` | 9 | **Compute fresh** | Re-compute from new genetic-only label distribution |
| `eval_metric` | AUC | **`aucpr`** | Precision-Recall AUC is more informative than ROC-AUC for imbalanced small datasets |
| `tree_method` | `gpu_hist` | `hist` or `gpu_hist` | Either works; `hist` is fine for 1,000 rows |

### 4.2 Compute scale_pos_weight from Genetic Data

```python
# After creating your genetic-only dataset
n_neg = (y_genetic == 0).sum()   # non-seizure patients
n_pos = (y_genetic == 1).sum()   # seizure patients
scale_pos_weight = n_neg / n_pos
print(f"scale_pos_weight: {scale_pos_weight:.2f}")
```

Do NOT reuse the old value of 9 — that was based on epoch-level imbalance (333K epochs). At patient level the ratio will be different.

### 4.3 Recommended Full Config

```python
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

params = {
    'objective':         'binary:logistic',
    'eval_metric':       'aucpr',          # Precision-Recall AUC
    'n_estimators':      200,
    'max_depth':         3,
    'learning_rate':     0.03,
    'min_child_weight':  4,
    'subsample':         0.75,
    'colsample_bytree':  0.80,
    'reg_alpha':         0.5,
    'reg_lambda':        2.0,
    'scale_pos_weight':  scale_pos_weight,
    'tree_method':       'hist',
    'seed':              42,
}

model = xgb.XGBClassifier(**params, early_stopping_rounds=20)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Train with cross-validation on the genetic dataset
```

---

## Part 5 — Training Strategy for Small Genetic Dataset

### 5.1 Stratified K-Fold (Not Simple Train/Val Split)

With only 8 real patients + 1,000 synthetic rows, a single 80/20 split is unreliable. Use **5-fold stratified cross-validation**. For the real 8 patients — do **leave-one-patient-out (LOPO)** as your final evaluation:

```python
# LOPO evaluation — the only truly honest test with 8 real patients
real_patient_ids = ['chb01', 'chb03', 'chb05', 'chb06', 'chb08', 'chb10', 'chb16', 'chb20']

for test_patient in real_patient_ids:
    X_train = genetic_df[genetic_df['patient'] != test_patient].drop(['patient','label'], axis=1)
    y_train = genetic_df[genetic_df['patient'] != test_patient]['label']
    X_test  = genetic_df[genetic_df['patient'] == test_patient].drop(['patient','label'], axis=1)
    y_test  = genetic_df[genetic_df['patient'] == test_patient]['label']
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    # Record per-patient AUC-PR
```

### 5.2 Handling the Real vs Synthetic Mix

The 1,000 CTGAN rows are noisy (KS pass rate 12.9%). Two strategies:

**Option A — Sample weight penalty on synthetic rows (recommended):**
```python
sample_weights = np.where(df['source'] == 'real', 3.0, 1.0)
# Real patient rows get 3× the weight during training
model.fit(X_train, y_train, sample_weight=sample_weights[train_idx])
```

**Option B — Train on real only, use synthetic only for validation diversity.**
Use CTGAN rows as a held-out validation set to check that the model generalises beyond the 8 training patients. Do NOT mix them into LOPO training folds.

### 5.3 Optuna Hyperparameter Search (Recommended over Grid Search)

```python
import optuna

def objective(trial):
    params = {
        'max_depth':         trial.suggest_int('max_depth', 2, 4),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'min_child_weight':  trial.suggest_int('min_child_weight', 2, 8),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda':        trial.suggest_float('reg_lambda', 0.5, 5.0),
        'n_estimators':      200,
        'objective':         'binary:logistic',
        'eval_metric':       'aucpr',
        'scale_pos_weight':  scale_pos_weight,
        'seed':              42,
    }
    # 5-fold CV score
    scores = cross_val_score(xgb.XGBClassifier(**params), X, y, 
                              cv=StratifiedKFold(5), scoring='average_precision')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

---

## Part 6 — Evaluation Metrics

Do NOT use accuracy or standard AUC as primary metrics. Your dataset is imbalanced and small.

| Metric | Why use it | How to compute |
|--------|-----------|----------------|
| **AUC-PR** (Precision-Recall AUC) | Directly measures quality on minority class; preferred for imbalanced data | `average_precision_score(y_true, y_prob)` |
| **F1-Score (macro)** | Harmonic mean of precision/recall across classes | `f1_score(y_true, y_pred, average='macro')` |
| **Sensitivity (Recall)** | Fraction of seizure-prone patients correctly flagged | `recall_score(y_true, y_pred)` |
| **Specificity** | Fraction of low-risk patients correctly passed | `(TN / (TN + FP))` |
| **SHAP values** | Interpretability — which genes drove each prediction | See Part 7 |

**Target thresholds (from similar epilepsy genetic ML papers):**
- AUC-PR ≥ 0.75 on held-out real patients
- Sensitivity ≥ 0.80 (missing a high-risk patient is worse than a false alarm)
- AUC-ROC ≥ 0.80 is a secondary check

---

## Part 7 — SHAP Analysis for Genetic Interpretability

SHAP is essential here — it lets you verify that the model learned real biology (SCN1A matters more than GABRA1) rather than CTGAN noise. The TLE diagnostic model paper (PMC 2025) used SHAP to confirm its 10 key genetic features; your setup should do the same.

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot — shows which features matter most globally
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Force plot for a single patient — shows what drove their risk score
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0], 
                feature_names=feature_names)
```

**Expected SHAP ranking (if model learned correctly):**
1. Mutation burden score or tier-1 flag (top contributor)
2. SCN1A weighted flag
3. PRS
4. KCNQ2 or STXBP1 flag
5. pLI scores
6. Ion channel burden

If PCDH19 or GABRA1 rank near the top, that is a red flag — either the CTGAN introduced spurious correlations or the label assignment needs checking.

---

## Part 8 — Output Format for Fusion Layer

After XGBoost is trained, it needs to output a **per-patient probability** that feeds into the attention fusion equation:

```
P_final = α × P_LSTM + (1 − α) × P_XGBoost
```

The XGBoost output should be:

```python
# Per patient
p_genetic = model.predict_proba(X_genetic)[:, 1]  # shape: (n_patients,)

# Broadcast to all epochs for that patient
# Every epoch from chb01 inherits chb01's genetic risk score
epoch_df['genetic_risk'] = epoch_df['patient_id'].map(dict(zip(patient_ids, p_genetic)))
```

This means XGBoost's contribution to the fusion is **static per patient** — the genetic risk doesn't change epoch-to-epoch, while LSTM's output varies per 5-second window. The attention weight α will learn to up-weight LSTM when EEG patterns are strong, and up-weight XGBoost when EEG is ambiguous.

---

## Part 9 — Iteration Plan

| Iteration | What to do | What to check |
|-----------|-----------|---------------|
| **v1** | Train baseline XGBoost on current 12-dim genetic vector | AUC-PR, SHAP top features |
| **v2** | Replace binary flags with risk-weighted flags | Check if AUC-PR improves; SHAP ordering should shift toward SCN1A/KCNQ2 |
| **v3** | Add 4 engineered features (mutation burden, ion channel burden, tier-1 flag, SCN1A proxy) | Check for overfitting on 1,000 rows; use LOPO to validate |
| **v4** | Run Optuna HPO with new 16-dim vector | Find optimal depth/regularisation trade-off |
| **v5** | Apply sample weights (real: 3×, synthetic: 1×) | Check if real-patient LOPO scores improve |
| **v6** | Lock final model; extract per-patient probabilities | Pass these into the fusion layer |

---

## Quick Reference — Key Papers

| Paper | Year | Relevance |
|-------|------|-----------|
| Brunklaus et al. — SCN1A prediction model | 2022 | SCN1A is #1 monogenic epilepsy gene; used for severity scoring |
| Thomas et al. — 339-patient clinical gene panel | 2019 | Gene frequency ranking: SCN1A > KCNQ2 > CDKL5 > SCN2A > SCN8A |
| EpiPred (STXBP1) preprint | 2025 | STXBP1 = 5th most common epilepsy gene; 95% of carriers have epilepsy |
| TLE XGBoost/DNN model — 287 RNA-seq samples | 2025 | XGBoost with SHAP on genetic features; 10 optimised features → AUC 1.0 on TLE |
| ILAE GWAS meta-analysis (29,000 patients) | 2023 | 26 risk loci; confirms PRS utility for common epilepsies |
| MLe-KCNQ2 ML model | 2024 | KCNQ2-specific ML with Variant Frequency Index; transfers to XGBoost feature design |
| Kress et al. — Gradient boosting on SCN1A in silico | 2022 | Direct precedent for XGBoost on epilepsy genetics |