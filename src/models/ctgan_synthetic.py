#!/usr/bin/env python3
"""
CTGAN Synthetic Data Generation
=================================
Generates synthetic paired EEG–genetic patient records using
Conditional Tabular GAN (CTGAN).

Pipeline:
    1. Aggregate per-epoch EEG features into per-file summaries
    2. Attach patient-level genetic profiles
    3. Train CTGAN on the combined tabular dataset
    4. Generate synthetic records
    5. Apply biological constraints
    6. Validate quality with statistical tests

Usage:
    python src/models/ctgan_synthetic.py                  # full run
    python src/models/ctgan_synthetic.py --epochs 50      # quick test
    python src/models/ctgan_synthetic.py --n-samples 5000 # more synthetic data
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from ctgan import CTGAN
from scipy.stats import ks_2samp, mannwhitneyu
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
FEATURE_NAMES = [
    "delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power",
    "alpha_beta_ratio", "theta_alpha_ratio", "spike_rate", "variance",
    "hjorth_activity", "hjorth_mobility", "hjorth_complexity", "sample_entropy",
]
N_FEATURES_PER_CHANNEL = 13

# Columns that contain near-zero values (V²/Hz) and need log-transform
LOG_TRANSFORM_STEMS = [
    "delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power",
    "variance", "hjorth_activity",
]

GENETIC_COLUMNS = [
    "SCN1A_mutation", "SCN8A_mutation", "KCNQ2_mutation", "SCN2A_mutation",
    "KCNT1_mutation", "DEPDC5_mutation", "PCDH19_mutation", "GRIN2A_mutation",
    "GABRA1_mutation", "SCN1A_pLI", "SCN8A_pLI", "polygenic_risk_score",
]

MUTATION_COLUMNS = [c for c in GENETIC_COLUMNS if c.endswith("_mutation")]
PLI_COLUMNS = ["SCN1A_pLI", "SCN8A_pLI"]

# Known gnomAD pLI values (constants — never synthesised)
KNOWN_PLI = {"SCN1A_pLI": 1.0, "SCN8A_pLI": 1.0}


def load_config(config_path=None):
    if config_path is None:
        candidates = [
            Path("configs/config.yaml"),
            Path(__file__).parent.parent.parent / "configs" / "config.yaml",
        ]
        for p in candidates:
            if p.exists():
                config_path = p
                break
        else:
            raise FileNotFoundError("configs/config.yaml not found")
    with open(config_path) as f:
        return yaml.safe_load(f)


# ============================================================
# 1. Build Summary Dataset
# ============================================================
def build_summary_dataset(config):
    """
    Aggregate per-epoch EEG features into per-file summaries.

    For each EDF file we compute:
      - Mean of each of 13 feature types (averaged across 17 channels)  → 13 values
      - Std of each                                                     → 13 values
      - Preictal ratio (fraction of preictal epochs)                    → 1 value
      - Total valid epochs                                              → 1 value

    Then attach the patient's 12-dim genetic profile.

    Returns:
        pd.DataFrame with shape ≈ (119, 41)
    """
    feat_dir = Path(config["paths"]["data"]["processed"]["eeg_features"])
    data_dir = Path(config["paths"]["data"]["raw"]["chb_mit"])
    gen_path = Path(config["paths"]["data"]["processed"]["genetic_vectors"]) / "genetic_profiles.csv"

    genetic_df = pd.read_csv(gen_path)
    genetic_df = genetic_df.set_index("patient_id")

    # Load seizure annotations to know which files had seizures
    ann_path = data_dir / "seizure_annotations.csv"
    ann_df = pd.read_csv(ann_path)
    ann_df = ann_df.apply(lambda c: c.str.strip() if c.dtype == object else c)

    # Auto-detect all patients with processed features
    all_patient_dirs = sorted(data_dir.iterdir())
    patients = []
    for d in all_patient_dirs:
        if d.is_dir() and (feat_dir / f"{d.name}_features.npy").exists():
            if d.name in genetic_df.index:
                patients.append(d.name)

    print(f"Building summary dataset for patients: {patients}")

    records = []

    for pat in patients:
        features = np.load(feat_dir / f"{pat}_features.npy")  # [N, 221]
        labels = np.load(feat_dir / f"{pat}_labels.npy")       # [N]
        n_channels = features.shape[1] // N_FEATURES_PER_CHANNEL  # 17

        # Get genetic profile for this patient
        if pat in genetic_df.index:
            gen_row = genetic_df.loc[pat]
        else:
            continue

        # Get list of EDF files
        edf_files = sorted((data_dir / pat).glob("*.edf"))

        # We need to figure out which epochs came from which file.
        # Each EDF is 1 hour = 3600s. With 5s epochs and 1s stride:
        # max epochs per file = (3600 - 5) / 1 + 1 = 3596
        # But some epochs are rejected, so actual count varies.
        # We'll split the epoch array proportionally by file count.
        # This is an approximation — good enough for summary stats.

        n_files = len(edf_files)
        n_epochs = len(labels)
        epochs_per_file = n_epochs // n_files
        remainder = n_epochs % n_files

        file_seizure_set = set(
            ann_df[ann_df["patient_id"] == pat]["file"].values
        )

        offset = 0
        for i, edf in enumerate(edf_files):
            # Determine epoch range for this file
            chunk_size = epochs_per_file + (1 if i < remainder else 0)
            if chunk_size == 0:
                continue
            end = offset + chunk_size

            file_feats = features[offset:end]   # [chunk, 221]
            file_labels = labels[offset:end]

            # Reshape to [chunk, n_channels, 13] and average across channels
            reshaped = file_feats.reshape(len(file_feats), n_channels, N_FEATURES_PER_CHANNEL)
            global_feats = reshaped.mean(axis=1)  # [chunk, 13]

            # Compute summary stats
            feat_means = np.nanmean(global_feats, axis=0)  # [13]
            feat_stds = np.nanstd(global_feats, axis=0)    # [13]

            preictal_ratio = (file_labels == 1).sum() / max(len(file_labels), 1)
            n_valid = len(file_labels)
            has_seizure = 1 if edf.name in file_seizure_set else 0

            record = {"patient_id": pat, "file": edf.name}
            for j, name in enumerate(FEATURE_NAMES):
                record[f"{name}_mean"] = float(feat_means[j])
                record[f"{name}_std"] = float(feat_stds[j])
            record["preictal_ratio"] = float(preictal_ratio)
            record["n_valid_epochs"] = int(n_valid)
            record["has_seizure"] = has_seizure

            # Attach genetic features
            for col in GENETIC_COLUMNS:
                record[col] = float(gen_row[col])

            records.append(record)
            offset = end

    df = pd.DataFrame(records)

    # ── Log-transform near-zero columns ──────────────────────
    # Band powers, variance, hjorth_activity are in V²/Hz (~1e-10).
    # CTGAN sees them all as 0. Log10(x + eps) makes them learnable.
    log_cols = []
    for stem in LOG_TRANSFORM_STEMS:
        for suffix in ["_mean", "_std"]:
            col = f"{stem}{suffix}"
            if col in df.columns:
                eps = 1e-15
                df[col] = np.log10(df[col].clip(lower=eps) + eps)
                log_cols.append(col)

    print(f"Summary dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Seizure files: {df['has_seizure'].sum()} / {len(df)}")
    print(f"  Patients: {df['patient_id'].nunique()}")
    print(f"  Log-transformed columns: {len(log_cols)}")
    return df, log_cols


# ============================================================
# 2. Train CTGAN
# ============================================================
def train_ctgan(summary_df, config, epochs=None):
    """
    Train CTGAN on the summary dataset.

    Applies StandardScaler on continuous columns before training
    so CTGAN sees well-behaved distributions.

    Returns:
        trained CTGAN model, training columns, discrete columns, scaler, continuous columns
    """
    ctgan_cfg = config["ctgan"]
    if epochs is None:
        epochs = ctgan_cfg["epochs"]

    # Drop metadata columns — keep only numeric features for training
    drop_cols = ["patient_id", "file"]
    train_df = summary_df.drop(columns=drop_cols, errors="ignore")

    # Identify discrete columns
    discrete_cols = MUTATION_COLUMNS + ["has_seizure"]
    discrete_cols = [c for c in discrete_cols if c in train_df.columns]

    # ── StandardScaler on continuous columns ──────────────────
    continuous_cols = [c for c in train_df.columns if c not in discrete_cols]
    scaler = StandardScaler()
    train_df[continuous_cols] = scaler.fit_transform(train_df[continuous_cols])

    print(f"\nTraining CTGAN:")
    print(f"  Training data: {train_df.shape}")
    print(f"  Continuous columns: {len(continuous_cols)} (StandardScaled)")
    print(f"  Discrete columns: {len(discrete_cols)} → {discrete_cols}")
    print(f"  Epochs: {epochs}")
    print(f"  Generator dim: {ctgan_cfg['generator_dim']}")
    print(f"  Discriminator dim: {ctgan_cfg['discriminator_dim']}")
    print(f"  Batch size: {ctgan_cfg['batch_size']}")

    model = CTGAN(
        epochs=epochs,
        generator_dim=tuple(ctgan_cfg["generator_dim"]),
        discriminator_dim=tuple(ctgan_cfg["discriminator_dim"]),
        batch_size=ctgan_cfg["batch_size"],
        verbose=True,
    )

    model.fit(train_df, discrete_columns=discrete_cols)

    return model, list(train_df.columns), discrete_cols, scaler, continuous_cols


# ============================================================
# 3. Generate & Constrain
# ============================================================
def generate_synthetic(model, n_samples, columns, discrete_cols,
                       scaler=None, continuous_cols=None, log_cols=None):
    """
    Generate synthetic samples and inverse-transform:
      1. Inverse StandardScaler on continuous columns
      2. Inverse log10 on log-transformed columns (10^x)
    """
    print(f"\nGenerating {n_samples} synthetic samples...")
    synthetic = model.sample(n_samples)
    synthetic.columns = columns

    # Inverse StandardScaler
    if scaler is not None and continuous_cols is not None:
        synthetic[continuous_cols] = scaler.inverse_transform(synthetic[continuous_cols])

    # Inverse log10 → back to original scale
    if log_cols:
        for col in log_cols:
            if col in synthetic.columns:
                synthetic[col] = np.power(10, synthetic[col])

    return synthetic


def apply_biological_constraints(df):
    """
    Post-generation biological constraint enforcement.

    Rules:
        1. Mutation flags ∈ {0, 1}
        2. pLI scores frozen to known gnomAD values
        3. PRS clipped to [−5, +5] SD
        4. Preictal ratio ∈ [0, 1]
        5. Band powers and variance ≥ 0
        6. Hjorth mobility and complexity ≥ 0
        7. has_seizure ∈ {0, 1}
        8. If all 9 mutation flags = 0 AND PRS < 0 → has_seizure forced to 0
           (lower genetic risk → less likely to have seizures in recording)
        9. If SCN1A=1 OR SCN8A=1 OR KCNQ2=1 → don't force has_seizure=0
    """
    df = df.copy()

    # 1. Mutation flags → binary
    for col in MUTATION_COLUMNS:
        if col in df.columns:
            df[col] = df[col].round().clip(0, 1).astype(int)

    # 2. Freeze pLI to known constants
    for col, val in KNOWN_PLI.items():
        if col in df.columns:
            df[col] = val

    # 3. Clip PRS
    if "polygenic_risk_score" in df.columns:
        df["polygenic_risk_score"] = df["polygenic_risk_score"].clip(-5, 5)

    # 4. Preictal ratio ∈ [0, 1]
    if "preictal_ratio" in df.columns:
        df["preictal_ratio"] = df["preictal_ratio"].clip(0, 1)

    # 5–6. Non-negative features
    power_cols = [c for c in df.columns if "power_mean" in c or "variance" in c]
    for col in power_cols:
        df[col] = df[col].clip(lower=0)

    mobility_cols = [c for c in df.columns if "mobility" in c or "complexity" in c]
    for col in mobility_cols:
        df[col] = df[col].clip(lower=0)

    # Entropy ≥ 0
    entropy_cols = [c for c in df.columns if "entropy" in c]
    for col in entropy_cols:
        df[col] = df[col].clip(lower=0)

    # Spike rate ≥ 0
    spike_cols = [c for c in df.columns if "spike" in c]
    for col in spike_cols:
        df[col] = df[col].clip(lower=0)

    # n_valid_epochs → positive integer
    if "n_valid_epochs" in df.columns:
        df["n_valid_epochs"] = df["n_valid_epochs"].clip(lower=100).round().astype(int)

    # 7. has_seizure → binary
    if "has_seizure" in df.columns:
        df["has_seizure"] = df["has_seizure"].round().clip(0, 1).astype(int)

    # 8–9. Genetic → seizure consistency
    if "has_seizure" in df.columns:
        all_mutations_zero = True
        for col in MUTATION_COLUMNS:
            if col in df.columns:
                all_mutations_zero = all_mutations_zero & (df[col] == 0)

        low_genetic_risk = all_mutations_zero & (df.get("polygenic_risk_score", 0) < -1)
        # Patients with very low genetic risk AND no mutations: reduce seizure likelihood
        df.loc[low_genetic_risk, "has_seizure"] = 0
        df.loc[low_genetic_risk, "preictal_ratio"] = df.loc[low_genetic_risk, "preictal_ratio"].clip(upper=0.05)

    n_constrained = df.shape[0]
    print(f"  Applied biological constraints to {n_constrained} records")
    print(f"  has_seizure distribution: {df['has_seizure'].value_counts().to_dict()}")
    for col in MUTATION_COLUMNS:
        if col in df.columns and df[col].sum() > 0:
            print(f"  {col}: {int(df[col].sum())} / {len(df)}")

    return df


# ============================================================
# 4. Validate Quality
# ============================================================
def validate_synthetic(real_df, synthetic_df, output_dir=None):
    """
    Validate synthetic data quality via:
      1. Column-wise Kolmogorov–Smirnov test (continuous columns)
      2. Correlation matrix preservation
      3. Summary statistics comparison

    Returns:
        dict with validation metrics
    """
    # Drop non-numeric / metadata
    drop_cols = ["patient_id", "file"]
    real = real_df.drop(columns=drop_cols, errors="ignore")
    syn = synthetic_df.copy()

    # Align columns
    common_cols = [c for c in real.columns if c in syn.columns]
    real = real[common_cols]
    syn = syn[common_cols]

    continuous_cols = [c for c in common_cols if c not in MUTATION_COLUMNS + ["has_seizure"]]

    # 1. KS tests
    ks_results = {}
    n_pass = 0
    for col in continuous_cols:
        stat, p_val = ks_2samp(real[col].dropna(), syn[col].dropna())
        ks_results[col] = {"statistic": round(stat, 4), "p_value": round(p_val, 4)}
        if p_val > 0.05:
            n_pass += 1

    ks_pass_rate = n_pass / len(continuous_cols) if continuous_cols else 0

    # 2. Correlation preservation
    real_corr = real[continuous_cols].corr()
    syn_corr = syn[continuous_cols].corr()
    corr_diff = (real_corr - syn_corr).abs()
    mean_corr_diff = float(corr_diff.mean().mean())

    # 3. Summary statistics
    summary = {}
    for col in continuous_cols:
        summary[col] = {
            "real_mean": round(float(real[col].mean()), 4),
            "syn_mean": round(float(syn[col].mean()), 4),
            "real_std": round(float(real[col].std()), 4),
            "syn_std": round(float(syn[col].std()), 4),
        }

    results = {
        "n_real": len(real),
        "n_synthetic": len(syn),
        "n_columns": len(common_cols),
        "ks_pass_rate": round(ks_pass_rate, 3),
        "n_ks_pass": n_pass,
        "n_ks_total": len(continuous_cols),
        "mean_correlation_diff": round(mean_corr_diff, 4),
        "ks_tests": ks_results,
    }

    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"  KS test pass rate (p > 0.05): {n_pass}/{len(continuous_cols)} ({ks_pass_rate:.1%})")
    print(f"  Mean correlation difference: {mean_corr_diff:.4f}")
    print(f"  (Lower is better — 0 = identical correlations)")

    # Print worst KS columns
    worst = sorted(ks_results.items(), key=lambda x: x[1]["p_value"])[:5]
    print(f"\n  Bottom 5 KS p-values (hardest to replicate):")
    for col, vals in worst:
        print(f"    {col}: D={vals['statistic']}, p={vals['p_value']}")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "validation_report.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Full report saved to {output_dir / 'validation_report.json'}")

    return results


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="CTGAN Synthetic Data Generation")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Training epochs (default: from config)")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Synthetic samples to generate (default: from config)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    ctgan_cfg = config["ctgan"]

    n_samples = args.n_samples or ctgan_cfg["num_synthetic_samples"]
    output_dir = Path(args.output_dir or config["paths"]["data"]["processed"]["synthetic"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CTGAN SYNTHETIC DATA GENERATION")
    print("=" * 60)

    # Step 1: Build summary dataset (with log-transform applied)
    summary_df, log_cols = build_summary_dataset(config)
    summary_df.to_csv(output_dir / "real_summary_dataset.csv", index=False)
    print(f"  Saved real summary to {output_dir / 'real_summary_dataset.csv'}")

    # Step 2: Train CTGAN (with StandardScaler applied)
    model, columns, discrete_cols, scaler, continuous_cols = train_ctgan(
        summary_df, config, epochs=args.epochs
    )

    # Save model
    model.save(str(output_dir / "ctgan_model.pkl"))
    print(f"  Model saved to {output_dir / 'ctgan_model.pkl'}")

    # Step 3: Generate synthetic data (inverse-scaled and inverse-logged)
    synthetic_df = generate_synthetic(
        model, n_samples, columns, discrete_cols,
        scaler=scaler, continuous_cols=continuous_cols, log_cols=log_cols,
    )

    # Step 4: Apply biological constraints
    synthetic_df = apply_biological_constraints(synthetic_df)

    # Step 5: Save synthetic data
    synthetic_df.to_csv(output_dir / "synthetic_records.csv", index=False)
    print(f"  Saved {len(synthetic_df)} synthetic records to {output_dir / 'synthetic_records.csv'}")

    # Step 6: Validate — un-log the real data first so comparison is in original scale
    real_for_val = summary_df.copy()
    for col in log_cols:
        if col in real_for_val.columns:
            real_for_val[col] = np.power(10, real_for_val[col])
    validate_synthetic(real_for_val, synthetic_df, output_dir=output_dir)

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    print(f"  Real dataset:      {output_dir / 'real_summary_dataset.csv'}")
    print(f"  Synthetic dataset: {output_dir / 'synthetic_records.csv'}")
    print(f"  CTGAN model:       {output_dir / 'ctgan_model.pkl'}")
    print(f"  Validation report: {output_dir / 'validation_report.json'}")


if __name__ == "__main__":
    main()
