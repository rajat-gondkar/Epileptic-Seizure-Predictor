#!/usr/bin/env python3
"""
CTGAN Synthetic Data Generator — EEG + Genetic Fusion (v3)
==========================================================
Trains CTGAN on the COMBINED EEG+genetic data to preserve
cross-modal relationships (e.g., seizure EEG ↔ high mutation burden).

Key fix: Clips bounded features after generation instead of
separating them.

Usage:
    python seizure_prediction/generate_ctgan_data.py --n-synthetic 5000 --seed 42
"""

import argparse
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats, signal
from scipy.signal import welch

warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# Constants
# ============================================================
TARGET_GENES = [
    'SCN1A', 'SCN8A', 'KCNQ2', 'SCN2A', 'KCNT1',
    'DEPDC5', 'PCDH19', 'GRIN2A', 'GABRA1'
]

GENE_RISK_WEIGHTS = {
    'SCN1A': 1.00, 'KCNQ2': 0.95, 'SCN2A': 0.90,
    'SCN8A': 0.85, 'KCNT1': 0.80, 'DEPDC5': 0.65,
    'GRIN2A': 0.55, 'GABRA1': 0.50, 'PCDH19': 0.40,
}

CARRIER_FREQS = {
    'SCN1A': 0.10, 'KCNQ2': 0.08, 'SCN2A': 0.07,
    'SCN8A': 0.06, 'KCNT1': 0.05, 'DEPDC5': 0.05,
    'PCDH19': 0.04, 'GRIN2A': 0.04, 'GABRA1': 0.03,
}

ION_CHANNEL_GENES = ['SCN1A', 'SCN2A', 'SCN8A', 'KCNQ2', 'KCNT1', 'GRIN2A']
TIER1_GENES = ['SCN1A', 'KCNQ2', 'SCN2A']

PLI_SCORES = {
    'SCN1A': 1.0, 'SCN8A': 1.0, 'KCNQ2': 0.99998,
    'SCN2A': 1.0, 'KCNT1': 0.00003, 'DEPDC5': 0.1165,
    'PCDH19': 0.99975, 'GRIN2A': 0.99998, 'GABRA1': 0.91489,
}

FREQ_BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 12.0),
    'beta': (12.0, 30.0),
    'gamma': (30.0, 70.0),
}

# Feature bounds for post-generation clipping
FEATURE_BOUNDS = {
    'seizure_ratio': (0.0, 1.0),
    'preictal_ratio': (0.0, 1.0),
    'mean_delta_power': (0.0, 1.0),
    'mean_theta_power': (0.0, 1.0),
    'mean_alpha_power': (0.0, 1.0),
    'mean_beta_power': (0.0, 1.0),
    'mean_gamma_power': (0.0, 1.0),
    'mutation_burden': (0.0, None),
    'ion_channel_burden': (0.0, None),
    'SCN1A_pLI': (0.0, 1.0),
    'SCN8A_pLI': (0.0, 1.0),
}

# Binary features → round to 0/1
BINARY_FEATURES = ['has_seizure', 'tier1_flag']

# Discrete features → round to nearest valid value
DISCRETE_FEATURES = {
    'weighted_mutation_0': GENE_RISK_WEIGHTS['SCN1A'],
    'weighted_mutation_1': GENE_RISK_WEIGHTS['SCN8A'],
    'weighted_mutation_2': GENE_RISK_WEIGHTS['KCNQ2'],
    'weighted_mutation_3': GENE_RISK_WEIGHTS['SCN2A'],
    'weighted_mutation_4': GENE_RISK_WEIGHTS['KCNT1'],
    'weighted_mutation_5': GENE_RISK_WEIGHTS['DEPDC5'],
    'weighted_mutation_6': GENE_RISK_WEIGHTS['PCDH19'],
    'weighted_mutation_7': GENE_RISK_WEIGHTS['GRIN2A'],
    'weighted_mutation_8': GENE_RISK_WEIGHTS['GABRA1'],
}


# ============================================================
# EEG Feature Extraction
# ============================================================
def compute_eeg_features(eeg_data, sampling_rate=256):
    """Extract compact EEG features."""
    n_channels, n_timepoints = eeg_data.shape
    features = {}

    features['global_mean'] = np.mean(eeg_data)
    features['global_std'] = np.std(eeg_data)
    features['global_max'] = np.max(eeg_data)
    features['global_min'] = np.min(eeg_data)
    features['global_range'] = features['global_max'] - features['global_min']
    features['global_rms'] = np.sqrt(np.mean(eeg_data**2))

    all_means, all_stds = [], []
    all_hjorth_act, all_hjorth_mob, all_hjorth_cpx = [], [], []
    band_power_totals = {b: [] for b in FREQ_BANDS}
    spectral_entropies = []
    nperseg = min(256, n_timepoints)

    for ch in range(n_channels):
        ch_data = eeg_data[ch, :]
        all_means.append(np.mean(ch_data))
        all_stds.append(np.std(ch_data))

        diff1 = np.diff(ch_data)
        diff2 = np.diff(diff1)
        activity = np.var(ch_data)
        mobility = np.sqrt(np.var(diff1) / activity) if activity > 0 else 0
        complexity = (np.sqrt(np.var(diff2) / np.var(diff1)) / mobility
                     if mobility > 0 else 0)
        all_hjorth_act.append(activity)
        all_hjorth_mob.append(mobility)
        all_hjorth_cpx.append(complexity)

        freqs, psd = welch(ch_data, fs=sampling_rate, nperseg=nperseg)
        total_power = np.trapz(psd, freqs)
        for band_name, (low, high) in FREQ_BANDS.items():
            mask = (freqs >= low) & (freqs <= high)
            bp = np.trapz(psd[mask], freqs[mask])
            band_power_totals[band_name].append(bp / (total_power + 1e-10))

        psd_norm = psd / (np.sum(psd) + 1e-10)
        spectral_entropies.append(-np.sum(psd_norm * np.log2(psd_norm + 1e-10)))

    features['mean_amplitude'] = np.mean(all_means)
    features['std_amplitude'] = np.mean(all_stds)
    features['mean_hjorth_activity'] = np.mean(all_hjorth_act)
    features['mean_hjorth_mobility'] = np.mean(all_hjorth_mob)
    features['mean_hjorth_complexity'] = np.mean(all_hjorth_cpx)

    for band_name in FREQ_BANDS:
        features[f'mean_{band_name}_power'] = np.mean(band_power_totals[band_name])

    features['mean_spectral_entropy'] = np.mean(spectral_entropies)

    if n_channels > 1:
        corr_matrix = np.corrcoef(eeg_data)
        upper_tri = corr_matrix[np.triu_indices(n_channels, k=1)]
        features['mean_channel_corr'] = np.mean(upper_tri)
        features['std_channel_corr'] = np.std(upper_tri)

    return features


def extract_eeg_features_for_file(edf_path, sampling_rate=256):
    """Extract EEG features from a single EDF file."""
    try:
        import pyedflib
        with pyedflib.EdfReader(edf_path) as reader:
            n_channels = reader.signals_in_file
            n_samples = reader.getNSamples()[0]
            eeg_data = np.zeros((n_channels, n_samples))
            for ch in range(n_channels):
                eeg_data[ch, :] = reader.readSignal(ch)
        return compute_eeg_features(eeg_data, sampling_rate)
    except Exception as e:
        print(f"  Warning: Failed to extract features from {edf_path}: {e}")
        return None


# ============================================================
# Annotation Parsing
# ============================================================
def parse_annotation_file(edf_path):
    """Parse .annotation.txt for seizure times."""
    annotation_path = str(edf_path) + '.annotation.txt'
    if not os.path.exists(annotation_path):
        return False, 0.0, 0.0
    try:
        seizures = []
        with open(annotation_path, 'r') as f:
            for line in f.readlines()[1:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) >= 2:
                    seizures.append((float(parts[0]), float(parts[0]) + float(parts[1])))
        if not seizures:
            return False, 0.0, 0.0
        total_duration = 3600.0
        total_seizure_time = sum(end - start for start, end in seizures)
        seizure_ratio = total_seizure_time / total_duration
        preictal_ratio = min(0.5, (len(seizures) * 1800) / total_duration)
        return True, seizure_ratio, preictal_ratio
    except Exception:
        return False, 0.0, 0.0


# ============================================================
# Genetic Profile Generation (16-dim)
# ============================================================
def build_genetic_profile_16d(patient_id, rng):
    """Build 16-dim genetic vector for a patient."""
    binary_flags = {}
    for gene in TARGET_GENES:
        freq = CARRIER_FREQS.get(gene, 0.05)
        binary_flags[gene] = int(rng.random() < freq)

    weighted = [GENE_RISK_WEIGHTS[g] * binary_flags[g] for g in TARGET_GENES]

    gwas_path = PROJECT_ROOT / 'data' / 'raw' / 'gwas' / 'epilepsy_snps.csv'
    prs = 0.0
    if gwas_path.exists():
        gwas_df = pd.read_csv(gwas_path)
        for _, snp in gwas_df.iterrows():
            beta = np.log(snp['or_beta']) if snp['or_beta'] > 0 else 0
            freq = snp.get('risk_allele_freq', 0.3)
            prob_0 = (1 - freq) ** 2
            prob_1 = 2 * freq * (1 - freq)
            r = rng.random()
            dosage = 0 if r < prob_0 else (1 if r < prob_0 + prob_1 else 2)
            prs += beta * dosage
    else:
        prs = rng.normal(0, 1)

    mutation_burden = sum(GENE_RISK_WEIGHTS[g] * binary_flags[g] for g in TARGET_GENES)
    ion_channel_burden = sum(GENE_RISK_WEIGHTS[g] * binary_flags[g] for g in ION_CHANNEL_GENES)
    tier1_flag = int(any(binary_flags[g] == 1 for g in TIER1_GENES))
    scn1a_proxy = float(PLI_SCORES['SCN1A'] * binary_flags['SCN1A'])

    return {
        'weighted_mutation_flags': weighted,
        'SCN1A_pLI': PLI_SCORES['SCN1A'],
        'SCN8A_pLI': PLI_SCORES['SCN8A'],
        'PRS': prs,
        'mutation_burden': mutation_burden,
        'ion_channel_burden': ion_channel_burden,
        'tier1_flag': tier1_flag,
        'SCN1A_severity_proxy': scn1a_proxy,
    }


# ============================================================
# Post-Generation Clipping
# ============================================================
def clip_synthetic_data(df):
    """Clip bounded features and round discrete features."""
    df = df.copy()

    # Clip bounded features
    for col, (lo, hi) in FEATURE_BOUNDS.items():
        if col in df.columns:
            if lo is not None:
                df[col] = df[col].clip(lower=lo)
            if hi is not None:
                df[col] = df[col].clip(upper=hi)

    # Round binary features
    for col in BINARY_FEATURES:
        if col in df.columns:
            df[col] = (df[col] > 0.5).astype(int)

    # Snap discrete features to valid values (0 or weight)
    for col, weight in DISCRETE_FEATURES.items():
        if col in df.columns:
            # Snap to nearest valid value: 0 or weight
            df[col] = df[col].apply(lambda x: weight if x > weight / 2 else 0.0)

    return df


# ============================================================
# Data Loading
# ============================================================
def load_eeg_file_list(csv_path):
    """Load EDF file paths from CSV."""
    df = pd.read_csv(csv_path)
    file_list = []
    for _, row in df.iterrows():
        edf_path = row['filename']
        match = re.search(r'(chb\d+)', edf_path)
        patient_id = match.group(1) if match else 'unknown'
        file_list.append((patient_id, edf_path))
    return file_list


# ============================================================
# Validation
# ============================================================
def validate_results(real_df, synthetic_df, output_dir):
    """Validate combined EEG+genetic synthetic data."""
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)

    report = {'n_real': len(real_df), 'n_synthetic': len(synthetic_df)}

    # 1. KS tests
    print("\n--- KS Tests (per feature) ---")
    ks_pass = 0
    ks_total = 0
    ks_results = {}

    for col in real_df.columns:
        if real_df[col].dtype in ['float64', 'float32', 'int64']:
            stat, pval = stats.ks_2samp(
                real_df[col].dropna().values,
                synthetic_df[col].dropna().values
            )
            passed = bool(pval > 0.05)
            if passed:
                ks_pass += 1
            ks_total += 1
            ks_results[col] = {
                'KS_stat': float(stat),
                'p_value': float(pval),
                'pass': passed,
                'real_mean': float(real_df[col].mean()),
                'synth_mean': float(synthetic_df[col].mean()),
            }
            status = '✓' if passed else '✗'
            print(f"  {status} {col}: KS={stat:.4f}, p={pval:.4f}")

    ks_rate = ks_pass / max(ks_total, 1)
    report['ks_tests'] = ks_results
    report['ks_pass_rate'] = ks_rate
    print(f"\nKS Pass Rate: {ks_rate*100:.1f}% ({ks_pass}/{ks_total})")

    # 2. Correlation preservation
    print("\n--- Correlation Preservation ---")
    num_cols = real_df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 1:
        real_corr = real_df[num_cols].corr().values
        synth_corr = synthetic_df[num_cols].corr().values
        mask = ~(np.isnan(real_corr) | np.isnan(synth_corr))
        if mask.any():
            corr_diff = float(np.abs(real_corr[mask] - synth_corr[mask]).mean())
            corr_corr = float(np.corrcoef(real_corr[mask], synth_corr[mask])[0, 1])
        else:
            corr_diff = 0.0
            corr_corr = 0.0
        print(f"  Mean Absolute Diff: {corr_diff:.4f}")
        print(f"  Correlation of Correlations: {corr_corr:.4f}")
        report['correlation_metrics'] = {'diff': corr_diff, 'corr': corr_corr}

    # 3. Sanity checks
    print("\n--- Sanity Checks ---")
    checks = {}

    for col in [f'weighted_mutation_{i}' for i in range(9)]:
        if col in synthetic_df.columns:
            neg = (synthetic_df[col] < 0).any()
            if neg:
                print(f"  ✗ {col} has negative values!")
            checks[f'{col}_non_negative'] = not neg

    for col in ['mutation_burden', 'ion_channel_burden']:
        if col in synthetic_df.columns:
            neg = (synthetic_df[col] < 0).any()
            if neg:
                print(f"  ✗ {col} has negative values!")
            checks[f'{col}_non_negative'] = not neg

    for col in ['seizure_ratio', 'preictal_ratio']:
        if col in synthetic_df.columns:
            invalid = (synthetic_df[col] < 0).any() or (synthetic_df[col] > 1).any()
            if invalid:
                print(f"  ✗ {col} out of [0,1] range!")
            checks[f'{col}_valid'] = not invalid

    for col in ['SCN1A_pLI', 'SCN8A_pLI']:
        if col in synthetic_df.columns:
            wrong = (synthetic_df[col] != 1.0).any()
            if wrong:
                print(f"  ✗ {col} is not constant 1.0!")
            checks[f'{col}_constant'] = not wrong

    for col in BINARY_FEATURES:
        if col in synthetic_df.columns:
            invalid = not synthetic_df[col].isin([0, 1]).all()
            if invalid:
                print(f"  ✗ {col} has values other than 0/1!")
            checks[f'{col}_binary'] = not invalid

    report['sanity_checks'] = checks
    all_pass = all(checks.values())
    print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")

    # Save report
    report_path = output_dir / 'validation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to: {report_path}")

    return report


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="CTGAN EEG+Genetic Synthetic Data (v3)")
    parser.add_argument('--train-csv', type=str,
                       default='seizure_prediction/DataCSVs/CHB-MIT/all_patients_train.csv')
    parser.add_argument('--n-synthetic', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='data/processed/synthetic')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()

    print("="*60)
    print("CTGAN SYNTHETIC DATA GENERATOR (v3)")
    print("Joint CTGAN on EEG + Genetic features")
    print("="*60)

    # Load config
    config_path = PROJECT_ROOT / args.config
    config = yaml.safe_load(open(config_path)) if config_path.exists() else {}
    ctgan_config = config.get('ctgan', {})

    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Step 1: Extract REAL combined features (EEG + Genetic per file)
    # ================================================================
    print("\n" + "="*60)
    print("STEP 1: EXTRACTING REAL COMBINED FEATURES")
    print("="*60)

    train_csv_path = PROJECT_ROOT / args.train_csv
    file_list = load_eeg_file_list(train_csv_path)
    print(f"Loaded {len(file_list)} EDF files")

    sampling_rate = config.get('eeg', {}).get('sampling_rate', 256)

    # Group files by patient for genetic profiles
    patient_files = {}
    for patient_id, edf_path in file_list:
        if patient_id not in patient_files:
            patient_files[patient_id] = []
        patient_files[patient_id].append(edf_path)

    # Generate genetic profiles per patient (same for all files of one patient)
    patient_genetics = {}
    for patient_id in patient_files:
        patient_genetics[patient_id] = build_genetic_profile_16d(patient_id, rng)

    # Extract features for ALL files with their patient's genetic profile
    real_records = []
    for patient_id, edf_paths in patient_files.items():
        genetics = patient_genetics[patient_id]

        for edf_path in edf_paths:
            # EEG features
            eeg_features = extract_eeg_features_for_file(edf_path, sampling_rate)
            if eeg_features is None:
                continue

            # Annotation
            has_seizure, seizure_ratio, preictal_ratio = parse_annotation_file(edf_path)

            # Combine: EEG + Annotation + Genetic
            record = {
                'has_seizure': int(has_seizure),
                'seizure_ratio': seizure_ratio,
                'preictal_ratio': preictal_ratio,
            }
            record.update(eeg_features)

            # Genetic features (same for all files of this patient)
            for i, val in enumerate(genetics['weighted_mutation_flags']):
                record[f'weighted_mutation_{i}'] = val
            record['SCN1A_pLI'] = genetics['SCN1A_pLI']
            record['SCN8A_pLI'] = genetics['SCN8A_pLI']
            record['PRS'] = genetics['PRS']
            record['mutation_burden'] = genetics['mutation_burden']
            record['ion_channel_burden'] = genetics['ion_channel_burden']
            record['tier1_flag'] = genetics['tier1_flag']
            record['SCN1A_severity_proxy'] = genetics['SCN1A_severity_proxy']

            real_records.append(record)

    real_df = pd.DataFrame(real_records)
    print(f"\nReal combined dataset: {real_df.shape}")
    print(f"Features: {list(real_df.columns)}")

    # Save real data
    real_path = output_dir / 'real_eeg_genetic_features.csv'
    real_df.to_csv(real_path, index=False)
    print(f"Saved: {real_path}")

    # ================================================================
    # Step 2: Train CTGAN on COMBINED data
    # ================================================================
    print("\n" + "="*60)
    print("STEP 2: TRAINING CTGAN ON COMBINED DATA")
    print("="*60)

    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata

    metadata = SingleTableMetadata()
    for col in real_df.columns:
        if real_df[col].dtype in ['int64', 'int32']:
            metadata.add_column(col, sdtype='categorical')
        else:
            metadata.add_column(col, sdtype='numerical')

    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=ctgan_config.get('epochs', 500),
        batch_size=ctgan_config.get('batch_size', 100),
        generator_dim=(128, 128),
        discriminator_dim=(128, 128),
        verbose=True,
    )

    start_time = time.time()
    synthesizer.fit(real_df)
    training_time = time.time() - start_time
    print(f"\nCTGAN training completed in {training_time:.1f}s")

    # ================================================================
    # Step 3: Generate + Clip
    # ================================================================
    print("\n" + "="*60)
    print("STEP 3: GENERATING SYNTHETIC DATA")
    print("="*60)

    synthetic_df = synthesizer.sample(num_rows=args.n_synthetic)

    # Clip bounded features
    synthetic_df = clip_synthetic_data(synthetic_df)

    print(f"Generated {len(synthetic_df)} synthetic samples")

    # Save synthetic data
    synthetic_path = output_dir / 'synthetic_eeg_genetic_data.csv'
    synthetic_df.to_csv(synthetic_path, index=False)
    print(f"Saved: {synthetic_path}")

    # ================================================================
    # Step 4: Validate
    # ================================================================
    validate_results(real_df, synthetic_df, output_dir)

    # ================================================================
    # Summary
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  Real combined:     {real_path}")
    print(f"  Synthetic combined: {synthetic_path}")
    print(f"  Validation report:  {output_dir / 'validation_report.json'}")


if __name__ == '__main__':
    main()
