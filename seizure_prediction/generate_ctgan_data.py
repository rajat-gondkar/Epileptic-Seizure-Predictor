#!/usr/bin/env python3
"""
CTGAN Synthetic Data Generator — EEG + Genetic Fusion
=====================================================
Generates synthetic multimodal training data by combining EEG features
extracted from CHB-MIT EDF files with simulated 16-dimensional genetic profiles.

Pipeline:
  1. Read EDF files → extract per-file EEG summary features
  2. Read annotation files → determine seizure/preictal/interictal ratios
  3. Construct 16-dim genetic profiles per patient
  4. Train CTGAN on the combined EEG+genetic tabular dataset
  5. Validate quality (KS tests, correlation preservation)
  6. Output synthetic CSV + validation report

Usage:
    python seizure_prediction/generate_ctgan_data.py
    python seizure_prediction/generate_ctgan_data.py --n-synthetic 5000 --seed 42

Requirements:
    - pyedflib, scipy, numpy, pandas, sdv (CTGAN), scikit-learn, pyyaml
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

# NumPy 2.0+ compatibility: np.trapz renamed to np.trapezoid
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

warnings.filterwarnings('ignore')

# -- Project root --
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Configuration
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

# pLI scores from gnomAD (loaded from file at runtime)
PLI_SCORES = {}  # Will be populated from data/raw/gnomad/pli_scores.csv

# EEG frequency bands (Hz)
FREQ_BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 12.0),
    'beta': (12.0, 30.0),
    'gamma': (30.0, 70.0),
}

# EEG channels to use (19 common channels from CHB-MIT)
EEG_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ',
]


# ============================================================
# EEG Feature Extraction
# ============================================================
def compute_eeg_features(eeg_data, sampling_rate=256):
    """
    Extract comprehensive EEG features from raw signal data.
    
    Args:
        eeg_data: np.array of shape (n_channels, n_timepoints)
        sampling_rate: Sampling frequency in Hz
    
    Returns:
        dict of feature_name -> value
    """
    n_channels, n_timepoints = eeg_data.shape
    features = {}
    
    # 1. Time-domain features per channel
    for ch_idx in range(n_channels):
        ch_data = eeg_data[ch_idx, :]
        
        # Basic statistics
        features[f'mean_ch{ch_idx}'] = np.mean(ch_data)
        features[f'std_ch{ch_idx}'] = np.std(ch_data)
        features[f'skew_ch{ch_idx}'] = float(stats.skew(ch_data))
        features[f'kurtosis_ch{ch_idx}'] = float(stats.kurtosis(ch_data))
        features[f'min_ch{ch_idx}'] = np.min(ch_data)
        features[f'max_ch{ch_idx}'] = np.max(ch_data)
        features[f'range_ch{ch_idx}'] = np.max(ch_data) - np.min(ch_data)
        features[f'rms_ch{ch_idx}'] = np.sqrt(np.mean(ch_data**2))
        features[f'peak_to_peak_ch{ch_idx}'] = np.ptp(ch_data)
        
        # Hjorth parameters
        diff1 = np.diff(ch_data)
        diff2 = np.diff(diff1)
        
        activity = np.var(ch_data)
        mobility = np.sqrt(np.var(diff1) / activity) if activity > 0 else 0
        complexity = (np.sqrt(np.var(diff2) / np.var(diff1)) / mobility 
                     if mobility > 0 else 0)
        
        features[f'hjorth_activity_ch{ch_idx}'] = activity
        features[f'hjorth_mobility_ch{ch_idx}'] = mobility
        features[f'hjorth_complexity_ch{ch_idx}'] = complexity
    
    # 2. Frequency-domain features
    for ch_idx in range(n_channels):
        ch_data = eeg_data[ch_idx, :]
        
        # Compute PSD using Welch's method
        nperseg = min(256, n_timepoints)
        freqs, psd = welch(ch_data, fs=sampling_rate, nperseg=nperseg)
        
        # Total power
        total_power = np.trapz(psd, freqs)
        features[f'total_power_ch{ch_idx}'] = total_power
        
        # Band power for each frequency band
        for band_name, (low, high) in FREQ_BANDS.items():
            band_mask = (freqs >= low) & (freqs <= high)
            band_power = np.trapz(psd[band_mask], freqs[band_mask])
            features[f'{band_name}_power_ch{ch_idx}'] = band_power
            
            # Relative band power
            if total_power > 0:
                features[f'{band_name}_rel_power_ch{ch_idx}'] = band_power / total_power
            else:
                features[f'{band_name}_rel_power_ch{ch_idx}'] = 0
        
        # Spectral entropy
        psd_norm = psd / (np.sum(psd) + 1e-10)
        features[f'spectral_entropy_ch{ch_idx}'] = -np.sum(
            psd_norm * np.log2(psd_norm + 1e-10)
        )
        
        # Peak frequency
        features[f'peak_freq_ch{ch_idx}'] = freqs[np.argmax(psd)]
        
        # Median frequency
        cumulative_power = np.cumsum(psd)
        median_idx = np.searchsorted(cumulative_power, cumulative_power[-1] / 2)
        features[f'median_freq_ch{ch_idx}'] = freqs[min(median_idx, len(freqs)-1)]
    
    # 3. Cross-channel features
    if n_channels > 1:
        # Correlation matrix summary
        corr_matrix = np.corrcoef(eeg_data)
        upper_tri = corr_matrix[np.triu_indices(n_channels, k=1)]
        
        features['mean_channel_corr'] = np.mean(upper_tri)
        features['std_channel_corr'] = np.std(upper_tri)
        features['max_channel_corr'] = np.max(upper_tri)
        features['min_channel_corr'] = np.min(upper_tri)
        
        # Coherence between specific channel pairs
        for i, j in [(0, 8), (1, 9), (2, 10), (3, 11)]:  # Left-Right pairs
            if i < n_channels and j < n_channels:
                f, coh = signal.coherence(
                    eeg_data[i, :], eeg_data[j, :], 
                    fs=sampling_rate, nperseg=min(128, n_timepoints)
                )
                features[f'coherence_{i}_{j}'] = np.mean(coh)
    
    return features


def compute_summary_statistics(eeg_data, sampling_rate=256):
    """
    Compute compact EEG summary statistics.
    Uses ~20 features instead of per-channel features for CTGAN compatibility.
    """
    n_channels, n_timepoints = eeg_data.shape
    features = {}
    
    # Global time-domain features
    features['global_mean'] = np.mean(eeg_data)
    features['global_std'] = np.std(eeg_data)
    features['global_max'] = np.max(eeg_data)
    features['global_min'] = np.min(eeg_data)
    features['global_range'] = features['global_max'] - features['global_min']
    features['global_rms'] = np.sqrt(np.mean(eeg_data**2))
    
    # Compute features across ALL channels then average
    all_means = []
    all_stds = []
    all_hjorth_activity = []
    all_hjorth_mobility = []
    all_hjorth_complexity = []
    
    band_power_totals = {band: [] for band in FREQ_BANDS}
    spectral_entropies = []
    
    nperseg = min(256, n_timepoints)
    
    for ch_idx in range(n_channels):
        ch_data = eeg_data[ch_idx, :]
        
        all_means.append(np.mean(ch_data))
        all_stds.append(np.std(ch_data))
        
        # Hjorth parameters
        diff1 = np.diff(ch_data)
        diff2 = np.diff(diff1)
        activity = np.var(ch_data)
        mobility = np.sqrt(np.var(diff1) / activity) if activity > 0 else 0
        complexity = (np.sqrt(np.var(diff2) / np.var(diff1)) / mobility 
                     if mobility > 0 else 0)
        
        all_hjorth_activity.append(activity)
        all_hjorth_mobility.append(mobility)
        all_hjorth_complexity.append(complexity)
        
        # PSD
        freqs, psd = welch(ch_data, fs=sampling_rate, nperseg=nperseg)
        total_power = np.trapz(psd, freqs)
        
        for band_name, (low, high) in FREQ_BANDS.items():
            band_mask = (freqs >= low) & (freqs <= high)
            band_power = np.trapz(psd[band_mask], freqs[band_mask])
            band_power_totals[band_name].append(band_power / (total_power + 1e-10))
        
        psd_norm = psd / (np.sum(psd) + 1e-10)
        spectral_entropies.append(-np.sum(psd_norm * np.log2(psd_norm + 1e-10)))
    
    # Average across channels
    features['mean_amplitude'] = np.mean(all_means)
    features['std_amplitude'] = np.mean(all_stds)
    features['mean_hjorth_activity'] = np.mean(all_hjorth_activity)
    features['mean_hjorth_mobility'] = np.mean(all_hjorth_mobility)
    features['mean_hjorth_complexity'] = np.mean(all_hjorth_complexity)
    
    for band_name in FREQ_BANDS:
        features[f'mean_{band_name}_power'] = np.mean(band_power_totals[band_name])
    
    features['mean_spectral_entropy'] = np.mean(spectral_entropies)
    
    # Cross-channel correlation summary
    if n_channels > 1:
        corr_matrix = np.corrcoef(eeg_data)
        upper_tri = corr_matrix[np.triu_indices(n_channels, k=1)]
        features['mean_channel_corr'] = np.mean(upper_tri)
        features['std_channel_corr'] = np.std(upper_tri)
    
    return features


# ============================================================
# Annotation Parsing
# ============================================================
def parse_annotation_file(edf_path, sampling_rate=256):
    """
    Parse .annotation.txt file for an EDF to get seizure times.
    
    Returns:
        has_seizure: bool
        seizure_ratio: float (fraction of file that is ictal)
        preictal_ratio: float (estimated preictal fraction)
    """
    annotation_path = str(edf_path) + '.annotation.txt'
    
    if not os.path.exists(annotation_path):
        return False, 0.0, 0.0
    
    try:
        seizures = []
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines[1:]:  # Skip header
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                start_sec = float(parts[0])
                duration_sec = float(parts[1])
                seizures.append((start_sec, start_sec + duration_sec))
        
        if not seizures:
            return False, 0.0, 0.0
        
        # Get EDF duration from filename pattern
        edf_name = os.path.basename(edf_path)
        seq_match = re.search(r'_(\d+)', edf_name)
        if seq_match:
            # Each file is typically 1 hour
            total_duration = 3600.0
        else:
            total_duration = 3600.0
        
        # Calculate seizure ratio
        total_seizure_time = sum(end - start for start, end in seizures)
        seizure_ratio = total_seizure_time / total_duration
        
        # Preictal: 30 min before each seizure
        preictal_window = 1800  # 30 minutes
        preictal_ratio = min(0.5, (len(seizures) * preictal_window) / total_duration)
        
        return True, seizure_ratio, preictal_ratio
        
    except Exception as e:
        print(f"  Warning: Failed to parse annotation for {edf_path}: {e}")
        return False, 0.0, 0.0


# ============================================================
# Data Loading Functions
# ============================================================
def load_pli_scores():
    """Load pLI scores from gnomAD data file."""
    global PLI_SCORES
    pli_path = PROJECT_ROOT / 'data' / 'raw' / 'gnomad' / 'pli_scores.csv'
    
    if pli_path.exists():
        df = pd.read_csv(pli_path)
        for _, row in df.iterrows():
            PLI_SCORES[row['gene']] = float(row['pLI'])
        print(f"Loaded pLI scores for {len(PLI_SCORES)} genes from {pli_path}")
    else:
        # Fallback to hardcoded values
        PLI_SCORES.update({
            'SCN1A': 1.00, 'SCN8A': 1.00, 'KCNQ2': 0.99998,
            'SCN2A': 1.00, 'KCNT1': 0.00003, 'DEPDC5': 0.11650,
            'PCDH19': 0.99975, 'GRIN2A': 0.99998, 'GABRA1': 0.91489,
        })
        print(f"Warning: pLI file not found, using hardcoded values")
    
    return PLI_SCORES


def load_clinvar_data():
    """Load ClinVar epilepsy variants for mutation flag generation."""
    clinvar_path = PROJECT_ROOT / 'data' / 'raw' / 'clinvar' / 'epilepsy_variants.csv'
    
    if clinvar_path.exists():
        df = pd.read_csv(clinvar_path)
        print(f"Loaded {len(df)} ClinVar variants from {clinvar_path}")
        
        # Count pathogenic variants per gene
        gene_counts = {}
        for gene in TARGET_GENES:
            gene_variants = df[df['GeneSymbol'] == gene]
            gene_counts[gene] = len(gene_variants)
        
        print("ClinVar variant counts per gene:")
        for gene, count in gene_counts.items():
            print(f"  {gene}: {count}")
        
        return df, gene_counts
    else:
        print(f"Warning: ClinVar file not found at {clinvar_path}")
        return None, {}


# ============================================================
# Genetic Profile Generation (16-dim)
# ============================================================
def build_genetic_profile_16d(patient_id, rng):
    """
    Build the 16-dimensional genetic feature vector for a patient.
    
    Vector layout:
        [0-8]   Weighted mutation flags (9 genes)
        [9]     SCN1A pLI
        [10]    SCN8A pLI
        [11]    PRS (standardized)
        [12]    Mutation burden
        [13]    Ion channel burden
        [14]    Tier-1 flag
        [15]    SCN1A severity proxy
    """
    # 1. Binary mutation flags
    binary_flags = {}
    for gene in TARGET_GENES:
        freq = CARRIER_FREQS.get(gene, 0.05)
        binary_flags[gene] = int(rng.random() < freq)
    
    # 2. Weighted mutation flags [0-8]
    weighted = []
    for gene in TARGET_GENES:
        weighted.append(GENE_RISK_WEIGHTS[gene] * binary_flags[gene])
    
    # 3. pLI scores [9-10]
    scn1a_pli = PLI_SCORES['SCN1A']
    scn8a_pli = PLI_SCORES['SCN8A']
    
    # 4. PRS [11] — simulated from GWAS data
    gwas_path = PROJECT_ROOT / 'data' / 'raw' / 'gwas' / 'epilepsy_snps.csv'
    if gwas_path.exists():
        gwas_df = pd.read_csv(gwas_path)
        prs = 0.0
        for _, snp in gwas_df.iterrows():
            beta = np.log(snp['or_beta']) if snp['or_beta'] > 0 else 0
            freq = snp.get('risk_allele_freq', 0.3)
            prob_0 = (1 - freq) ** 2
            prob_1 = 2 * freq * (1 - freq)
            r = rng.random()
            if r < prob_0:
                dosage = 0
            elif r < prob_0 + prob_1:
                dosage = 1
            else:
                dosage = 2
            prs += beta * dosage
    else:
        prs = rng.normal(0, 1)
    
    # 5. Engineered features [12-15]
    mutation_burden = sum(GENE_RISK_WEIGHTS[g] * binary_flags[g] for g in TARGET_GENES)
    ion_channel_burden = sum(GENE_RISK_WEIGHTS[g] * binary_flags[g] for g in ION_CHANNEL_GENES)
    tier1_flag = int(any(binary_flags[g] == 1 for g in TIER1_GENES))
    scn1a_proxy = float(scn1a_pli * binary_flags['SCN1A'])
    
    return {
        'weighted_mutation_flags': weighted,
        'SCN1A_pLI': scn1a_pli,
        'SCN8A_pLI': scn8a_pli,
        'PRS': prs,
        'mutation_burden': mutation_burden,
        'ion_channel_burden': ion_channel_burden,
        'tier1_flag': tier1_flag,
        'SCN1A_severity_proxy': scn1a_proxy,
    }


# ============================================================
# Data Loading
# ============================================================
def load_eeg_file_list(csv_path):
    """
    Load list of EDF file paths from a CSV file.
    
    Returns:
        list of (patient_id, edf_path) tuples
    """
    df = pd.read_csv(csv_path)
    file_list = []
    
    for _, row in df.iterrows():
        edf_path = row['filename']
        
        # Extract patient ID from path
        match = re.search(r'(chb\d+)', edf_path)
        if match:
            patient_id = match.group(1)
        else:
            patient_id = 'unknown'
        
        file_list.append((patient_id, edf_path))
    
    return file_list


def extract_eeg_features_for_file(edf_path, sampling_rate=256):
    """
    Extract EEG features from a single EDF file.
    
    Returns:
        dict of features, or None if extraction fails
    """
    try:
        import pyedflib
        
        with pyedflib.EdfReader(edf_path) as reader:
            n_channels = reader.signals_in_file
            n_samples = reader.getNSamples()[0]
            
            # Read all channels
            eeg_data = np.zeros((n_channels, n_samples))
            for ch in range(n_channels):
                eeg_data[ch, :] = reader.readSignal(ch)
        
        # Compute features
        features = compute_summary_statistics(eeg_data, sampling_rate)
        return features
        
    except Exception as e:
        print(f"  Warning: Failed to extract features from {edf_path}: {e}")
        return None


# ============================================================
# CTGAN Training
# ============================================================
def train_ctgan(real_data_df, config):
    """
    Train CTGAN on the combined EEG+genetic tabular data.
    
    Args:
        real_data_df: DataFrame with all features
        config: CTGAN configuration dict
    
    Returns:
        trained CTGAN model
    """
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    
    print("\n" + "="*60)
    print("TRAINING CTGAN")
    print("="*60)
    
    # Create metadata
    metadata = SingleTableMetadata()
    
    # Detect column types
    for col in real_data_df.columns:
        if real_data_df[col].dtype in ['int64', 'int32']:
            metadata.add_column(col, sdtype='categorical')
        elif real_data_df[col].dtype in ['float64', 'float32']:
            metadata.add_column(col, sdtype='numerical')
        else:
            metadata.add_column(col, sdtype='categorical')
    
    # Configure CTGAN - use smaller network for limited real data
    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=config.get('epochs', 500),
        batch_size=config.get('batch_size', 100),
        generator_dim=(128, 128),
        discriminator_dim=(128, 128),
        verbose=True,
    )
    
    # Add constraints
    synthesizer.add_constraints([
        {
            'constraint_class': 'Positive',
            'constraint_parameters': {
                'column_name': 'mutation_burden',
                'low_value': 0,
            }
        },
        {
            'constraint_class': 'Positive',
            'constraint_parameters': {
                'column_name': 'ion_channel_burden',
                'low_value': 0,
            }
        },
    ])
    
    # Train
    start_time = time.time()
    synthesizer.fit(real_data_df)
    training_time = time.time() - start_time
    
    print(f"\nCTGAN training completed in {training_time:.1f}s")
    
    return synthesizer


# ============================================================
# Validation
# ============================================================
def validate_synthetic_data(real_df, synthetic_df, output_dir):
    """
    Validate synthetic data quality using KS tests and correlation analysis.
    
    Returns:
        validation_report: dict with metrics
    """
    print("\n" + "="*60)
    print("VALIDATING SYNTHETIC DATA")
    print("="*60)
    
    report = {
        'n_real': len(real_df),
        'n_synthetic': len(synthetic_df),
        'columns': list(real_df.columns),
        'ks_tests': {},
        'correlation_metrics': {},
        'summary_stats': {},
    }
    
    # 1. KS tests for each numerical column
    ks_pass_count = 0
    ks_total = 0
    
    for col in real_df.columns:
        if real_df[col].dtype in ['float64', 'float32']:
            # KS test
            statistic, p_value = stats.ks_2samp(
                real_df[col].dropna(), 
                synthetic_df[col].dropna()
            )
            
            report['ks_tests'][col] = {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'pass': bool(p_value > 0.05)  # Fail to reject null hypothesis
            }
            
            if p_value > 0.05:
                ks_pass_count += 1
            ks_total += 1
    
    report['ks_pass_rate'] = ks_pass_count / max(ks_total, 1)
    print(f"KS Test Pass Rate: {report['ks_pass_rate']*100:.1f}% ({ks_pass_count}/{ks_total})")
    
    # 2. Correlation preservation
    if len(real_df.columns) > 1:
        real_corr = real_df.corr().values
        synth_corr = synthetic_df.corr().values
        
        # Mask NaN values
        mask = ~(np.isnan(real_corr) | np.isnan(synth_corr))
        if mask.any():
            corr_diff = np.abs(real_corr[mask] - synth_corr[mask]).mean()
            corr_corr = np.corrcoef(real_corr[mask], synth_corr[mask])[0, 1]
        else:
            corr_diff = 0
            corr_corr = 0
        
        report['correlation_metrics'] = {
            'mean_absolute_diff': float(corr_diff),
            'correlation_of_correlations': float(corr_corr),
        }
        print(f"Mean Correlation Diff: {corr_diff:.4f}")
        print(f"Correlation of Correlations: {corr_corr:.4f}")
    
    # 3. Summary statistics comparison
    for col in real_df.columns:
        if real_df[col].dtype in ['float64', 'float32']:
            report['summary_stats'][col] = {
                'real_mean': float(real_df[col].mean()),
                'real_std': float(real_df[col].std()),
                'synth_mean': float(synthetic_df[col].mean()),
                'synth_std': float(synthetic_df[col].std()),
            }
    
    # 4. Genetic feature validation
    genetic_cols = [c for c in real_df.columns if any(g in c for g in TARGET_GENES + ['PRS', 'mutation', 'ion_channel', 'tier1', 'SCN1A'])]
    if genetic_cols:
        print(f"\nGenetic Features ({len(genetic_cols)} columns):")
        for col in genetic_cols[:5]:  # Show first 5
            real_mean = real_df[col].mean()
            synth_mean = synthetic_df[col].mean()
            print(f"  {col}: real={real_mean:.4f}, synth={synth_mean:.4f}")
    
    # 5. Save validation report
    report_path = output_dir / 'validation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nValidation report saved to: {report_path}")
    
    return report


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Generate CTGAN synthetic EEG+genetic data")
    parser.add_argument('--train-csv', type=str,
                       default='seizure_prediction/DataCSVs/CHB-MIT/all_patients_train.csv',
                       help='Path to training CSV with EDF file list')
    parser.add_argument('--n-synthetic', type=int, default=5000,
                       help='Number of synthetic samples to generate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output-dir', type=str,
                       default='data/processed/synthetic',
                       help='Output directory')
    parser.add_argument('--config', type=str,
                       default='configs/config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    print("="*60)
    print("CTGAN SYNTHETIC DATA GENERATOR")
    print("EEG + Genetic Multimodal Fusion")
    print("="*60)
    
    # Load config
    config_path = PROJECT_ROOT / args.config
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    ctgan_config = config.get('ctgan', {})
    
    # Set random seeds
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    
    # ================================================================
    # Load genetic data files
    # ================================================================
    print("\n" + "="*60)
    print("LOADING GENETIC DATA")
    print("="*60)
    
    pli_scores = load_pli_scores()
    clinvar_df, clinvar_counts = load_clinvar_data()
    
    # Create output directory
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ================================================================
    # Step 1: Load EDF file list
    # ================================================================
    print("\n" + "="*60)
    print("STEP 1: LOADING EDF FILE LIST")
    print("="*60)
    
    train_csv_path = PROJECT_ROOT / args.train_csv
    if not train_csv_path.exists():
        print(f"ERROR: Train CSV not found: {train_csv_path}")
        print("Please ensure the CHB-MIT data is set up correctly.")
        return
    
    file_list = load_eeg_file_list(train_csv_path)
    print(f"Loaded {len(file_list)} EDF files")
    
    # Group by patient
    patient_files = {}
    for patient_id, edf_path in file_list:
        if patient_id not in patient_files:
            patient_files[patient_id] = []
        patient_files[patient_id].append(edf_path)
    
    print(f"Found {len(patient_files)} patients:")
    for pid, files in patient_files.items():
        print(f"  {pid}: {len(files)} files")
    
    # ================================================================
    # Step 2: Extract EEG features
    # ================================================================
    print("\n" + "="*60)
    print("STEP 2: EXTRACTING EEG FEATURES")
    print("="*60)
    
    eeg_features_list = []
    genetic_profiles_list = []
    
    sampling_rate = config.get('eeg', {}).get('sampling_rate', 256)
    
    for patient_id, edf_paths in patient_files.items():
        print(f"\nProcessing {patient_id} ({len(edf_paths)} files)...")
        
        for edf_path in edf_paths:
            # Extract EEG features
            eeg_features = extract_eeg_features_for_file(edf_path, sampling_rate)
            
            if eeg_features is None:
                continue
            
            # Parse annotation
            has_seizure, seizure_ratio, preictal_ratio = parse_annotation_file(
                edf_path, sampling_rate
            )
            
            # Build genetic profile (same for all files of one patient)
            genetic_profile = build_genetic_profile_16d(patient_id, rng)
            
            # Combine features
            record = {
                'patient_id': patient_id,
                'filename': os.path.basename(edf_path),
                'has_seizure': int(has_seizure),
                'seizure_ratio': seizure_ratio,
                'preictal_ratio': preictal_ratio,
            }
            
            # Add EEG features
            record.update(eeg_features)
            
            # Add genetic features
            for i, val in enumerate(genetic_profile['weighted_mutation_flags']):
                record[f'weighted_mutation_{i}'] = val
            record['SCN1A_pLI'] = genetic_profile['SCN1A_pLI']
            record['SCN8A_pLI'] = genetic_profile['SCN8A_pLI']
            record['PRS'] = genetic_profile['PRS']
            record['mutation_burden'] = genetic_profile['mutation_burden']
            record['ion_channel_burden'] = genetic_profile['ion_channel_burden']
            record['tier1_flag'] = genetic_profile['tier1_flag']
            record['SCN1A_severity_proxy'] = genetic_profile['SCN1A_severity_proxy']
            
            eeg_features_list.append(record)
    
    if not eeg_features_list:
        print("ERROR: No EEG features extracted. Check EDF file paths.")
        return
    
    real_df = pd.DataFrame(eeg_features_list)
    print(f"\nExtracted features from {len(real_df)} files")
    print(f"Feature dimensions: {real_df.shape}")
    
    # Save real features
    real_features_path = output_dir / 'real_eeg_genetic_features.csv'
    real_df.to_csv(real_features_path, index=False)
    print(f"Real features saved to: {real_features_path}")
    
    # ================================================================
    # Step 3: Augment real data
    # ================================================================
    print("\n" + "="*60)
    print("STEP 3: DATA AUGMENTATION")
    print("="*60)
    
    # Augment to have enough training data
    n_augment = max(10, args.n_synthetic // len(real_df))
    augmented_dfs = [real_df]
    
    for i in range(n_augment - 1):
        aug_df = real_df.copy()
        # Add small noise to numerical columns
        for col in aug_df.select_dtypes(include=[np.number]).columns:
            if col not in ['has_seizure', 'tier1_flag']:
                noise = rng.normal(0, 0.01, len(aug_df))
                aug_df[col] = aug_df[col] + noise
        augmented_dfs.append(aug_df)
    
    augmented_df = pd.concat(augmented_dfs, ignore_index=True)
    print(f"Augmented dataset: {len(augmented_df)} samples")
    
    # ================================================================
    # Step 4: Train CTGAN
    # ================================================================
    # Select numerical columns for CTGAN
    numerical_cols = augmented_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = ['patient_id', 'filename']
    
    ctgan_df = augmented_df[numerical_cols].copy()
    
    # Fill NaN values
    ctgan_df = ctgan_df.fillna(0)
    
    synthesizer = train_ctgan(ctgan_df, ctgan_config)
    
    # ================================================================
    # Step 5: Generate synthetic data
    # ================================================================
    print("\n" + "="*60)
    print("STEP 5: GENERATING SYNTHETIC DATA")
    print("="*60)
    
    synthetic_df = synthesizer.sample(num_rows=args.n_synthetic)
    print(f"Generated {len(synthetic_df)} synthetic samples")
    
    # Save synthetic data
    synthetic_path = output_dir / 'synthetic_eeg_genetic_data.csv'
    synthetic_df.to_csv(synthetic_path, index=False)
    print(f"Synthetic data saved to: {synthetic_path}")
    
    # ================================================================
    # Step 6: Validate
    # ================================================================
    validation_report = validate_synthetic_data(ctgan_df, synthetic_df, output_dir)
    
    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  Real features:    {real_features_path}")
    print(f"  Synthetic data:   {synthetic_path}")
    print(f"  Validation report: {output_dir / 'validation_report.json'}")
    print(f"\nKey Metrics:")
    print(f"  KS Pass Rate:     {validation_report.get('ks_pass_rate', 0)*100:.1f}%")
    print(f"  Correlation Diff: {validation_report.get('correlation_metrics', {}).get('mean_absolute_diff', 0):.4f}")


if __name__ == '__main__':
    main()
