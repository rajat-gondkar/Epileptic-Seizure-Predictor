#!/usr/bin/env python3
"""
Generate Synthetic Genetic Patient Cohort (v3 — Strong Signal)
===============================================================
Creates an all-synthetic patient population (no real data used) with
stronger genetic signal for XGBoost training.

Key changes for strong signal:
  - Higher mutation prevalence (5-10% instead of 0.3-1.5%)
  - Stronger genetic coefficients (Tier-1 = +2.5 log-odds)
  - Lower label noise (sigma = 0.1 instead of 0.4)
  - Larger cohort (5000 patients by default)
  - No real patient data mixed in

Usage:
    python scripts/generate_synthetic_genetic_patients.py
    python scripts/generate_synthetic_genetic_patients.py --n-patients 5000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -- Project root --
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.genetic_feature_engineering import (
    TARGET_GENES,
    PLI_GENES,
    get_pli_scores,
    compute_prs,
)

# ============================================================
# Literature-backed risk weights
# ============================================================
GENE_RISK_WEIGHTS = {
    'SCN1A':  1.00,
    'KCNQ2':  0.95,
    'SCN2A':  0.90,
    'SCN8A':  0.85,
    'KCNT1':  0.80,
    'DEPDC5': 0.65,
    'GRIN2A': 0.55,
    'GABRA1': 0.50,
    'PCDH19': 0.40,
}

# Higher carrier frequencies for stronger signal
CARRIER_FREQS_STRONG = {
    'SCN1A':  0.10,   # 10%
    'KCNQ2':  0.08,
    'SCN2A':  0.07,
    'SCN8A':  0.06,
    'KCNT1':  0.05,
    'DEPDC5': 0.05,
    'PCDH19': 0.04,
    'GRIN2A': 0.04,
    'GABRA1': 0.03,
}

ION_CHANNEL_GENES = ['SCN1A', 'SCN2A', 'SCN8A', 'KCNQ2', 'KCNT1', 'GRIN2A']
TIER1_GENES       = ['SCN1A', 'KCNQ2', 'SCN2A']

DEFAULT_N_PATIENTS = 5000
RANDOM_SEED = 42


def load_gwas_data():
    """Load GWAS SNP weights if available."""
    gwas_path = PROJECT_ROOT / 'data' / 'raw' / 'gwas' / 'epilepsy_snps.csv'
    if gwas_path.exists():
        return pd.read_csv(gwas_path)
    return None


def generate_patient(rng, gwas_df=None):
    """Generate one synthetic patient with strong genetic signal."""
    # 1. Binary mutation flags
    binary_flags = {}
    for gene in TARGET_GENES:
        freq = CARRIER_FREQS_STRONG.get(gene, 0.05)
        binary_flags[gene] = int(rng.random() < freq)

    # 2. Weighted mutation flags
    weighted = {}
    for gene in TARGET_GENES:
        weighted[f"{gene}_mutation"] = float(GENE_RISK_WEIGHTS[gene] * binary_flags[gene])

    # 3. pLI scores
    pli_scores = get_pli_scores(genes=PLI_GENES)
    pli_dict = {
        'SCN1A_pLI': float(pli_scores[0]),
        'SCN8A_pLI': float(pli_scores[1]),
    }

    # 4. PRS
    prs = float(compute_prs(patient_snp_dosages=None, snp_weights_df=gwas_df, rng=rng))

    # 5. Engineered features
    mutation_burden = sum(GENE_RISK_WEIGHTS[g] * binary_flags[g] for g in TARGET_GENES)
    ion_channel_burden = sum(GENE_RISK_WEIGHTS[g] * binary_flags[g] for g in ION_CHANNEL_GENES)
    tier1_flag = int(any(binary_flags[g] == 1 for g in TIER1_GENES))
    scn1a_proxy = float(pli_dict['SCN1A_pLI'] * binary_flags['SCN1A'])

    # 6. Seizure label — STRONG genetic signal, LOW noise
    tier1_count = sum(binary_flags[g] for g in TIER1_GENES)
    tier2_count = sum(binary_flags[g] for g in ['SCN8A', 'KCNT1'])
    tier3_count = sum(binary_flags[g] for g in ['DEPDC5', 'GRIN2A', 'GABRA1', 'PCDH19'])

    logit = (
        -1.2                                    # baseline
        + tier1_count * 2.5                     # Tier-1: very strong
        + tier2_count * 1.5                     # Tier-2: strong
        + tier3_count * 0.8                     # Tier-3: moderate
        + prs * 0.8                             # PRS contribution
        + rng.normal(0, 0.1)                   # LOW biological noise
    )
    seizure_prob = 1.0 / (1.0 + np.exp(-logit))
    seizure_prob = np.clip(seizure_prob, 0.02, 0.98)
    has_seizure = int(rng.random() < seizure_prob)

    # 7. Preictal ratio
    if has_seizure:
        preictal_ratio = float(np.clip(0.06 + logit * 0.05 + rng.normal(0, 0.01), 0.03, 0.40))
    else:
        preictal_ratio = float(np.clip(0.015 + rng.normal(0, 0.008), 0.0, 0.05))

    return {
        'patient_id': f"synth_{rng.randint(0, 10_000_000):07d}",
        **{f'{g}_binary': binary_flags[g] for g in TARGET_GENES},
        **weighted,
        **pli_dict,
        'polygenic_risk_score': prs,
        'mutation_burden': float(mutation_burden),
        'ion_channel_burden': float(ion_channel_burden),
        'tier1_flag': tier1_flag,
        'scn1a_proxy': scn1a_proxy,
        'has_seizure': has_seizure,
        'preictal_ratio': preictal_ratio,
    }


def generate_cohort(n_patients, seed=RANDOM_SEED):
    """Generate N synthetic patients."""
    rng = np.random.RandomState(seed)
    gwas_df = load_gwas_data()

    print(f"Generating {n_patients} all-synthetic genetic patients (seed={seed})...")
    records = [generate_patient(rng, gwas_df) for _ in range(n_patients)]
    df = pd.DataFrame(records)

    # Standardise PRS
    if len(df) > 1 and df['polygenic_risk_score'].std() > 0:
        df['polygenic_risk_score'] = (
            (df['polygenic_risk_score'] - df['polygenic_risk_score'].mean()) /
            df['polygenic_risk_score'].std()
        )

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic genetic patient cohort")
    parser.add_argument('--n-patients', type=int, default=DEFAULT_N_PATIENTS,
                        help="Number of synthetic patients (default: 5000)")
    parser.add_argument('--output', type=str,
                        default='data/processed/genetic_vectors/genetic_training_cohort.csv',
                        help="Output CSV path")
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help="Random seed")
    args = parser.parse_args()

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_cohort(args.n_patients, seed=args.seed)
    df.to_csv(out_path, index=False)

    print(f"\nSaved cohort to: {out_path}")
    print(f"  Total patients: {len(df)}")
    print(f"  Seizure rate: {df['has_seizure'].mean()*100:.1f}%")
    print(f"  Mean preictal ratio: {df['preictal_ratio'].mean():.4f}")
    print(f"\nMutation prevalence:")
    for gene in TARGET_GENES:
        prev = df[f'{gene}_binary'].mean() * 100
        print(f"    {gene}: {prev:.1f}%")
    print(f"\nEngineered features:")
    print(f"    Mean mutation burden: {df['mutation_burden'].mean():.3f}")
    print(f"    Mean ion channel burden: {df['ion_channel_burden'].mean():.3f}")
    print(f"    Tier-1 carriers: {df['tier1_flag'].sum()} ({df['tier1_flag'].mean()*100:.1f}%)")


if __name__ == '__main__':
    main()
