#!/usr/bin/env python3
"""
Generate Synthetic Genetic Patient Cohort (v2 — Literature-Backed)
==================================================================
Creates a synthetic patient population with realistic genetic profiles
using literature-derived risk weights for seizure prediction.

Follows the tier system from Brunklaus et al. 2022, Thomas et al. 2019:
  Tier 1 (highest): SCN1A, KCNQ2, SCN2A
  Tier 2 (high):    SCN8A, KCNT1
  Tier 3 (moderate): DEPDC5, GRIN2A, GABRA1, PCDH19

Each patient gets a 16-dim genetic vector:
  [0-8]   Weighted mutation flags (9 genes)
  [9]     SCN1A pLI score
  [10]    SCN8A pLI score
  [11]    PRS (standardised)
  [12]    Mutation burden score
  [13]    Ion channel gene burden
  [14]    Tier-1 carrier flag
  [15]    SCN1A severity proxy

Usage:
    python scripts/generate_synthetic_genetic_patients.py
    python scripts/generate_synthetic_genetic_patients.py --n-patients 1200
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Project root ──
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.genetic_feature_engineering import (
    TARGET_GENES,
    PLI_GENES,
    CARRIER_FREQUENCIES,
    get_pli_scores,
    compute_prs,
)

# ============================================================
# Literature-backed risk weights (from Genetic xgboost.md)
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

ION_CHANNEL_GENES = ['SCN1A', 'SCN2A', 'SCN8A', 'KCNQ2', 'KCNT1', 'GRIN2A']
TIER1_GENES       = ['SCN1A', 'KCNQ2', 'SCN2A']

DEFAULT_N_PATIENTS = 1200
RANDOM_SEED = 42


def generate_patient(rng, gwas_df=None):
    """Generate one synthetic patient with literature-weighted features."""
    # 1. Binary mutation flags (from population carrier frequencies)
    binary_flags = {}
    for gene in TARGET_GENES:
        freq = CARRIER_FREQUENCIES.get(gene, 0.005)
        binary_flags[gene] = int(rng.random() < freq)

    # 2. Weighted mutation flags (literature risk × binary flag)
    weighted = {}
    for gene in TARGET_GENES:
        weighted[f"{gene}_mutation"] = float(GENE_RISK_WEIGHTS[gene] * binary_flags[gene])

    # 3. pLI scores (fixed constants from gnomAD)
    pli_scores = get_pli_scores(genes=PLI_GENES)
    pli_dict = {
        'SCN1A_pLI': float(pli_scores[0]),
        'SCN8A_pLI': float(pli_scores[1]),
    }

    # 4. PRS (simulated from GWAS SNPs)
    prs = float(compute_prs(patient_snp_dosages=None, snp_weights_df=gwas_df, rng=rng))

    # 5. Engineered features
    mutation_burden = sum(GENE_RISK_WEIGHTS[g] * binary_flags[g] for g in TARGET_GENES)
    ion_channel_burden = sum(GENE_RISK_WEIGHTS[g] * binary_flags[g] for g in ION_CHANNEL_GENES)
    tier1_flag = int(any(binary_flags[g] == 1 for g in TIER1_GENES))
    scn1a_proxy = float(pli_dict['SCN1A_pLI'] * binary_flags['SCN1A'])

    # 6. Seizure label (biologically informed)
    # Tier-1 mutations have much stronger effect than Tier-3
    tier1_count = sum(binary_flags[g] for g in TIER1_GENES)
    tier2_count = sum(binary_flags[g] for g in ['SCN8A', 'KCNT1'])
    tier3_count = sum(binary_flags[g] for g in ['DEPDC5', 'GRIN2A', 'GABRA1', 'PCDH19'])

    # Logistic risk model: strong effect from Tier-1, moderate from Tier-2/3, PRS adds noise
    logit = (
        -1.5                                    # baseline (most people don't have seizures)
        + tier1_count * 1.8                     # strong effect
        + tier2_count * 0.9
        + tier3_count * 0.4
        + prs * 0.6                             # PRS contributes but less than pathogenic variants
        + rng.normal(0, 0.4)                   # biological noise
    )
    seizure_prob = 1.0 / (1.0 + np.exp(-logit))
    seizure_prob = np.clip(seizure_prob, 0.02, 0.98)  # keep realistic bounds
    has_seizure = int(rng.random() < seizure_prob)

    # 7. Preictal ratio (higher genetic risk → more preictal burden)
    if has_seizure:
        preictal_ratio = float(np.clip(
            0.06 + logit * 0.04 + rng.normal(0, 0.015), 0.03, 0.35
        ))
    else:
        preictal_ratio = float(np.clip(
            0.015 + rng.normal(0, 0.008), 0.0, 0.04
        ))

    record = {
        'patient_id': f"synth_{rng.randint(0, 1_000_000):06d}",
        'source': 'synthetic',
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
    return record


def load_gwas_data():
    """Load GWAS SNP weights if available."""
    gwas_path = PROJECT_ROOT / 'data' / 'raw' / 'gwas' / 'epilepsy_snps.csv'
    if gwas_path.exists():
        return pd.read_csv(gwas_path)
    return None


def generate_cohort(n_patients, seed=RANDOM_SEED):
    """Generate N synthetic patients."""
    rng = np.random.RandomState(seed)
    gwas_df = load_gwas_data()

    print(f"Generating {n_patients} synthetic genetic patients (seed={seed})...")
    records = [generate_patient(rng, gwas_df) for _ in range(n_patients)]
    df = pd.DataFrame(records)

    # Standardise PRS across cohort
    if len(df) > 1 and df['polygenic_risk_score'].std() > 0:
        df['polygenic_risk_score'] = (
            (df['polygenic_risk_score'] - df['polygenic_risk_score'].mean()) /
            df['polygenic_risk_score'].std()
        )

    return df


def attach_real_patients(df):
    """Append the 8 real CHB-MIT patients and compute 16-dim features."""
    real_path = PROJECT_ROOT / 'data' / 'processed' / 'genetic_vectors' / 'genetic_profiles.csv'
    if not real_path.exists():
        print(f"  [WARNING] Real genetic profiles not found at {real_path}")
        return df

    real_df = pd.read_csv(real_path)

    # Compute engineered features for real patients
    real_records = []
    for _, row in real_df.iterrows():
        binary = {g: int(row.get(f'{g}_mutation', 0)) for g in TARGET_GENES}
        weighted = {f"{g}_mutation": float(GENE_RISK_WEIGHTS[g] * binary[g]) for g in TARGET_GENES}
        burden = sum(GENE_RISK_WEIGHTS[g] * binary[g] for g in TARGET_GENES)
        ion_burden = sum(GENE_RISK_WEIGHTS[g] * binary[g] for g in ION_CHANNEL_GENES)
        t1_flag = int(any(binary[g] == 1 for g in TIER1_GENES))
        scn1a_p = float(row.get('SCN1A_pLI', 1.0) * binary['SCN1A'])

        real_records.append({
            'patient_id': row['patient_id'],
            'source': 'real',
            **{f'{g}_binary': binary[g] for g in TARGET_GENES},
            **weighted,
            'SCN1A_pLI': float(row.get('SCN1A_pLI', 1.0)),
            'SCN8A_pLI': float(row.get('SCN8A_pLI', 1.0)),
            'polygenic_risk_score': float(row.get('polygenic_risk_score', 0.0)),
            'mutation_burden': float(burden),
            'ion_channel_burden': float(ion_burden),
            'tier1_flag': t1_flag,
            'scn1a_proxy': scn1a_p,
            'has_seizure': 1,  # all CHB-MIT patients have seizures
            'preictal_ratio': 0.08,
        })

    real_out = pd.DataFrame(real_records)

    # Align columns
    for col in df.columns:
        if col not in real_out.columns:
            real_out[col] = np.nan
    for col in real_out.columns:
        if col not in df.columns:
            df[col] = np.nan

    combined = pd.concat([df, real_out[df.columns]], ignore_index=True)
    print(f"  Attached {len(real_out)} real patients. Total: {len(combined)}")
    return combined


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic genetic patient cohort")
    parser.add_argument('--n-patients', type=int, default=DEFAULT_N_PATIENTS,
                        help="Number of synthetic patients to generate")
    parser.add_argument('--output', type=str,
                        default='data/processed/genetic_vectors/genetic_training_cohort.csv',
                        help="Output CSV path")
    parser.add_argument('--include-real', action='store_true', default=True,
                        help="Append real CHB-MIT patients")
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help="Random seed")
    args = parser.parse_args()

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_cohort(args.n_patients, seed=args.seed)
    if args.include_real:
        df = attach_real_patients(df)

    df.to_csv(out_path, index=False)

    print(f"\nSaved cohort to: {out_path}")
    print(f"  Total patients: {len(df)}")
    print(f"  Synthetic: {(df['source']=='synthetic').sum()}")
    print(f"  Real:      {(df['source']=='real').sum()}")
    print(f"  Seizure rate: {df['has_seizure'].mean()*100:.1f}%")
    print(f"  Mean preictal ratio: {df['preictal_ratio'].mean():.4f}")
    print(f"\nMutation prevalence (binary):")
    for gene in TARGET_GENES:
        col = f"{gene}_binary"
        if col in df.columns:
            prev = df[col].mean() * 100
            print(f"    {gene}: {prev:.2f}%")
    print(f"\nEngineered features:")
    print(f"    Mean mutation burden: {df['mutation_burden'].mean():.3f}")
    print(f"    Mean ion channel burden: {df['ion_channel_burden'].mean():.3f}")
    print(f"    Tier-1 carriers: {df['tier1_flag'].sum()} ({df['tier1_flag'].mean()*100:.1f}%)")


if __name__ == '__main__':
    main()
