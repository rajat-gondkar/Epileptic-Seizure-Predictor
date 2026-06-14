#!/usr/bin/env python3
"""
Generate Synthetic Genetic Patient Cohort (v6 — Balanced Realism)
================================================================
Uses REAL parameters from ClinVar/gnomAD/GWAS but adjusts for
trainability: slightly elevated carrier frequencies and stronger
penetrance to create enough signal for the model to learn.

Key insight: With TRUE population frequencies (0.1%), there are
only ~50-100 seizure patients in 10,000 — too few to train on.
We use 1-2% carrier frequencies (still rare, but trainable).

Usage:
    python scripts/generate_synthetic_genetic_patients.py
    python scripts/generate_synthetic_genetic_patients.py --n-patients 10000
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
    EXTENDED_PLI_GENES,
    get_pli_scores,
    compute_prs,
)


# ============================================================
# Load REAL data from raw files
# ============================================================
def load_clinvar_data():
    """Load ClinVar variant counts per gene."""
    clinvar_path = PROJECT_ROOT / 'data' / 'raw' / 'clinvar' / 'epilepsy_variants.csv'
    if not clinvar_path.exists():
        print(f"WARNING: ClinVar file not found at {clinvar_path}")
        return {}

    df = pd.read_csv(clinvar_path)
    df_grch38 = df[df['Assembly'] == 'GRCh38']
    variant_counts = df_grch38.groupby('GeneSymbol').size().to_dict()
    total_variants = sum(variant_counts.values())

    gene_burden = {}
    for gene in TARGET_GENES:
        count = variant_counts.get(gene, 0)
        gene_burden[gene] = {
            'variant_count': count,
            'burden_fraction': count / total_variants if total_variants > 0 else 0,
        }
    return gene_burden


def load_gnomad_data():
    """Load gnomAD gene constraint metrics."""
    gnomad_path = PROJECT_ROOT / 'data' / 'raw' / 'gnomad' / 'pli_scores.csv'
    if not gnomad_path.exists():
        return {}

    df = pd.read_csv(gnomad_path)
    gene_metrics = {}
    for _, row in df.iterrows():
        gene_metrics[row['gene']] = {
            'pLI': row['pLI'],
            'oe_lof': row['oe_lof'],
            'oe_lof_upper': row['oe_lof_upper'],
            'oe_mis': row['oe_mis'],
            'obs_lof': row['obs_lof'],
            'exp_lof': row['exp_lof'],
        }
    return gene_metrics


def load_gwas_data():
    """Load GWAS SNP weights."""
    gwas_path = PROJECT_ROOT / 'data' / 'raw' / 'gwas' / 'epilepsy_snps.csv'
    if not gwas_path.exists():
        return None
    return pd.read_csv(gwas_path)


# ============================================================
# Derive parameters from REAL data (with trainability adjustments)
# ============================================================
def compute_carrier_frequencies(clinvar_data, gnomad_data, n_alleles=250000):
    """
    Derive carrier frequencies from ClinVar + gnomAD.

    REAL frequencies are ~0.01-0.5%. For trainability, we scale up
    to ~1-2% so there are enough positive cases to learn from.
    """
    carrier_freqs = {}

    for gene in TARGET_GENES:
        clinvar_count = clinvar_data.get(gene, {}).get('variant_count', 0)
        gnomad_info = gnomad_data.get(gene, {})
        obs_lof = gnomad_info.get('obs_lof', 0)

        # Real carrier frequency estimate
        estimated_carriers = obs_lof + (clinvar_count * 0.1)
        real_freq = estimated_carriers / n_alleles

        # Scale up for trainability (10-20x real frequency)
        # This gives us ~1-2% carrier rates instead of ~0.1%
        trainability_factor = 15.0
        train_freq = min(real_freq * trainability_factor, 0.03)  # Cap at 3%

        # Floor at 0.5% so every gene has some signal
        carrier_freqs[gene] = max(train_freq, 0.005)

    return carrier_freqs


def compute_risk_weights(clinvar_data, gnomad_data):
    """
    Derive gene risk weights from ClinVar burden + gnomAD constraint.
    """
    raw_scores = {}
    for gene in TARGET_GENES:
        burden = clinvar_data.get(gene, {}).get('burden_fraction', 0)
        gnomad = gnomad_data.get(gene, {})
        oe_lof = gnomad.get('oe_lof', 0.5)
        constraint = max(1.0 - oe_lof, 0.0)
        pli = gnomad.get('pLI', 0.5)
        raw_scores[gene] = burden * constraint * pli

    # Normalize to [0.4, 1.0]
    if raw_scores:
        min_score = min(raw_scores.values())
        max_score = max(raw_scores.values())
        score_range = max_score - min_score
        risk_weights = {}
        for gene in TARGET_GENES:
            if score_range > 0:
                normalized = (raw_scores[gene] - min_score) / score_range
            else:
                normalized = 0.5
            risk_weights[gene] = 0.4 + normalized * 0.6
    else:
        risk_weights = {g: 0.5 for g in TARGET_GENES}
    return risk_weights


def compute_seizure_penetrance(risk_weights):
    """
    Compute seizure penetrance for each gene.

    Uses literature base penetrance, scaled by risk weight.
    Higher penetrance = stronger signal for the model.
    """
    # Base penetrance from literature
    base_penetrance = {
        'SCN1A':  0.85,   # Dravet syndrome
        'KCNQ2':  0.75,   # Neonatal DEE
        'SCN2A':  0.70,   # Variable severity
        'SCN8A':  0.65,   # Early-onset EE
        'KCNT1':  0.60,   # Focal epilepsy
        'DEPDC5': 0.50,   # Focal epilepsy
        'GRIN2A': 0.45,   # Sleep-related epilepsy
        'GABRA1': 0.40,   # Absence epilepsy
        'PCDH19': 0.55,   # Female-limited
    }

    # Scale by risk weight (0.4-1.0)
    penetrance = {}
    for gene in TARGET_GENES:
        base = base_penetrance.get(gene, 0.5)
        weight = risk_weights.get(gene, 0.5)
        penetrance[gene] = min(base * (0.6 + weight * 0.4), 0.95)

    return penetrance


# ============================================================
# Patient Generation
# ============================================================
def generate_patient(rng, carrier_freqs, risk_weights, penetrance, gwas_df):
    """Generate one synthetic patient."""
    # 1. Binary mutation flags
    binary_flags = {}
    for gene in TARGET_GENES:
        freq = carrier_freqs.get(gene, 0.01)
        binary_flags[gene] = int(rng.random() < freq)

    # 2. Weighted mutation flags
    weighted = {}
    for gene in TARGET_GENES:
        weighted[f"{gene}_mutation"] = float(risk_weights[gene] * binary_flags[gene])

    # 3. pLI scores (real gnomAD values)
    pli_scores = get_pli_scores(genes=EXTENDED_PLI_GENES)
    pli_dict = {}
    for i, gene in enumerate(EXTENDED_PLI_GENES):
        pli_dict[f'{gene}_pLI'] = float(pli_scores[i])

    # 4. PRS (real GWAS via Hardy-Weinberg)
    prs = float(compute_prs(patient_snp_dosages=None, snp_weights_df=gwas_df, rng=rng))

    # 5. Interaction features
    sodium_interaction = float(binary_flags['SCN1A'] * binary_flags['SCN2A'] * binary_flags['SCN8A'])
    potassium_interaction = float(binary_flags['KCNQ2'] * binary_flags['KCNT1'])
    receptor_interaction = float(binary_flags['GABRA1'] * binary_flags['GRIN2A'])
    tier1_flag = int(any(binary_flags[g] == 1 for g in ['SCN1A', 'KCNQ2', 'SCN2A']))
    prs_tier1_interaction = float(prs * tier1_flag)
    mutation_burden = sum(risk_weights[g] * binary_flags[g] for g in TARGET_GENES)

    # 6. Compute seizure probability
    # Use log-odds for proper probability calculation
    base_log_odds = np.log(0.02 / 0.98)  # ~2% baseline (slightly elevated)

    # Additive log-odds from each mutation
    mutation_log_odds = 0.0
    n_mutations = sum(1 for g in TARGET_GENES if binary_flags[g] == 1)
    for gene in TARGET_GENES:
        if binary_flags[gene] == 1:
            p = penetrance.get(gene, 0.5)
            mutation_log_odds += np.log(p / (1 - p + 1e-8))

    # PRS effect (moderate)
    prs_effect = prs * 0.15

    # Interaction effects
    interaction_effect = 0.0
    if sodium_interaction > 0:
        interaction_effect += 0.6
    if potassium_interaction > 0:
        interaction_effect += 0.4
    if receptor_interaction > 0:
        interaction_effect += 0.3

    # Combine in log-odds space
    total_log_odds = (
        base_log_odds
        + mutation_log_odds * 0.7  # Stronger mutation effect
        + prs_effect
        + interaction_effect
    )

    # Add noise
    noise = rng.normal(0, 0.25)
    logit = total_log_odds + noise
    seizure_prob = 1.0 / (1.0 + np.exp(-logit))
    seizure_prob = np.clip(seizure_prob, 0.005, 0.99)

    has_seizure = int(rng.random() < seizure_prob)

    # 7. Preictal ratio
    if has_seizure:
        preictal_ratio = float(np.clip(0.08 + rng.normal(0, 0.04), 0.02, 0.40))
    else:
        preictal_ratio = float(np.clip(0.01 + rng.normal(0, 0.008), 0.0, 0.05))

    return {
        'patient_id': f"synth_{rng.randint(0, 100_000_000):08d}",
        **{f'{g}_binary': binary_flags[g] for g in TARGET_GENES},
        **weighted,
        **pli_dict,
        'polygenic_risk_score': prs,
        'sodium_channel_interaction': sodium_interaction,
        'potassium_channel_interaction': potassium_interaction,
        'receptor_interaction': receptor_interaction,
        'prs_tier1_interaction': prs_tier1_interaction,
        'mutation_burden': float(mutation_burden),
        'has_seizure': has_seizure,
        'preictal_ratio': preictal_ratio,
        'n_mutations': n_mutations,
        'seizure_prob': float(seizure_prob),
    }


def generate_cohort(n_patients, seed=42):
    """Generate N synthetic patients."""
    rng = np.random.RandomState(seed)

    print("=" * 70)
    print("LOADING REAL DATA FROM RAW FILES")
    print("=" * 70)

    clinvar_data = load_clinvar_data()
    gnomad_data = load_gnomad_data()
    gwas_df = load_gwas_data()

    print(f"ClinVar: {sum(v['variant_count'] for v in clinvar_data.values())} variants")
    print(f"gnomAD: {len(gnomad_data)} genes")
    print(f"GWAS: {len(gwas_df) if gwas_df is not None else 0} SNPs")

    print("\n" + "=" * 70)
    print("DERIVING PARAMETERS (with trainability adjustments)")
    print("=" * 70)

    carrier_freqs = compute_carrier_frequencies(clinvar_data, gnomad_data)
    risk_weights = compute_risk_weights(clinvar_data, gnomad_data)
    penetrance = compute_seizure_penetrance(risk_weights)

    print("\nCarrier Frequencies (scaled for trainability):")
    for gene in TARGET_GENES:
        print(f"  {gene:>8s}: {carrier_freqs[gene]*100:.2f}%")

    print("\nRisk Weights:")
    for gene in TARGET_GENES:
        print(f"  {gene:>8s}: {risk_weights[gene]:.3f}")

    print("\nSeizure Penetrance:")
    for gene in TARGET_GENES:
        print(f"  {gene:>8s}: {penetrance[gene]*100:.1f}%")

    # Generate cohort
    print(f"\n{'='*70}")
    print(f"GENERATING {n_patients} PATIENTS")
    print(f"{'='*70}")

    records = [generate_patient(rng, carrier_freqs, risk_weights, penetrance, gwas_df)
               for _ in range(n_patients)]
    df = pd.DataFrame(records)

    # Standardize PRS
    if len(df) > 1 and df['polygenic_risk_score'].std() > 0:
        df['polygenic_risk_score'] = (
            (df['polygenic_risk_score'] - df['polygenic_risk_score'].mean()) /
            df['polygenic_risk_score'].std()
        )

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate realistic synthetic genetic cohort")
    parser.add_argument('--n-patients', type=int, default=10000,
                        help="Number of synthetic patients")
    parser.add_argument('--output', type=str,
                        default='data/processed/genetic_vectors/genetic_training_cohort.csv',
                        help="Output CSV path")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_cohort(args.n_patients, seed=args.seed)

    # Remove debug columns
    debug_cols = ['n_mutations', 'seizure_prob']
    for col in debug_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    df.to_csv(out_path, index=False)

    print(f"\n{'='*70}")
    print(f"SAVED: {out_path}")
    print(f"{'='*70}")
    print(f"  Total patients: {len(df)}")
    print(f"  Features: {len(df.columns) - 2}")
    print(f"  Seizure rate: {df['has_seizure'].mean()*100:.1f}%")
    print(f"  No-seizure: {(df['has_seizure']==0).sum()}")
    print(f"  Seizure: {(df['has_seizure']==1).sum()}")

    print(f"\nMutation prevalence:")
    for gene in TARGET_GENES:
        prev = df[f'{gene}_binary'].mean() * 100
        print(f"  {gene:>8s}: {prev:.2f}%")


if __name__ == '__main__':
    main()
