#!/usr/bin/env python3
"""
Generate Synthetic Genetic Patient Cohort for XGBoost Training
===============================================================
Creates a large synthetic patient population with realistic genetic profiles
and seizure outcomes, used to train the Genetic Branch (XGBoost).

Since CHB-MIT provides only 8 real patients with simulated genetic data,
we expand the cohort by generating additional patients from the same
population-level distributions (ClinVar carrier frequencies, gnomAD pLI,
GWAS effect sizes).

Each synthetic patient gets:
    - 9 binary mutation flags (sampled from carrier frequencies)
    - 2 pLI scores (fixed gene-level constants)
    - 1 Polygenic Risk Score (simulated from GWAS SNPs)
    - 1 seizure outcome (label) derived from genetic risk

Usage:
    python scripts/generate_synthetic_genetic_patients.py
    python scripts/generate_synthetic_genetic_patients.py --n-patients 2000 --output data/processed/genetic_vectors/genetic_training_cohort.csv
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
# Config
# ============================================================
DEFAULT_N_PATIENTS = 1000
RANDOM_SEED = 42


def generate_patient(rng, gwas_df=None):
    """Generate one synthetic patient's genetic profile and seizure label."""
    # 1. Mutation flags (binary, from carrier frequencies)
    mutations = {}
    for gene in TARGET_GENES:
        freq = CARRIER_FREQUENCIES.get(gene, 0.005)
        mutations[f"{gene}_mutation"] = int(rng.random() < freq)

    # 2. pLI scores (fixed constants)
    pli_scores = get_pli_scores(genes=PLI_GENES)
    pli_dict = {f"{gene}_pLI": float(pli_scores[i]) for i, gene in enumerate(PLI_GENES)}

    # 3. PRS (simulated from GWAS SNPs)
    prs = compute_prs(patient_snp_dosages=None, snp_weights_df=gwas_df, rng=rng)

    # 4. Genetic risk score (composite)
    #    Higher mutation count + higher PRS → higher seizure risk
    mutation_count = sum(mutations.values())
    # Weight mutations by inverse carrier frequency (rarer = higher impact)
    weighted_mutations = sum(
        mutations[f"{g}_mutation"] / max(CARRIER_FREQUENCIES.get(g, 0.005), 0.001)
        for g in TARGET_GENES
    )
    genetic_risk = weighted_mutations * 0.3 + prs * 0.5

    # 5. Seizure label (probabilistic, based on genetic risk)
    #    Sigmoid mapping: higher genetic_risk → higher P(seizure)
    #    Base rate ~8% (matching CHB-MIT preictal ratio), scaled by risk
    base_prob = 0.08
    risk_multiplier = 1.0 / (1.0 + np.exp(-genetic_risk))  # sigmoid [0,1]
    seizure_prob = min(base_prob + risk_multiplier * 0.4, 0.95)
    has_seizure = int(rng.random() < seizure_prob)

    # 6. Preictal ratio (continuous proxy for seizure burden)
    if has_seizure:
        # Higher genetic risk → more preictal epochs
        preictal_ratio = max(0.05, min(0.35, 0.08 + genetic_risk * 0.05 + rng.normal(0, 0.02)))
    else:
        preictal_ratio = max(0.0, min(0.05, 0.02 + rng.normal(0, 0.01)))

    record = {
        "patient_id": f"synth_{rng.randint(0, 1_000_000):06d}",
        **mutations,
        **pli_dict,
        "polygenic_risk_score": float(prs),
        "genetic_risk_score": float(genetic_risk),
        "has_seizure": has_seizure,
        "preictal_ratio": float(preictal_ratio),
    }
    return record


def load_gwas_data():
    """Load GWAS SNP weights if available."""
    gwas_path = PROJECT_ROOT / "data" / "raw" / "gwas" / "epilepsy_snps.csv"
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
    if len(df) > 1 and df["polygenic_risk_score"].std() > 0:
        df["polygenic_risk_score"] = (
            (df["polygenic_risk_score"] - df["polygenic_risk_score"].mean()) /
            df["polygenic_risk_score"].std()
        )

    return df


def attach_real_patients(df):
    """Append the 8 real CHB-MIT patients with their genetic profiles."""
    real_path = PROJECT_ROOT / "data" / "processed" / "genetic_vectors" / "genetic_profiles.csv"
    if not real_path.exists():
        print(f"  [WARNING] Real genetic profiles not found at {real_path}")
        return df

    real_df = pd.read_csv(real_path)

    # Compute seizure labels for real patients from EEG stats
    # Load feature_stats.json for preictal ratios
    stats_path = PROJECT_ROOT / "presentation" / "data" / "feature_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        # Map patients to preictal ratios from the stats table (hardcoded for now)
        preictal_map = {
            "chb01": 8862 / 95699,
            "chb03": 7261 / 87780,
            "chb05": 2812 / 49919,
            "chb06": 1256 / 17472,
            "chb08": 1779 / 14662,
            "chb10": 1803 / 16642,
            "chb16": 2286 / 17547,
            "chb20": 6509 / 33894,
        }
        real_df["has_seizure"] = 1  # All CHB-MIT patients have seizures
        real_df["preictal_ratio"] = real_df["patient_id"].map(preictal_map).fillna(0.08)
        real_df["genetic_risk_score"] = np.nan
    else:
        real_df["has_seizure"] = 1
        real_df["preictal_ratio"] = 0.08
        real_df["genetic_risk_score"] = np.nan

    # Align columns
    for col in df.columns:
        if col not in real_df.columns:
            real_df[col] = np.nan
    for col in real_df.columns:
        if col not in df.columns:
            df[col] = np.nan

    combined = pd.concat([df, real_df[df.columns]], ignore_index=True)
    print(f"  Attached {len(real_df)} real patients. Total: {len(combined)}")
    return combined


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic genetic patient cohort")
    parser.add_argument("--n-patients", type=int, default=DEFAULT_N_PATIENTS,
                        help="Number of synthetic patients to generate")
    parser.add_argument("--output", type=str,
                        default="data/processed/genetic_vectors/genetic_training_cohort.csv",
                        help="Output CSV path")
    parser.add_argument("--include-real", action="store_true", default=True,
                        help="Append real CHB-MIT patients to the cohort")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
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
    print(f"  Seizure rate: {df['has_seizure'].mean()*100:.1f}%")
    print(f"  Mean preictal ratio: {df['preictal_ratio'].mean():.4f}")
    print(f"\nMutation prevalence:")
    for gene in TARGET_GENES:
        col = f"{gene}_mutation"
        if col in df.columns:
            prev = df[col].mean() * 100
            print(f"    {gene}: {prev:.2f}%")


if __name__ == "__main__":
    main()
