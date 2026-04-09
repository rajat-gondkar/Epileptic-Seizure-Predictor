#!/usr/bin/env python3
"""
Genetic Feature Engineering Module
=====================================
Constructs 12-point genetic profile vectors for each patient.

Feature Vector (12 dimensions):
    [0-8]   Mutation flags for 9 epilepsy genes (binary: 0/1)
    [9-10]  pLI scores for SCN1A, SCN8A (continuous: 0-1)
    [11]    Polygenic Risk Score (standardized)

For CHB-MIT patients (no real genetic data available):
    Generates simulated genetic profiles using population-level 
    allele frequencies from gnomAD and GWAS effect sizes.

Usage:
    python src/data_pipeline/genetic_feature_engineering.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ============================================================
# Configuration
# ============================================================
TARGET_GENES = [
    'SCN1A', 'SCN8A', 'KCNQ2', 'SCN2A', 'KCNT1',
    'DEPDC5', 'PCDH19', 'GRIN2A', 'GABRA1'
]

PLI_GENES = ['SCN1A', 'SCN8A']

# Population-level pathogenic variant carrier frequencies
# (estimated from literature: ~1/2000 to ~1/20000 for rare epilepsy genes)
CARRIER_FREQUENCIES = {
    'SCN1A': 0.015,    # Most common epilepsy gene mutation
    'SCN8A': 0.008,
    'KCNQ2': 0.010,
    'SCN2A': 0.007,
    'KCNT1': 0.004,
    'DEPDC5': 0.005,
    'PCDH19': 0.006,
    'GRIN2A': 0.005,
    'GABRA1': 0.003,
}


def load_config(config_path="configs/config.yaml"):
    """Load configuration from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ============================================================
# Mutation Flag Construction
# ============================================================
def build_mutation_flags(patient_variants=None, clinvar_df=None, 
                         simulated=True, rng=None):
    """
    Build binary mutation flags for the 9 target genes.
    
    Args:
        patient_variants: DataFrame of patient's variants (if real data available)
        clinvar_df: ClinVar DataFrame for reference
        simulated: Whether to simulate flags based on population frequencies
        rng: numpy RandomState for reproducibility
        
    Returns:
        np.array of shape [9], binary flags
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    flags = np.zeros(9, dtype=np.float32)
    
    if patient_variants is not None and not simulated:
        # Real patient data - check for pathogenic variants
        for i, gene in enumerate(TARGET_GENES):
            gene_variants = patient_variants[
                patient_variants['GeneSymbol'] == gene
            ]
            if len(gene_variants) > 0:
                flags[i] = 1.0
    else:
        # Simulate based on population carrier frequencies
        for i, gene in enumerate(TARGET_GENES):
            freq = CARRIER_FREQUENCIES.get(gene, 0.005)
            flags[i] = 1.0 if rng.random() < freq else 0.0
    
    return flags


# ============================================================
# pLI Score Extraction
# ============================================================
def get_pli_scores(gnomad_df=None, genes=None):
    """
    Extract pLI scores for specified genes from gnomAD data.
    
    Args:
        gnomad_df: gnomAD DataFrame with pLI column
        genes: List of gene names to extract
        
    Returns:
        np.array of pLI scores
    """
    if genes is None:
        genes = PLI_GENES
    
    # Known pLI values as fallback
    known_pli = {
        'SCN1A': 1.00, 'SCN8A': 1.00, 'KCNQ2': 0.99999,
        'SCN2A': 1.00, 'KCNT1': 0.00003, 'DEPDC5': 0.11650,
        'PCDH19': 0.99975, 'GRIN2A': 0.99998, 'GABRA1': 0.91489
    }
    
    pli_scores = np.zeros(len(genes), dtype=np.float32)
    
    if gnomad_df is not None:
        for i, gene in enumerate(genes):
            gene_row = gnomad_df[gnomad_df['gene'] == gene]
            if len(gene_row) > 0 and 'pLI' in gene_row.columns:
                pli_scores[i] = float(gene_row['pLI'].iloc[0])
            else:
                pli_scores[i] = known_pli.get(gene, 0.5)
    else:
        for i, gene in enumerate(genes):
            pli_scores[i] = known_pli.get(gene, 0.5)
    
    return pli_scores


# ============================================================
# Polygenic Risk Score
# ============================================================
def compute_prs(patient_snp_dosages=None, snp_weights_df=None, rng=None):
    """
    Compute Polygenic Risk Score.
    
    PRS = sum over all GWAS SNPs of: beta_i * genotype_i
    genotype_i in {0, 1, 2} (number of effect alleles)
    beta_i = log(OR) from GWAS Catalog
    
    For simulated patients: sample genotypes from allele frequencies.
    
    Args:
        patient_snp_dosages: dict of {snp_id: dosage (0/1/2)}
        snp_weights_df: DataFrame with 'snp_id', 'or_beta', 'risk_allele_freq'
        rng: numpy RandomState
        
    Returns:
        prs: float, raw Polygenic Risk Score
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    if snp_weights_df is None:
        return 0.0
    
    prs = 0.0
    
    for _, snp in snp_weights_df.iterrows():
        beta = np.log(snp['or_beta']) if snp['or_beta'] > 0 else 0
        
        if patient_snp_dosages is not None and snp['snp_id'] in patient_snp_dosages:
            dosage = patient_snp_dosages[snp['snp_id']]
        else:
            # Simulate genotype from allele frequency (Hardy-Weinberg)
            freq = snp.get('risk_allele_freq', 0.3)
            prob_0 = (1 - freq) ** 2
            prob_1 = 2 * freq * (1 - freq)
            # prob_2 = freq ** 2
            
            r = rng.random()
            if r < prob_0:
                dosage = 0
            elif r < prob_0 + prob_1:
                dosage = 1
            else:
                dosage = 2
        
        prs += beta * dosage
    
    return prs


# ============================================================
# Full Genetic Vector Construction
# ============================================================
def build_genetic_vector(patient_id, clinvar_df=None, gnomad_df=None, 
                         gwas_df=None, patient_variants=None,
                         simulated=True, seed=None):
    """
    Build the complete 12-dimensional genetic feature vector for a patient.
    
    Args:
        patient_id: str, patient identifier
        clinvar_df: ClinVar pathogenic variants DataFrame
        gnomad_df: gnomAD pLI scores DataFrame
        gwas_df: GWAS SNP weights DataFrame
        patient_variants: Patient-specific variant data (if available)
        simulated: Whether to simulate genetic data
        seed: Random seed for reproducibility
        
    Returns:
        np.array of shape [12]
    """
    # Create reproducible RNG per patient
    if seed is None:
        seed = hash(patient_id) % (2**31)
    rng = np.random.RandomState(seed)
    
    # [0-8] Mutation flags (9 features)
    mutation_flags = build_mutation_flags(
        patient_variants=patient_variants,
        clinvar_df=clinvar_df,
        simulated=simulated,
        rng=rng
    )
    
    # [9-10] pLI scores (2 features) - these are gene-level constants
    pli_scores = get_pli_scores(gnomad_df, genes=PLI_GENES)
    
    # [11] Polygenic Risk Score (1 feature)
    prs = compute_prs(
        patient_snp_dosages=None,  # Simulated
        snp_weights_df=gwas_df,
        rng=rng
    )
    
    # Concatenate
    genetic_vector = np.concatenate([mutation_flags, pli_scores, [prs]])
    
    return genetic_vector.astype(np.float32)


# ============================================================
# Batch Processing
# ============================================================
def generate_genetic_profiles(patients, clinvar_df=None, gnomad_df=None,
                              gwas_df=None, output_path=None):
    """
    Generate genetic profiles for all patients.
    
    For CHB-MIT patients (no real genetic data), generates simulated 
    profiles using population-level frequencies.
    
    Args:
        patients: List of patient IDs
        clinvar_df: ClinVar data
        gnomad_df: gnomAD data
        gwas_df: GWAS data
        output_path: Path to save CSV
        
    Returns:
        DataFrame with genetic profiles
    """
    records = []
    
    for patient_id in patients:
        vector = build_genetic_vector(
            patient_id,
            clinvar_df=clinvar_df,
            gnomad_df=gnomad_df,
            gwas_df=gwas_df,
            simulated=True
        )
        
        record = {'patient_id': patient_id}
        for i, gene in enumerate(TARGET_GENES):
            record[f'{gene}_mutation'] = int(vector[i])
        record['SCN1A_pLI'] = vector[9]
        record['SCN8A_pLI'] = vector[10]
        record['polygenic_risk_score'] = vector[11]
        
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Standardize PRS across cohort
    if len(df) > 1 and df['polygenic_risk_score'].std() > 0:
        df['polygenic_risk_score'] = (
            (df['polygenic_risk_score'] - df['polygenic_risk_score'].mean()) / 
            df['polygenic_risk_score'].std()
        )
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved genetic profiles for {len(df)} patients to {output_path}")
    
    return df


# ============================================================
# Main
# ============================================================
def main():
    config = load_config()
    
    print("=" * 60)
    print("GENETIC FEATURE ENGINEERING")
    print("=" * 60)
    
    # Load genetic datasets
    clinvar_path = Path(config['paths']['data']['raw']['clinvar']) / 'epilepsy_variants.csv'
    gnomad_path = Path(config['paths']['data']['raw']['gnomad']) / 'pli_scores.csv'
    gwas_path = Path(config['paths']['data']['raw']['gwas']) / 'epilepsy_snps.csv'
    
    clinvar_df = pd.read_csv(clinvar_path) if clinvar_path.exists() else None
    gnomad_df = pd.read_csv(gnomad_path) if gnomad_path.exists() else None
    gwas_df = pd.read_csv(gwas_path) if gwas_path.exists() else None
    
    print(f"ClinVar variants loaded: {len(clinvar_df) if clinvar_df is not None else 0}")
    print(f"gnomAD genes loaded: {len(gnomad_df) if gnomad_df is not None else 0}")
    print(f"GWAS SNPs loaded: {len(gwas_df) if gwas_df is not None else 0}")
    
    # Get patient list
    patients = config['dataset']['chb_mit_patients']
    print(f"\nGenerating simulated genetic profiles for {len(patients)} patients...")
    
    # Generate profiles
    output_path = Path(config['paths']['data']['processed']['genetic_vectors']) / 'genetic_profiles.csv'
    
    df = generate_genetic_profiles(
        patients,
        clinvar_df=clinvar_df,
        gnomad_df=gnomad_df,
        gwas_df=gwas_df,
        output_path=output_path
    )
    
    # Display results
    print(f"\nGenetic Profile Summary:")
    print(df.to_string(index=False))
    
    # Statistics
    print(f"\nMutation prevalence:")
    for gene in TARGET_GENES:
        col = f'{gene}_mutation'
        if col in df.columns:
            prev = df[col].mean() * 100
            print(f"  {gene}: {prev:.1f}%")
    
    print(f"\nPRS statistics:")
    print(f"  Mean: {df['polygenic_risk_score'].mean():.4f}")
    print(f"  Std:  {df['polygenic_risk_score'].std():.4f}")
    print(f"  Min:  {df['polygenic_risk_score'].min():.4f}")
    print(f"  Max:  {df['polygenic_risk_score'].max():.4f}")


if __name__ == '__main__':
    main()
