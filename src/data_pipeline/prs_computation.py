#!/usr/bin/env python3
"""
Polygenic Risk Score (PRS) Computation Module
================================================
Computes Polygenic Risk Score from GWAS SNP effect sizes.

PRS = sum over all GWAS SNPs of: beta_i * genotype_i
    genotype_i in {0, 1, 2} (number of effect alleles)
    beta_i = log(OR) from GWAS Catalog
    Standardize: (PRS - mean_PRS) / std_PRS across cohort

Usage:
    from src.data_pipeline.prs_computation import PRSComputer
    
    prs_computer = PRSComputer(gwas_snps_df)
    prs = prs_computer.compute_patient_prs(patient_snp_dosages)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class PRSComputer:
    """Computes Polygenic Risk Score for patients."""
    
    def __init__(self, gwas_df: pd.DataFrame):
        """
        Args:
            gwas_df: DataFrame with columns: 'snp_id', 'or_beta', 'risk_allele_freq'
        """
        self.gwas_df = gwas_df.copy()
        
        # Pre-compute log(OR) = beta weights
        self.gwas_df['beta'] = np.log(self.gwas_df['or_beta'].clip(lower=0.01))
        
        self.snp_ids = self.gwas_df['snp_id'].tolist()
        self.betas = self.gwas_df['beta'].values
        self.allele_freqs = self.gwas_df['risk_allele_freq'].values
    
    def simulate_genotypes(self, rng=None):
        """
        Simulate genotypes based on Hardy-Weinberg equilibrium.
        
        Args:
            rng: numpy RandomState for reproducibility
            
        Returns:
            dosages: dict of {snp_id: dosage (0/1/2)}
        """
        if rng is None:
            rng = np.random.RandomState()
        
        dosages = {}
        for snp_id, freq in zip(self.snp_ids, self.allele_freqs):
            # Hardy-Weinberg: p^2 + 2pq + q^2 = 1
            p = freq  # risk allele frequency
            q = 1 - p
            
            prob_0 = q ** 2
            prob_1 = 2 * p * q
            # prob_2 = p ** 2
            
            r = rng.random()
            if r < prob_0:
                dosages[snp_id] = 0
            elif r < prob_0 + prob_1:
                dosages[snp_id] = 1
            else:
                dosages[snp_id] = 2
        
        return dosages
    
    def compute_patient_prs(self, snp_dosages: Dict[str, int]) -> float:
        """
        Compute raw PRS for a single patient.
        
        Args:
            snp_dosages: dict of {snp_id: dosage (0/1/2)}
            
        Returns:
            prs: float, raw PRS value
        """
        prs = 0.0
        for snp_id, beta in zip(self.snp_ids, self.betas):
            dosage = snp_dosages.get(snp_id, 0)
            prs += beta * dosage
        
        return prs
    
    def compute_cohort_prs(self, n_patients: int, seed: int = 42) -> np.ndarray:
        """
        Compute standardized PRS for a simulated cohort.
        
        Args:
            n_patients: Number of patients to simulate
            seed: Random seed
            
        Returns:
            prs_standardized: np.array of shape [n_patients]
        """
        rng = np.random.RandomState(seed)
        
        raw_prs = np.zeros(n_patients)
        for i in range(n_patients):
            dosages = self.simulate_genotypes(rng)
            raw_prs[i] = self.compute_patient_prs(dosages)
        
        # Standardize
        mean_prs = np.mean(raw_prs)
        std_prs = np.std(raw_prs)
        
        if std_prs > 0:
            prs_standardized = (raw_prs - mean_prs) / std_prs
        else:
            prs_standardized = raw_prs
        
        return prs_standardized
    
    def get_expected_prs_distribution(self, n_simulations: int = 10000, 
                                      seed: int = 42) -> dict:
        """
        Compute expected PRS distribution statistics.
        
        Returns:
            dict with mean, std, percentiles of simulated PRS
        """
        prs_values = self.compute_cohort_prs(n_simulations, seed)
        
        return {
            'mean': float(np.mean(prs_values)),
            'std': float(np.std(prs_values)),
            'median': float(np.median(prs_values)),
            'p5': float(np.percentile(prs_values, 5)),
            'p25': float(np.percentile(prs_values, 25)),
            'p75': float(np.percentile(prs_values, 75)),
            'p95': float(np.percentile(prs_values, 95)),
            'min': float(np.min(prs_values)),
            'max': float(np.max(prs_values))
        }


if __name__ == '__main__':
    # Quick demo
    import yaml
    
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    gwas_path = f"{config['paths']['data']['raw']['gwas']}/epilepsy_snps.csv"
    gwas_df = pd.read_csv(gwas_path)
    
    computer = PRSComputer(gwas_df)
    
    # Compute distribution
    dist = computer.get_expected_prs_distribution()
    print("Expected PRS Distribution (10K simulations):")
    for key, val in dist.items():
        print(f"  {key}: {val:.4f}")
