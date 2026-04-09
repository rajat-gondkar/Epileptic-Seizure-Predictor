#!/usr/bin/env python3
"""
Master Data Acquisition Script
================================
Orchestrates the entire data acquisition pipeline:
1. CHB-MIT EEG dataset download
2. ClinVar, gnomAD, GWAS genetic data download
3. Verification of all downloads

Usage:
    python scripts/run_acquisition.py [--skip-eeg] [--skip-genetic] [--verify-only]
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.acquire_chbmit import main as acquire_eeg
from scripts.acquire_genetic_data import main as acquire_genetic


def verify_downloads(project_dir):
    """Verify that all expected data files exist."""
    project_dir = Path(project_dir)
    
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    checks = [
        # EEG data
        ("CHB-MIT RECORDS", "data/raw/chb-mit/RECORDS"),
        ("Seizure annotations", "data/raw/chb-mit/seizure_annotations.csv"),
        # Genetic data
        ("ClinVar epilepsy variants", "data/raw/clinvar/epilepsy_variants.csv"),
        ("gnomAD pLI scores", "data/raw/gnomad/pli_scores.csv"),
        ("GWAS epilepsy SNPs", "data/raw/gwas/epilepsy_snps.csv"),
        # Documentation
        ("Access notes", "data/raw/ACCESS_NOTES.md"),
    ]
    
    all_ok = True
    for desc, rel_path in checks:
        full_path = project_dir / rel_path
        if full_path.exists():
            size = full_path.stat().st_size
            size_str = f"{size/1024:.1f} KB" if size > 1024 else f"{size} bytes"
            print(f"  ✅ {desc}: {size_str}")
        else:
            print(f"  ❌ {desc}: MISSING ({rel_path})")
            all_ok = False
    
    # Check for patient directories
    chbmit_dir = project_dir / "data/raw/chb-mit"
    if chbmit_dir.exists():
        patient_dirs = [d for d in chbmit_dir.iterdir() if d.is_dir() and d.name.startswith('chb')]
        edf_count = sum(1 for d in patient_dirs for f in d.glob('*.edf'))
        print(f"\n  EEG Data Summary:")
        print(f"    Patient directories: {len(patient_dirs)}")
        print(f"    Total EDF files: {edf_count}")
    
    if all_ok:
        print("\n  ✅ ALL CHECKS PASSED")
    else:
        print("\n  ⚠️  Some files are missing — check the errors above")
    
    return all_ok


def main():
    parser = argparse.ArgumentParser(description='Master Data Acquisition Pipeline')
    parser.add_argument('--skip-eeg', action='store_true', help='Skip EEG download')
    parser.add_argument('--skip-genetic', action='store_true', help='Skip genetic data download')
    parser.add_argument('--verify-only', action='store_true', help='Only verify existing downloads')
    parser.add_argument('--project-dir', type=str, default='.', help='Project root directory')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MASTER DATA ACQUISITION PIPELINE")
    print("EEG-Genetic Fusion for Epilepsy Prediction")
    print("=" * 60)
    
    if not args.verify_only:
        if not args.skip_eeg:
            print("\n[STEP 1/2] Acquiring EEG Data...")
            # Use sys.argv manipulation to pass through correctly
            original_argv = sys.argv
            sys.argv = ['acquire_chbmit.py']
            try:
                acquire_eeg()
            except SystemExit:
                pass
            sys.argv = original_argv
        
        if not args.skip_genetic:
            print("\n[STEP 2/2] Acquiring Genetic Data...")
            original_argv = sys.argv
            sys.argv = ['acquire_genetic_data.py']
            try:
                acquire_genetic()
            except SystemExit:
                pass
            sys.argv = original_argv
    
    # Verify
    verify_downloads(args.project_dir)


if __name__ == '__main__':
    main()
