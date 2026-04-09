#!/usr/bin/env python3
"""
CHB-MIT EEG Dataset Acquisition Script
=======================================
Downloads CHB-MIT Scalp EEG dataset from PhysioNet for specified patients.
Parses seizure annotations from summary files and creates seizure_annotations.csv.

Usage:
    python scripts/acquire_chbmit.py [--patients chb01 chb02 ...] [--data-dir data/raw/chb-mit]
"""

import os
import re
import sys
import argparse
import csv
import time
from pathlib import Path

import yaml
import requests
from tqdm import tqdm


# PhysioNet base URL for CHB-MIT
PHYSIONET_BASE = "https://physionet.org/files/chbmit/1.0.0"


def load_config(config_path="configs/config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_patient_file_list(patient_id):
    """
    Get list of EDF files for a given patient from PhysioNet RECORDS file.
    Falls back to sequential file naming if RECORDS unavailable.
    """
    # Try to get RECORDS file listing
    records_url = f"{PHYSIONET_BASE}/RECORDS"
    try:
        resp = requests.get(records_url, timeout=30)
        resp.raise_for_status()
        all_records = resp.text.strip().split('\n')
        # Filter for this patient
        patient_files = [r.strip() for r in all_records if r.strip().startswith(patient_id + '/')]
        if patient_files:
            return patient_files
    except Exception as e:
        print(f"  Warning: Could not fetch RECORDS file: {e}")
    
    return []


def download_file(url, dest_path, max_retries=3):
    """Download a file with progress bar and retry logic."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True  # Already downloaded
    
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            
            total_size = int(resp.headers.get('content-length', 0))
            
            with open(dest_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, 
                         desc=dest_path.name, leave=False) as pbar:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            return True
            
        except Exception as e:
            print(f"  Attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    
    return False


def download_patient_data(patient_id, data_dir):
    """Download all EDF files for a specific patient."""
    patient_dir = Path(data_dir) / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Downloading data for patient: {patient_id}")
    print(f"{'='*60}")
    
    # Get file list
    file_list = get_patient_file_list(patient_id)
    
    if not file_list:
        print(f"  No files found for {patient_id}")
        return False
    
    print(f"  Found {len(file_list)} files")
    
    success_count = 0
    for file_path in file_list:
        url = f"{PHYSIONET_BASE}/{file_path}"
        dest = Path(data_dir) / file_path
        
        if download_file(url, dest):
            success_count += 1
        else:
            print(f"  FAILED: {file_path}")
    
    print(f"  Successfully downloaded {success_count}/{len(file_list)} files")
    
    # Also download the summary file for this patient
    summary_file = f"{patient_id}/{patient_id}-summary.txt"
    summary_url = f"{PHYSIONET_BASE}/{summary_file}"
    summary_dest = Path(data_dir) / summary_file
    download_file(summary_url, summary_dest)
    
    return success_count > 0


def download_global_files(data_dir):
    """Download global CHB-MIT files (RECORDS, seizure summary, etc.)."""
    global_files = [
        "RECORDS",
        "RECORDS-WITH-SEIZURES",
        "SUBJECT-INFO",
    ]
    
    for fname in global_files:
        url = f"{PHYSIONET_BASE}/{fname}"
        dest = Path(data_dir) / fname
        print(f"Downloading {fname}...")
        download_file(url, dest)


def parse_seizure_summary(data_dir, patients):
    """
    Parse per-patient summary files to extract seizure annotations.
    
    Returns a list of dicts with: patient_id, file, seizure_start_sec, seizure_end_sec
    """
    annotations = []
    
    for patient_id in patients:
        summary_path = Path(data_dir) / patient_id / f"{patient_id}-summary.txt"
        
        if not summary_path.exists():
            print(f"  Warning: No summary file for {patient_id}")
            continue
        
        with open(summary_path, 'r') as f:
            content = f.read()
        
        # Parse the summary file
        # Format varies slightly but generally:
        # File Name: chb01_03.edf
        # Number of Seizures in File: 1
        # Seizure Start Time: 2996 seconds
        # Seizure End Time: 3036 seconds
        
        current_file = None
        lines = content.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Match file name
            file_match = re.match(r'File Name:\s*(.+\.edf)', line, re.IGNORECASE)
            if file_match:
                current_file = file_match.group(1).strip()
                i += 1
                continue
            
            # Match number of seizures
            seizure_count_match = re.match(r'Number of Seizures in File:\s*(\d+)', line, re.IGNORECASE)
            if seizure_count_match:
                num_seizures = int(seizure_count_match.group(1))
                
                if num_seizures > 0 and current_file:
                    # Read the seizure start/end pairs
                    for s in range(num_seizures):
                        # Look for start time
                        i += 1
                        while i < len(lines):
                            start_match = re.match(
                                r'Seizure\s*\d*\s*Start Time:\s*(\d+)\s*seconds?', 
                                lines[i].strip(), re.IGNORECASE
                            )
                            if start_match:
                                start_sec = int(start_match.group(1))
                                break
                            i += 1
                        
                        # Look for end time
                        i += 1
                        while i < len(lines):
                            end_match = re.match(
                                r'Seizure\s*\d*\s*End Time:\s*(\d+)\s*seconds?',
                                lines[i].strip(), re.IGNORECASE
                            )
                            if end_match:
                                end_sec = int(end_match.group(1))
                                break
                            i += 1
                        
                        annotations.append({
                            'patient_id': patient_id,
                            'file': current_file,
                            'seizure_start_sec': start_sec,
                            'seizure_end_sec': end_sec
                        })
                        
                        print(f"  Found seizure: {patient_id}/{current_file} "
                              f"[{start_sec}s - {end_sec}s] "
                              f"(duration: {end_sec - start_sec}s)")
            
            i += 1
    
    return annotations


def save_seizure_annotations(annotations, output_path):
    """Save seizure annotations to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['patient_id', 'file', 'seizure_start_sec', 'seizure_end_sec'])
        writer.writeheader()
        writer.writerows(annotations)
    
    print(f"\nSaved {len(annotations)} seizure annotations to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Download CHB-MIT EEG Dataset')
    parser.add_argument('--patients', nargs='+', 
                       default=None,
                       help='Patient IDs to download (e.g., chb01 chb02)')
    parser.add_argument('--data-dir', type=str, 
                       default='data/raw/chb-mit',
                       help='Directory to save downloaded data')
    parser.add_argument('--config', type=str,
                       default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--annotations-only', action='store_true',
                       help='Only parse annotations (skip download)')
    
    args = parser.parse_args()
    
    # Load config for default patients
    if args.patients is None:
        try:
            config = load_config(args.config)
            args.patients = config['dataset']['chb_mit_patients']
        except Exception:
            args.patients = ['chb01', 'chb02', 'chb03', 'chb04', 'chb05']
    
    print("=" * 60)
    print("CHB-MIT EEG Dataset Acquisition")
    print("=" * 60)
    print(f"Patients: {', '.join(args.patients)}")
    print(f"Data directory: {args.data_dir}")
    print()
    
    if not args.annotations_only:
        # Download global files
        print("Downloading global files...")
        download_global_files(args.data_dir)
        
        # Download patient data
        for patient_id in args.patients:
            download_patient_data(patient_id, args.data_dir)
    
    # Parse seizure annotations
    print("\n" + "=" * 60)
    print("Parsing seizure annotations...")
    print("=" * 60)
    
    annotations = parse_seizure_summary(args.data_dir, args.patients)
    
    # Save annotations
    annotations_path = Path(args.data_dir) / 'seizure_annotations.csv'
    save_seizure_annotations(annotations, annotations_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("ACQUISITION SUMMARY")
    print("=" * 60)
    print(f"Patients downloaded: {len(args.patients)}")
    print(f"Total seizures found: {len(annotations)}")
    for patient_id in args.patients:
        patient_seizures = [a for a in annotations if a['patient_id'] == patient_id]
        print(f"  {patient_id}: {len(patient_seizures)} seizures")
    print(f"Annotations saved: {annotations_path}")
    print("Done!")


if __name__ == '__main__':
    main()
