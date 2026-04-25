#!/usr/bin/env python3
"""
Selective CHB-MIT Download & Preprocessing
============================================
Downloads ONLY seizure files + a few interictal files for additional patients,
then runs the full EEG preprocessing pipeline on them.

Saves ~70% disk space compared to downloading all files.

Usage:
    python scripts/selective_download_and_preprocess.py          # full run
    python scripts/selective_download_and_preprocess.py --dry-run # show plan only
    python scripts/selective_download_and_preprocess.py --download-only  # no preprocessing
"""

import argparse
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

# ── Add project root to path ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PHYSIONET_BASE = "https://physionet.org/files/chbmit/1.0.0"

# ── Patients to add (already have chb01, chb03, chb05) ──
NEW_PATIENTS = ["chb06", "chb08", "chb10", "chb16", "chb20"]
MAX_INTERICTAL_FILES = 8  # keep download small


# ============================================================
# 1. Parse summary file → seizure timings + file lists
# ============================================================
def parse_summary(summary_path):
    """Parse a CHB-MIT summary file to extract seizure annotations and file lists."""
    text = Path(summary_path).read_text()
    seizures = []
    all_files = []
    current_file = None
    pending_start = None

    for line in text.split("\n"):
        line = line.strip()

        m = re.match(r"File Name:\s*(\S+)", line)
        if m:
            current_file = m.group(1)
            all_files.append(current_file)

        m = re.match(r"Seizure\s*\d*\s*Start Time:\s*(\d+)", line)
        if m:
            pending_start = int(m.group(1))

        m = re.match(r"Seizure\s*\d*\s*End Time:\s*(\d+)", line)
        if m and pending_start is not None:
            seizures.append({
                "file": current_file,
                "start": pending_start,
                "end": int(m.group(1)),
            })
            pending_start = None

    seizure_files = sorted(set(s["file"] for s in seizures))
    interictal_files = [f for f in all_files if f not in seizure_files]

    return seizures, seizure_files, interictal_files


# ============================================================
# 2. Download a single EDF file with resume support
# ============================================================
def download_edf(patient_id, filename, data_dir, dry_run=False):
    """Download one EDF file. Skips if already present and correct size."""
    url = f"{PHYSIONET_BASE}/{patient_id}/{filename}"
    dest = data_dir / patient_id / filename

    if dest.exists() and dest.stat().st_size > 30_000_000:  # > 40 MB = likely complete EDF
        return "skip"

    if dry_run:
        print(f"  [DRY-RUN] Would download: {url}")
        return "dry"

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {filename} ...", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, str(dest))
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f" {size_mb:.1f} MB ✓")
        return "ok"
    except Exception as e:
        print(f" FAILED: {e}")
        return "fail"


# ============================================================
# 3. Update seizure_annotations.csv
# ============================================================
def update_annotations(data_dir, patient_id, seizures):
    """Append seizure annotations for a new patient."""
    ann_path = data_dir / "seizure_annotations.csv"

    # Read existing
    if ann_path.exists():
        existing = ann_path.read_text()
        existing_lines = [l.strip() for l in existing.strip().split("\n")]
    else:
        existing_lines = ["patient_id,file,seizure_start_sec,seizure_end_sec"]

    # Check if already present
    if any(patient_id in line for line in existing_lines[1:]):
        return  # already in file

    for s in seizures:
        existing_lines.append(f"{patient_id},{s['file']},{s['start']},{s['end']}")

    ann_path.write_text("\n".join(existing_lines) + "\n")


# ============================================================
# 4. Generate genetic profile for new patient
# ============================================================
def ensure_genetic_profile(patient_id):
    """Add a genetic profile for the patient if not already present."""
    gen_path = PROJECT_ROOT / "data" / "processed" / "genetic_vectors" / "genetic_profiles.csv"
    if not gen_path.exists():
        return

    import pandas as pd
    df = pd.read_csv(gen_path)
    if patient_id in df["patient_id"].values:
        return

    # Use the existing genetic feature engineering module
    from src.data_pipeline.genetic_feature_engineering import (
        build_genetic_vector, load_config as gen_load_config, TARGET_GENES
    )

    config = gen_load_config()

    # Load reference datasets
    gnomad_path = PROJECT_ROOT / config["paths"]["data"]["raw"]["gnomad"] / "pli_scores.csv"
    gwas_path = PROJECT_ROOT / config["paths"]["data"]["raw"]["gwas"] / "epilepsy_snps.csv"

    gnomad_df = pd.read_csv(gnomad_path) if gnomad_path.exists() else None
    gwas_df = pd.read_csv(gwas_path) if gwas_path.exists() else None

    vector = build_genetic_vector(
        patient_id, gnomad_df=gnomad_df, gwas_df=gwas_df, simulated=True
    )

    new_row = {"patient_id": patient_id}
    for i, gene in enumerate(TARGET_GENES):
        new_row[f"{gene}_mutation"] = int(vector[i])
    new_row["SCN1A_pLI"] = float(vector[9])
    new_row["SCN8A_pLI"] = float(vector[10])
    new_row["polygenic_risk_score"] = float(vector[11])

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(gen_path, index=False)
    print(f"  Added genetic profile for {patient_id}")



# ============================================================
# 5. Run EEG preprocessing for one patient
# ============================================================
def preprocess_patient(patient_id):
    """Run the full EEG preprocessing pipeline for one patient."""
    from src.data_pipeline.eeg_preprocessing import process_patient, load_config
    config = load_config()
    stats = process_patient(patient_id, config)
    return stats


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Selective CHB-MIT download & preprocess")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without downloading")
    parser.add_argument("--download-only", action="store_true", help="Download without preprocessing")
    parser.add_argument("--patients", nargs="+", default=NEW_PATIENTS,
                        help=f"Patients to process (default: {NEW_PATIENTS})")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "data" / "raw" / "chb-mit"

    print("=" * 60)
    print("SELECTIVE CHB-MIT DOWNLOAD & PREPROCESSING")
    print("=" * 60)
    print(f"Patients: {args.patients}")
    print(f"Max interictal files per patient: {MAX_INTERICTAL_FILES}")
    print(f"Dry run: {args.dry_run}")
    print()

    total_downloaded = 0
    total_skipped = 0
    total_failed = 0
    all_stats = []

    for patient_id in args.patients:
        print(f"\n{'─'*60}")
        print(f"Patient: {patient_id}")
        print(f"{'─'*60}")

        # Step 1: Ensure summary file exists
        summary_path = data_dir / patient_id / f"{patient_id}-summary.txt"
        if not summary_path.exists():
            print(f"  Downloading summary file...")
            download_edf(patient_id, f"{patient_id}-summary.txt", data_dir, dry_run=args.dry_run)
            # Re-download as it's not an EDF
            if not args.dry_run:
                url = f"{PHYSIONET_BASE}/{patient_id}/{patient_id}-summary.txt"
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(url, str(summary_path))

        if not summary_path.exists() and not args.dry_run:
            print(f"  ERROR: Could not get summary file, skipping")
            continue

        if args.dry_run and not summary_path.exists():
            print(f"  [DRY-RUN] Would parse summary and download files")
            continue

        # Step 2: Parse summary
        seizures, seizure_files, interictal_files = parse_summary(summary_path)
        selected_interictal = interictal_files[:MAX_INTERICTAL_FILES]
        all_files = seizure_files + selected_interictal

        print(f"  Seizure files: {len(seizure_files)}")
        print(f"  Interictal files (selected): {len(selected_interictal)}")
        print(f"  Total to download: {len(all_files)} (~{len(all_files)*50} MB)")

        # Step 3: Download
        for fname in all_files:
            result = download_edf(patient_id, fname, data_dir, dry_run=args.dry_run)
            if result == "ok":
                total_downloaded += 1
            elif result == "skip":
                total_skipped += 1
            elif result == "fail":
                total_failed += 1

        if args.dry_run:
            continue

        # Step 4: Update seizure annotations
        update_annotations(data_dir, patient_id, seizures)
        print(f"  Updated seizure annotations ({len(seizures)} seizures)")

        # Step 5: Generate genetic profile
        ensure_genetic_profile(patient_id)

        # Step 6: Preprocess
        if not args.download_only:
            feat_dir = PROJECT_ROOT / "data" / "processed" / "eeg_features"
            already_done = all(
                (feat_dir / f"{patient_id}_{s}.npy").exists()
                for s in ["features", "labels", "sequences"]
            )
            if already_done:
                print(f"  ⏭ Preprocessing already complete for {patient_id} — skipping")
            else:
                print(f"\n  Starting preprocessing for {patient_id}...")
                stats = preprocess_patient(patient_id)
                if stats:
                    all_stats.append(stats)


    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Downloaded: {total_downloaded}")
    print(f"  Skipped (already existed): {total_skipped}")
    print(f"  Failed: {total_failed}")

    if all_stats:
        import pandas as pd
        df = pd.DataFrame(all_stats)
        print(f"\n{df.to_string(index=False)}")
        print(f"\n  Total new epochs: {df['total_epochs'].sum():,}")
        print(f"  Total preictal: {df['preictal_epochs'].sum():,}")

    print(f"\n✅ Done at {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
