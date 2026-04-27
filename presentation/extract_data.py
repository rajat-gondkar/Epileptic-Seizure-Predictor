#!/usr/bin/env python3
"""
Extract real data samples for the presentation frontend.
Outputs JSON files that the HTML/JS frontend can load.
"""

import json
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd

mne.set_log_level("ERROR")

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)


def extract_eeg_signals():
    """Extract raw vs preprocessed EEG for side-by-side comparison."""
    from src.data_pipeline.eeg_preprocessing import EEGPreprocessor, load_config

    config = load_config()
    pre = EEGPreprocessor(config)

    # Pick a seizure file for interesting data
    edf_path = PROJECT / "data" / "raw" / "chb-mit" / "chb01" / "chb01_03.edf"
    if not edf_path.exists():
        # fallback to any available
        edfs = sorted((PROJECT / "data" / "raw" / "chb-mit" / "chb01").glob("*.edf"))
        edf_path = edfs[0] if edfs else None
    if not edf_path:
        print("No EDF files found"); return

    print(f"Loading: {edf_path.name}")

    # Channels to display (subset for readability)
    display_channels = ["FP1-F7", "F7-T7", "C3-P3", "FZ-CZ"]

    # ── RAW signal (no preprocessing) ──
    raw_orig = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    available = [ch for ch in display_channels if ch in raw_orig.ch_names]
    raw_orig.pick(available)

    # Take 10 seconds starting at second 100 (skip initial artifacts)
    sfreq = int(raw_orig.info["sfreq"])
    start = 100 * sfreq
    stop  = start + 10 * sfreq
    raw_data = raw_orig.get_data(start=start, stop=stop)  # [n_ch, n_samples]

    # ── PREPROCESSED signal ──
    raw_proc = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    raw_proc = pre.preprocess(raw_proc)
    raw_proc.pick(available)
    proc_data = raw_proc.get_data(start=start, stop=stop)

    # Convert to JSON-friendly format (downsample for frontend performance)
    # Keep every 2nd sample → 128 Hz effective, still smooth
    stride = 2
    result = {
        "sfreq": sfreq // stride,
        "duration_sec": 10,
        "channels": available,
        "file": edf_path.name,
        "patient": "chb01",
        "raw": {},
        "processed": {},
    }

    for i, ch in enumerate(available):
        # Scale to microvolts for display
        raw_uv = (raw_data[i, ::stride] * 1e6).tolist()
        proc_uv = (proc_data[i, ::stride] * 1e6).tolist()
        result["raw"][ch] = [round(v, 2) for v in raw_uv]
        result["processed"][ch] = [round(v, 2) for v in proc_uv]

    with open(OUT / "eeg_signals.json", "w") as f:
        json.dump(result, f)

    print(f"  EEG signals: {len(available)} channels × {len(raw_uv)} samples")

    # ── Also extract feature statistics for display ──
    feat_path = PROJECT / "data" / "processed" / "eeg_features" / "chb01_features.npy"
    lab_path  = PROJECT / "data" / "processed" / "eeg_features" / "chb01_labels.npy"
    if feat_path.exists():
        features = np.load(feat_path)
        labels   = np.load(lab_path)
        # Average features across channels for display
        n_ch = 17
        reshaped = features[:500].reshape(-1, n_ch, 13)  # first 500 epochs
        global_feats = reshaped.mean(axis=1)  # [500, 13]

        feature_names = [
            "Delta Power", "Theta Power", "Alpha Power", "Beta Power",
            "Gamma Power", "α/β Ratio", "θ/α Ratio", "Spike Rate",
            "Variance", "Hjorth Activity", "Hjorth Mobility",
            "Hjorth Complexity", "Sample Entropy"
        ]

        feat_stats = {}
        for i, name in enumerate(feature_names):
            vals = global_feats[:, i]
            feat_stats[name] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

        epoch_info = {
            "total_epochs": int(len(labels)),
            "preictal": int((labels == 1).sum()),
            "interictal": int((labels == 0).sum()),
            "feature_dim": int(features.shape[1]),
            "feature_stats": feat_stats,
        }

        with open(OUT / "feature_stats.json", "w") as f:
            json.dump(epoch_info, f, indent=2)
        print(f"  Feature stats extracted")


def extract_genetic_data():
    """Extract genetic profiles for display."""
    gen_path = PROJECT / "data" / "processed" / "genetic_vectors" / "genetic_profiles.csv"
    if not gen_path.exists():
        print("No genetic profiles found"); return

    df = pd.read_csv(gen_path)

    patients_used = ["chb01", "chb03", "chb05", "chb06", "chb08", "chb10", "chb16", "chb20"]
    df = df[df["patient_id"].isin(patients_used)]

    result = {
        "columns": list(df.columns),
        "data": df.to_dict(orient="records"),
        "gene_names": ["SCN1A", "SCN8A", "KCNQ2", "SCN2A", "KCNT1",
                        "DEPDC5", "PCDH19", "GRIN2A", "GABRA1"],
    }

    with open(OUT / "genetic_profiles.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Genetic profiles: {len(df)} patients")


def extract_ctgan_data():
    """Extract CTGAN real vs synthetic comparison data."""
    real_path = PROJECT / "data" / "processed" / "synthetic" / "real_summary_dataset.csv"
    syn_path  = PROJECT / "data" / "processed" / "synthetic" / "synthetic_records.csv"
    val_path  = PROJECT / "data" / "processed" / "synthetic" / "validation_report.json"

    if not real_path.exists() or not syn_path.exists():
        print("No CTGAN data found"); return

    real_df = pd.read_csv(real_path)
    syn_df  = pd.read_csv(syn_path)

    # Log-transform stems for un-logging real data
    log_stems = ["delta_power", "theta_power", "alpha_power", "beta_power",
                 "gamma_power", "variance", "hjorth_activity"]

    for stem in log_stems:
        for suffix in ["_mean", "_std"]:
            col = f"{stem}{suffix}"
            if col in real_df.columns:
                real_df[col] = np.power(10, real_df[col])

    # Compare key features
    compare_cols = [
        "delta_power_mean", "theta_power_mean", "alpha_power_mean",
        "beta_power_mean", "gamma_power_mean",
        "spike_rate_mean", "sample_entropy_mean",
        "preictal_ratio", "polygenic_risk_score",
    ]

    comparisons = {}
    for col in compare_cols:
        if col in real_df.columns and col in syn_df.columns:
            r = real_df[col].dropna()
            s = syn_df[col].dropna()
            comparisons[col] = {
                "real_mean": float(r.mean()),
                "real_std": float(r.std()),
                "syn_mean": float(s.mean()),
                "syn_std": float(s.std()),
                # Histogram data (20 bins)
                "real_hist": np.histogram(r, bins=20)[0].tolist(),
                "syn_hist": np.histogram(s, bins=20)[0].tolist(),
                "bin_edges": np.histogram(r, bins=20)[1].tolist(),
            }

    # Seizure distribution
    seizure_dist = {
        "real": {"seizure": int(real_df["has_seizure"].sum()),
                 "no_seizure": int((real_df["has_seizure"] == 0).sum())},
        "syn":  {"seizure": int(syn_df["has_seizure"].sum()),
                 "no_seizure": int((syn_df["has_seizure"] == 0).sum())},
    }

    # Mutation distribution
    mut_cols = [c for c in real_df.columns if c.endswith("_mutation")]
    mutations = {}
    for col in mut_cols:
        mutations[col] = {
            "real_rate": float(real_df[col].mean()),
            "syn_rate": float(syn_df[col].mean()),
        }

    # Validation report
    val_report = {}
    if val_path.exists():
        with open(val_path) as f:
            val_report = json.load(f)

    result = {
        "real_count": len(real_df),
        "syn_count": len(syn_df),
        "real_patients": int(real_df["patient_id"].nunique()) if "patient_id" in real_df else 0,
        "comparisons": comparisons,
        "seizure_dist": seizure_dist,
        "mutations": mutations,
        "validation": val_report,
    }

    # Also extract a synthetic EEG-like signal for animation
    # Use the synthetic band power values to generate a simulated waveform
    syn_sample = syn_df.head(5)
    syn_signals = {}
    for idx, row in syn_sample.iterrows():
        # Generate a signal from the band powers (for visual illustration)
        t = np.linspace(0, 5, 640)  # 5 seconds at 128 Hz
        signal = np.zeros_like(t)
        if "delta_power_mean" in row:
            signal += np.sqrt(abs(row.get("delta_power_mean", 0))) * 1e5 * np.sin(2 * np.pi * 2 * t)
            signal += np.sqrt(abs(row.get("theta_power_mean", 0))) * 1e5 * np.sin(2 * np.pi * 6 * t)
            signal += np.sqrt(abs(row.get("alpha_power_mean", 0))) * 1e5 * np.sin(2 * np.pi * 10 * t)
            signal += np.sqrt(abs(row.get("beta_power_mean", 0))) * 1e5 * np.sin(2 * np.pi * 20 * t)
        syn_signals[f"synthetic_{idx}"] = [round(v, 3) for v in signal.tolist()]

    result["syn_signals"] = syn_signals
    result["syn_signal_sfreq"] = 128

    with open(OUT / "ctgan_results.json", "w") as f:
        json.dump(result, f)

    print(f"  CTGAN: {len(real_df)} real, {len(syn_df)} synthetic, {len(comparisons)} features compared")


if __name__ == "__main__":
    print("Extracting presentation data...")
    print()
    extract_eeg_signals()
    extract_genetic_data()
    extract_ctgan_data()
    print("\n✅ All data extracted to presentation/data/")
