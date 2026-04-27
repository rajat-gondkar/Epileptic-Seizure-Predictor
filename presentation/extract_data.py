#!/usr/bin/env python3
"""
Extract curated EEG segments for presentation.
Finds 5 specific phenomena from real data:
  1. Normal baseline (calm interictal)
  2. Eye blink artifacts (large frontal spikes → suppressed after preprocessing)
  3. Pre-ictal period (30s before seizure onset)
  4. During seizure (ictal)
  5. Muscle/movement artifact (high-frequency noise)
"""

import json
import sys
from pathlib import Path

import mne
import numpy as np

mne.set_log_level("ERROR")

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)

# Channels to display
DISPLAY_CHANNELS = ["FP1-F7", "F7-T7", "C3-P3", "FZ-CZ"]

# chb01_03.edf seizure: 2996–3036s
# chb01_04.edf seizure: 1467–1494s
EDF_FILE = "chb01_03.edf"
SEIZURE_START = 2996
SEIZURE_END = 3036

SEGMENT_DURATION = 6  # seconds per segment


def load_raw_and_processed(edf_path, channels):
    """Load raw and preprocessed versions of an EDF file."""
    from src.data_pipeline.eeg_preprocessing import EEGPreprocessor, load_config
    config = load_config()
    pre = EEGPreprocessor(config)

    # Raw
    raw_orig = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    available = [ch for ch in channels if ch in raw_orig.ch_names]
    raw_copy = raw_orig.copy()
    raw_copy.pick(available)

    # Processed
    raw_proc = pre.preprocess(raw_orig)
    raw_proc.pick(available)

    sfreq = int(raw_copy.info["sfreq"])
    return raw_copy, raw_proc, available, sfreq


def find_eye_blinks(raw, sfreq):
    """Find eye blink segments — large amplitude spikes in FP1-F7."""
    fp1_idx = raw.ch_names.index("FP1-F7") if "FP1-F7" in raw.ch_names else 0
    data = raw.get_data()[fp1_idx]  # FP1-F7 channel
    data_uv = data * 1e6  # to µV

    # Scan with a sliding window — find segments with large spikes in frontal channel
    # but NOT sustained high amplitude (which would be seizure)
    window = SEGMENT_DURATION * sfreq
    stride = sfreq * 2  # check every 2 seconds
    best_score = 0
    best_start = 100 * sfreq  # default fallback

    for start in range(50 * sfreq, len(data) - window, stride):
        seg = data_uv[start:start + window]
        # Skip if near seizure
        time_s = start / sfreq
        if abs(time_s - SEIZURE_START) < 120:
            continue

        peak = np.max(np.abs(seg))
        std = np.std(seg)
        # Eye blink: high peaks but moderate std (spiky, not sustained)
        if peak > 150 and std < 80 and peak / std > 3:
            score = peak / std
            if score > best_score:
                best_score = score
                best_start = start

    return best_start // sfreq


def find_muscle_artifact(raw, sfreq):
    """Find muscle artifact — high frequency noise, high variance segment."""
    data = raw.get_data()
    data_uv = data * 1e6
    window = SEGMENT_DURATION * sfreq
    stride = sfreq * 2

    best_var = 0
    best_start = 200 * sfreq

    for start in range(50 * sfreq, data.shape[1] - window, stride):
        time_s = start / sfreq
        if abs(time_s - SEIZURE_START) < 120:
            continue

        seg = data_uv[:, start:start + window]
        # High-frequency content: compute variance of diff (approximates HF energy)
        hf_energy = np.mean(np.var(np.diff(seg, axis=1), axis=1))
        total_var = np.mean(np.var(seg, axis=1))

        # Want high HF energy relative to total variance (noisy, not just big swings)
        if hf_energy > best_var and total_var < 5000:
            best_var = hf_energy
            best_start = start

    return best_start // sfreq


def find_normal_baseline(raw, sfreq):
    """Find a calm, normal baseline segment — low variance, far from seizure."""
    data = raw.get_data()
    data_uv = data * 1e6
    window = SEGMENT_DURATION * sfreq
    stride = sfreq * 5

    best_var = float("inf")
    best_start = 300 * sfreq

    for start in range(100 * sfreq, min(data.shape[1] - window, 1500 * sfreq), stride):
        time_s = start / sfreq
        if abs(time_s - SEIZURE_START) < 300:
            continue

        seg = data_uv[:, start:start + window]
        total_var = np.mean(np.var(seg, axis=1))
        peak = np.max(np.abs(seg))

        # Low variance AND no extreme peaks
        if total_var < best_var and peak < 200:
            best_var = total_var
            best_start = start

    return best_start // sfreq


def extract_segment(raw_orig, raw_proc, start_sec, channels, sfreq):
    """Extract a segment and convert to JSON-ready dict."""
    stride = 2  # downsample for frontend
    start_samp = start_sec * sfreq
    end_samp = start_samp + SEGMENT_DURATION * sfreq

    raw_data = raw_orig.get_data(start=start_samp, stop=end_samp)
    proc_data = raw_proc.get_data(start=start_samp, stop=end_samp)

    raw_dict = {}
    proc_dict = {}
    for i, ch in enumerate(channels):
        raw_uv = (raw_data[i, ::stride] * 1e6).tolist()
        proc_uv = (proc_data[i, ::stride] * 1e6).tolist()
        raw_dict[ch] = [round(v, 2) for v in raw_uv]
        proc_dict[ch] = [round(v, 2) for v in proc_uv]

    return raw_dict, proc_dict


def main():
    edf_path = PROJECT / "data" / "raw" / "chb-mit" / "chb01" / EDF_FILE
    print(f"Loading {edf_path.name}...")

    raw_orig, raw_proc, channels, sfreq = load_raw_and_processed(edf_path, DISPLAY_CHANNELS)
    print(f"  Channels: {channels}, sfreq: {sfreq}")

    # Find interesting segments
    print("\nFinding segments...")

    normal_start = find_normal_baseline(raw_orig, sfreq)
    print(f"  Normal baseline: {normal_start}s")

    blink_start = find_eye_blinks(raw_orig, sfreq)
    print(f"  Eye blink artifact: {blink_start}s")

    muscle_start = find_muscle_artifact(raw_orig, sfreq)
    print(f"  Muscle artifact: {muscle_start}s")

    preictal_start = SEIZURE_START - 35  # 35s before seizure
    print(f"  Pre-ictal: {preictal_start}s (seizure at {SEIZURE_START}s)")

    ictal_start = SEIZURE_START + 2  # 2s into seizure
    print(f"  Ictal (seizure): {ictal_start}s")

    # Define segments
    segments = [
        {
            "id": "normal",
            "title": "Normal Baseline",
            "description": "Calm interictal EEG — typical background rhythms with no pathological activity. The preprocessing preserves the underlying signal while reducing minor noise.",
            "start_sec": normal_start,
            "raw_label": "Raw — Background Noise Present",
            "proc_label": "Filtered — Clean Baseline",
        },
        {
            "id": "blink",
            "title": "Eye Blink Artifact",
            "description": "Large amplitude spikes in the frontal channel (FP1-F7) caused by eye blinks. The bandpass and artifact rejection pipeline suppresses these transients while preserving the underlying neural signal.",
            "start_sec": blink_start,
            "raw_label": "Raw — Blink Spikes Visible",
            "proc_label": "Filtered — Artifacts Suppressed",
        },
        {
            "id": "muscle",
            "title": "Muscle / Movement Artifact",
            "description": "High-frequency noise from scalp muscle contractions (EMG contamination). The 70 Hz low-pass filter removes most of this high-frequency artifact.",
            "start_sec": muscle_start,
            "raw_label": "Raw — EMG Contamination",
            "proc_label": "Filtered — HF Noise Removed",
        },
        {
            "id": "preictal",
            "title": "Pre-Ictal Period",
            "description": f"~30 seconds before seizure onset (seizure at {SEIZURE_START}s). Subtle changes in rhythmic activity may emerge. The preprocessing preserves these critical pre-seizure patterns.",
            "start_sec": preictal_start,
            "raw_label": "Raw — Pre-Seizure",
            "proc_label": "Filtered — Patterns Preserved",
        },
        {
            "id": "ictal",
            "title": "During Seizure (Ictal)",
            "description": f"Active seizure activity — high-amplitude rhythmic discharges across channels. Seizure window: {SEIZURE_START}–{SEIZURE_END}s. Preprocessing preserves the seizure morphology.",
            "start_sec": ictal_start,
            "raw_label": "Raw — Seizure Activity",
            "proc_label": "Filtered — Seizure Preserved",
        },
    ]

    # Extract all segments
    result = {
        "file": EDF_FILE,
        "patient": "chb01",
        "sfreq": sfreq // 2,  # after downsampling
        "duration_sec": SEGMENT_DURATION,
        "channels": channels,
        "segments": [],
    }

    for seg in segments:
        print(f"\n  Extracting: {seg['title']} (t={seg['start_sec']}s)...")
        raw_dict, proc_dict = extract_segment(
            raw_orig, raw_proc, seg["start_sec"], channels, sfreq
        )

        # Verify there's actual signal
        for ch in channels:
            r_range = max(raw_dict[ch]) - min(raw_dict[ch])
            p_range = max(proc_dict[ch]) - min(proc_dict[ch])
            print(f"    {ch}: raw range={r_range:.1f}µV, proc range={p_range:.1f}µV")

        result["segments"].append({
            "id": seg["id"],
            "title": seg["title"],
            "description": seg["description"],
            "start_sec": seg["start_sec"],
            "raw_label": seg["raw_label"],
            "proc_label": seg["proc_label"],
            "raw": raw_dict,
            "processed": proc_dict,
        })

    with open(OUT / "eeg_signals.json", "w") as f:
        json.dump(result, f)

    print(f"\n✅ Saved {len(segments)} segments to {OUT / 'eeg_signals.json'}")
    print(f"   File size: {(OUT / 'eeg_signals.json').stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
