#!/usr/bin/env python3
"""
EEG Preprocessing & Feature Extraction Pipeline
==================================================
Processes raw EDF files from CHB-MIT dataset through:
1. Signal preprocessing (filtering, re-referencing, artifact rejection)
2. Epoching with seizure-aware labeling (interictal vs preictal)
3. Feature extraction (234-dimensional spectral/statistical features)
4. Raw time-series tensor extraction for LSTM

Usage:
    python src/data_pipeline/eeg_preprocessing.py --patient chb01
    python src/data_pipeline/eeg_preprocessing.py --all
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import mne
from scipy import signal as scipy_signal
from scipy.integrate import trapezoid as scipy_trapz
from tqdm import tqdm

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
mne.set_log_level('ERROR')


# ============================================================
# Configuration
# ============================================================
def load_config(config_path=None):
    """Load configuration from YAML. Searches upward for configs/config.yaml."""
    if config_path is None:
        candidates = [
            Path('configs/config.yaml'),
            Path(__file__).parent.parent.parent / 'configs' / 'config.yaml',
        ]
        for p in candidates:
            if p.exists():
                config_path = p
                break
        else:
            raise FileNotFoundError("configs/config.yaml not found")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ============================================================
# 1. EEG Signal Preprocessing
# ============================================================
class EEGPreprocessor:
    """
    Preprocesses raw EDF files from CHB-MIT dataset.

    Pipeline:
        1. Load EDF file, deduplicate channels
        2. Select standard 18-channel 10-20 system subset
        3. Bandpass filter: 0.5-70 Hz (4th order Butterworth, zero-phase)
        4. Notch filter: 60 Hz
        5. Re-reference to Common Average Reference (CAR)
        6. Artifact rejection (peak-to-peak > 150 µV)
        7. Epoch into 5-second windows with 1-second stride
    """

    def __init__(self, config=None):
        if config is None:
            config = load_config()

        self.eeg_config = config['eeg']
        self.target_channels = self.eeg_config['channels']  # 18 standard channels
        self.sfreq = self.eeg_config['sampling_rate']
        self.epoch_length = self.eeg_config['epoch_length_sec']
        self.epoch_stride = self.eeg_config['epoch_stride_sec']
        self.preictal_window = self.eeg_config['preictal_window_min'] * 60  # → seconds
        self.postictal_exclusion = self.eeg_config['postictal_exclusion_min'] * 60
        self.bandpass_low = self.eeg_config['bandpass_low']
        self.bandpass_high = self.eeg_config['bandpass_high']
        self.notch_freq = self.eeg_config['notch_freq']
        # CHB-MIT EDFs store data in volts. 500 µV → 5e-4 V is a practical
        # epilepsy EEG threshold (150 µV is too tight; many valid epochs get dropped).
        base_threshold_uv = self.eeg_config.get('artifact_threshold_uv', 500)
        self.artifact_threshold = base_threshold_uv * 1e-6  # µV → V
        self.bands = self.eeg_config['bands']

    def load_edf(self, filepath):
        """
        Load an EDF file, deduplicate channel names, then select the
        standard 18-channel 10-20 subset.

        Returns:
            raw: MNE Raw object with exactly the target channels, or None on failure.
        """
        filepath = str(filepath)

        try:
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
        except Exception as e:
            print(f"    Error loading {filepath}: {e}")
            return None

        # ── Step 1: deduplicate channel names ───────────────────────────
        # CHB-MIT has 'T8-P8' duplicated → MNE renames them T8-P8-0, T8-P8-1
        # Keep only the first occurrence; drop -1 variant
        ch_names_orig = raw.ch_names[:]
        drop_channels = []
        seen = {}
        for ch in ch_names_orig:
            # Strip MNE-added suffix e.g. "-0", "-1"
            base = ch.rsplit('-', 1)
            if len(base) == 2 and base[1].isdigit():
                canonical = base[0]
            else:
                canonical = ch
            if canonical in seen:
                drop_channels.append(ch)   # duplicate → drop
            else:
                seen[canonical] = ch

        if drop_channels:
            raw = raw.drop_channels(drop_channels)

        # ── Step 2: match raw channel names to our 18 target channels ───
        target_upper = {t.upper(): t for t in self.target_channels}
        selected = []   # final raw channel names we want
        rename_map = {}

        for ch in raw.ch_names:
            # Normalise: strip whitespace, collapse separators
            ch_norm = ch.strip().upper().replace(' - ', '-').replace(' ', '-').replace('.', '')
            if ch_norm in target_upper:
                selected.append(ch)
                rename_map[ch] = target_upper[ch_norm]

        if len(selected) < 10:
            # Fallback: fuzzy match (strip hyphens and compare)
            for ch in raw.ch_names:
                if ch in selected:
                    continue
                ch_stripped = ch.strip().upper().replace('-', '').replace(' ', '').replace('.', '')
                for tgt_upper, tgt_orig in target_upper.items():
                    tgt_stripped = tgt_upper.replace('-', '')
                    if ch_stripped == tgt_stripped and tgt_orig not in rename_map.values():
                        selected.append(ch)
                        rename_map[ch] = tgt_orig
                        break

        if len(selected) == 0:
            print(f"    Warning: no matching channels in {filepath}. Available: {raw.ch_names[:5]}")
            return None

        try:
            raw = raw.pick(selected)
            raw = raw.rename_channels(rename_map)
        except Exception as e:
            print(f"    Error selecting/renaming channels: {e}")
            return None

        return raw

    def preprocess(self, raw):
        """Apply bandpass, notch, and CAR to a loaded Raw object."""
        if raw is None:
            return None

        # Bandpass 0.5–70 Hz (4th order Butterworth, zero-phase)
        raw = raw.filter(
            l_freq=self.bandpass_low,
            h_freq=self.bandpass_high,
            method='iir',
            iir_params={'order': 4, 'ftype': 'butter'},
            verbose=False,
        )

        # Notch at 60 Hz
        raw = raw.notch_filter(freqs=self.notch_freq, verbose=False)

        # Common Average Reference
        raw = raw.set_eeg_reference('average', verbose=False)

        return raw

    def create_epochs(self, raw, seizure_times=None):
        """
        Sliding-window epoching with seizure-aware labelling.

        Labels:
            
        Returns:
            epochs_data: np.array [N, samples, channels]
            labels: np.array [N]
            valid_mask: np.array [N] - True for usable epochs
        """
        if raw is None:
            return None, None, None

        data = raw.get_data()           # [C, T]
        n_channels, n_total = data.shape

        samples_per_epoch = int(self.epoch_length * self.sfreq)
        stride_samples    = int(self.epoch_stride  * self.sfreq)

        if seizure_times is None:
            seizure_times = []

        epoch_starts = range(0, n_total - samples_per_epoch + 1, stride_samples)

        epochs_list, labels_list, valid_list = [], [], []

        for start in epoch_starts:
            end   = start + samples_per_epoch
            epoch = data[:, start:end]   # [C, samples]

            # Artifact rejection
            if np.any(np.ptp(epoch, axis=1) > self.artifact_threshold):
                epochs_list.append(epoch.T)
                labels_list.append(-1)
                valid_list.append(False)
                continue

            start_sec = start / self.sfreq
            end_sec   = end   / self.sfreq
            mid_sec   = (start_sec + end_sec) / 2

            label, is_valid = 0, True

            for sz_start, sz_end in seizure_times:
                if start_sec < sz_end and end_sec > sz_start:          # ictal
                    label, is_valid = -1, False; break
                if start_sec >= sz_end and start_sec < sz_end + self.postictal_exclusion:
                    label, is_valid = -1, False; break                  # postictal
                if (sz_start - self.preictal_window) <= mid_sec < sz_start:
                    label = 1                                           # preictal

            epochs_list.append(epoch.T)
            labels_list.append(label)
            valid_list.append(is_valid)

        if not epochs_list:
            return None, None, None

        return (
            np.array(epochs_list, dtype=np.float32),
            np.array(labels_list, dtype=np.int8),
            np.array(valid_list,  dtype=bool),
        )


# ============================================================
# 2. Feature Extraction
# ============================================================
class EEGFeatureExtractor:
    """
    Extracts 13 spectral/statistical features per channel → 234-dim vector.

    Features per channel:
        [0-4]  Band power: Delta, Theta, Alpha, Beta, Gamma
        [5]    Alpha/Beta ratio
        [6]    Theta/Alpha ratio
        [7]    Spike rate (peaks > 3 SD / sec)
        [8]    Signal variance
        [9-11] Hjorth: Activity, Mobility, Complexity
        [12]   Sample Entropy (vectorised, ~256-point downsample)

    Total: 13 × 18 channels = 234 dimensions
    """

    def __init__(self, config=None):
        if config is None:
            config = load_config()
        self.sfreq = config['eeg']['sampling_rate']
        self.bands = config['eeg']['bands']

    def _band_power(self, x, band):
        freqs, psd = scipy_signal.welch(x, fs=self.sfreq,
                                        nperseg=min(len(x), 256), noverlap=128)
        mask = (freqs >= band[0]) & (freqs <= band[1])
        return float(scipy_trapz(psd[mask], freqs[mask]))

    def _spike_rate(self, x):
        thresh = np.mean(x) + 3 * np.std(x)
        peaks, _ = scipy_signal.find_peaks(x, height=thresh)
        return len(peaks) / (len(x) / self.sfreq)

    def _hjorth(self, x):
        act = float(np.var(x))
        d1  = np.diff(x);  d2 = np.diff(d1)
        mob    = float(np.sqrt(np.var(d1) / act))            if act        > 0 else 0.0
        mob_d1 = float(np.sqrt(np.var(d2) / np.var(d1)))    if np.var(d1) > 0 else 0.0
        cmp    = mob_d1 / mob                                 if mob        > 0 else 0.0
        return act, mob, cmp

    def _sample_entropy(self, x, m=2, r_factor=0.2):
        """Vectorised sample entropy. Down-samples to 256 pts for speed."""
        if len(x) > 256:
            x = scipy_signal.resample(x, 256)
        N = len(x);  r = r_factor * float(np.std(x))
        if r == 0 or N <= m + 1:
            return 0.0

        def _phi(m_val):
            templates = np.array([x[i:i + m_val] for i in range(N - m_val)])
            count = 0
            for i in range(len(templates)):
                diffs = np.max(np.abs(templates[i] - templates), axis=1)
                count += np.sum(diffs < r) - 1
            return count

        A, B = _phi(m + 1), _phi(m)
        return float(-np.log(A / B)) if A > 0 and B > 0 else 0.0

    def extract_channel_features(self, x):
        bp = {name: self._band_power(x, rng) for name, rng in self.bands.items()}
        act, mob, cmp = self._hjorth(x)
        return np.array([
            bp['delta'], bp['theta'], bp['alpha'], bp['beta'], bp['gamma'],
            bp['alpha'] / bp['beta']  if bp['beta']  > 0 else 0.0,
            bp['theta'] / bp['alpha'] if bp['alpha'] > 0 else 0.0,
            self._spike_rate(x),
            float(np.var(x)),
            act, mob, cmp,
            self._sample_entropy(x),
        ], dtype=np.float32)

    def extract_features(self, epochs_data):
        """epochs_data: [N, T, C]  →  [N, 13*C]"""
        N, T, C = epochs_data.shape
        out = np.zeros((N, 13 * C), dtype=np.float32)
        for i in tqdm(range(N), desc="  Extracting features", leave=False):
            for c in range(C):
                out[i, c * 13:(c + 1) * 13] = self.extract_channel_features(epochs_data[i, :, c])
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================
# 3. Patient Processing Pipeline
# ============================================================
def process_patient(patient_id, config=None, data_dir=None, output_dir=None):
    """Full pipeline for one patient: load → preprocess → epoch → features → save."""
    if config is None:
        config = load_config()

    data_dir   = Path(data_dir   or config['paths']['data']['raw']['chb_mit'])
    output_dir = Path(output_dir or config['paths']['data']['processed']['eeg_features'])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\nProcessing patient: {patient_id}\n{'='*60}")

    preprocessor = EEGPreprocessor(config)
    extractor    = EEGFeatureExtractor(config)

    ann_path = data_dir / 'seizure_annotations.csv'
    if ann_path.exists():
        ann_df = pd.read_csv(ann_path)
        ann_df = ann_df.apply(lambda col: col.str.strip() if col.dtype == object else col)
        patient_ann = ann_df[ann_df['patient_id'] == patient_id]
    else:
        print(f"  Warning: seizure_annotations.csv not found"); patient_ann = pd.DataFrame()

    patient_dir = data_dir / patient_id
    if not patient_dir.exists():
        print(f"  Error: {patient_dir} not found"); return None

    edf_files = sorted(patient_dir.glob('*.edf'))
    if not edf_files:
        print(f"  Error: no EDF files in {patient_dir}"); return None

    print(f"  EDF files: {len(edf_files)} | Annotated seizures: {len(patient_ann)}")

    all_epochs, all_features, all_labels = [], [], []
    stats = dict(patient_id=patient_id, edf_files=len(edf_files),
                 total_epochs=0, interictal_epochs=0,
                 preictal_epochs=0, rejected_epochs=0, seizures=len(patient_ann))

    for edf_path in tqdm(edf_files, desc=f"  {patient_id}", leave=True):
        fname    = edf_path.name
        file_ann = patient_ann[patient_ann['file'] == fname]
        seizure_times = list(zip(file_ann['seizure_start_sec'].values,
                                 file_ann['seizure_end_sec'].values)) if len(file_ann) > 0 else []

        raw = preprocessor.load_edf(edf_path);   raw = preprocessor.preprocess(raw) if raw else None
        if raw is None: continue

        epochs_data, labels, valid_mask = preprocessor.create_epochs(raw, seizure_times)
        if epochs_data is None: continue

        valid_epochs = epochs_data[valid_mask];  valid_labels = labels[valid_mask]
        if len(valid_epochs) == 0: continue

        features = extractor.extract_features(valid_epochs)
        all_epochs.append(valid_epochs); all_features.append(features); all_labels.append(valid_labels)

        stats['total_epochs']      += len(valid_labels)
        stats['interictal_epochs'] += int(np.sum(valid_labels == 0))
        stats['preictal_epochs']   += int(np.sum(valid_labels == 1))
        stats['rejected_epochs']   += int(np.sum(~valid_mask))

    if not all_epochs:
        print(f"  No valid epochs for {patient_id}"); return stats

    seqs  = np.concatenate(all_epochs,   axis=0)
    feats = np.concatenate(all_features, axis=0)
    labs  = np.concatenate(all_labels,   axis=0)

    np.save(output_dir / f'{patient_id}_sequences.npy', seqs)
    np.save(output_dir / f'{patient_id}_features.npy',  feats)
    np.save(output_dir / f'{patient_id}_labels.npy',    labs)

    print(f"\n  ✅ {patient_id} done:")
    print(f"     Interictal : {stats['interictal_epochs']:,}")
    print(f"     Preictal   : {stats['preictal_epochs']:,}")
    print(f"     Rejected   : {stats['rejected_epochs']:,}")
    print(f"     Sequences  : {seqs.shape}  |  Features: {feats.shape}")
    print(f"     Saved to   : {output_dir}")
    return stats


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='EEG Preprocessing & Feature Extraction')
    parser.add_argument('--patient', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--data-dir',   type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    if args.all:
        patients = config['dataset']['chb_mit_patients']
    elif args.patient:
        patients = [args.patient]
    else:
        print("Specify --patient <id> or --all"); return

    print("=" * 60)
    print("EEG PREPROCESSING & FEATURE EXTRACTION")
    print("=" * 60)
    print(f"Patients: {', '.join(patients)}\n")

    all_stats = []
    for pid in patients:
        s = process_patient(pid, config, data_dir=args.data_dir, output_dir=args.output_dir)
        if s:
            all_stats.append(s)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    if all_stats:
        df = pd.DataFrame(all_stats)
        print(df.to_string(index=False))
        print(f"\n  Total patients : {len(all_stats)}")
        print(f"  Total epochs   : {df['total_epochs'].sum():,}")
        print(f"  Total preictal : {df['preictal_epochs'].sum():,}")
        ratio = df['interictal_epochs'].sum() / max(df['preictal_epochs'].sum(), 1)
        print(f"  Class ratio (interictal:preictal) : {ratio:.1f}:1")


if __name__ == '__main__':
    main()

