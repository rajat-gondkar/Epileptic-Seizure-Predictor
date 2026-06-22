#!/usr/bin/env python3
"""
EEG seizure-prediction inference engine.

Loads the trained multi-patient BiLSTM (all_patients model, 19 channels,
3840 timesteps @ 128 Hz, 3 classes) and runs genuine sliding-window inference
on an uploaded EDF file, using the exact preprocessing from training:

    19 common channels -> 0.5-45 Hz Butterworth bandpass -> resample to 128 Hz
    -> per-channel z-score -> (time, channels) tensor -> BiLSTM + attention.

Class map: 0 = interictal, 1 = preictal, 2 = ictal.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal as sp_signal
import pyedflib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SEIZURE_DIR = PROJECT_ROOT / "seizure_prediction"
sys.path.insert(0, str(SEIZURE_DIR))

from libModelLSTM import clsLSTM  # noqa: E402

# ------------------------------------------------------------------
# Configuration (matches the documented training pipeline)
# ------------------------------------------------------------------
MODEL_PATH = SEIZURE_DIR / "SavedModels" / (
    "EEGLSTM_CHB-MIT_all_patients_train_Epoch-15_TLoss-0.0501_VLoss-0.7880_20260605-152850.net"
)

# The 19 common bipolar channels (exact order used during all-patients training)
COMMON_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ", "T8-P8",
]

WINDOW_SEC = 30
TARGET_SF = 128
BANDPASS = (0.5, 45.0)
CLASS_NAMES = ["interictal", "preictal", "ictal"]


class SeizureInferenceEngine:
    def __init__(self, model_path: Path = MODEL_PATH, device: str = "cpu"):
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        self.model = None
        self.meta = {}
        self._load()

    def _load(self):
        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model = clsLSTM(
            ckpt["intFeaturesDim"], ckpt["intHiddenDim"], ckpt["intNumLayers"],
            ckpt["intOutputSize"], argDropProb=ckpt["fltDropProb"],
        )
        self.model.load_state_dict(ckpt["dctStateDict"])
        self.model.to(self.device)
        self.model.eval()
        self.meta = {
            "model_file": self.model_path.name,
            "features_dim": ckpt["intFeaturesDim"],
            "hidden_dim": ckpt["intHiddenDim"],
            "num_layers": ckpt["intNumLayers"],
            "output_size": ckpt["intOutputSize"],
            "seq_len": ckpt.get("intSeqLen"),
            "params": sum(p.numel() for p in self.model.parameters()),
            "channels": COMMON_CHANNELS,
            "window_sec": WINDOW_SEC,
            "target_sf": TARGET_SF,
        }

    # --------------------------------------------------------------
    def _read_channels(self, edf_path: str):
        """Read the 19 common channels from an EDF (zeros for any missing)."""
        with pyedflib.EdfReader(edf_path) as fh:
            labels = [l.strip() for l in fh.getSignalLabels()]
            sf = float(fh.getSampleFrequency(0))
            n_pts = fh.getNSamples()[0]
            label_to_idx = {}
            for i, lab in enumerate(labels):
                label_to_idx.setdefault(lab, i)

            data = np.zeros((len(COMMON_CHANNELS), n_pts), dtype=np.float32)
            present = []
            for out_idx, ch in enumerate(COMMON_CHANNELS):
                if ch in label_to_idx:
                    raw = fh.readSignal(label_to_idx[ch])
                    data[out_idx, : len(raw)] = raw[:n_pts].astype(np.float32)
                    present.append(ch)
        return data, sf, n_pts, present

    def _preprocess_window(self, win, sf):
        """Bandpass -> resample to 3840 -> per-channel z-score. win: (ch, pts)."""
        nyq = sf / 2.0
        if BANDPASS[1] < nyq:
            sos = sp_signal.butter(4, [BANDPASS[0] / nyq, BANDPASS[1] / nyq],
                                   btype="band", output="sos")
            win = sp_signal.sosfilt(sos, win, axis=1)
        target_pts = WINDOW_SEC * TARGET_SF
        win = sp_signal.resample(win, target_pts, axis=1)
        for ch in range(win.shape[0]):
            std = win[ch].std()
            if std > 1e-8:
                win[ch] = (win[ch] - win[ch].mean()) / std
        return win.astype(np.float32)

    # --------------------------------------------------------------
    def predict_edf(self, edf_path: str, max_windows: int = 240, batch_size: int = 16,
                    preview_points: int = 70):
        """
        Run sliding-window inference over an EDF.

        Returns a dict with per-window predictions, a compact real waveform
        preview per window, and a summary.
        """
        data, sf, n_pts, present = self._read_channels(edf_path)
        win_pts = int(round(WINDOW_SEC * sf))
        n_windows = n_pts // win_pts
        if n_windows == 0:
            raise ValueError("Recording is shorter than one 30-second window.")
        n_windows = min(n_windows, max_windows)

        # Build all preprocessed windows
        proc, previews = [], []
        for w in range(n_windows):
            seg = data[:, w * win_pts:(w + 1) * win_pts]
            pw = self._preprocess_window(seg, sf)
            proc.append(pw.T)  # (time, channels)
            # compact real preview: channel-mean, downsampled
            avg = seg.mean(axis=0)
            idx = np.linspace(0, len(avg) - 1, preview_points).astype(int)
            prev = avg[idx]
            rng = np.abs(prev).max()
            previews.append((prev / rng if rng > 1e-6 else prev).round(3).tolist())

        windows = []
        with torch.no_grad():
            for start in range(0, len(proc), batch_size):
                chunk = proc[start:start + batch_size]
                x = torch.from_numpy(np.stack(chunk)).float().to(self.device)
                hidden = self.model.initHidden(x.shape[0], argTrainOnGPU=False)
                logits, _ = self.model(x, hidden)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                for j, p in enumerate(probs):
                    wi = start + j
                    pred = int(np.argmax(p))
                    windows.append({
                        "index": wi,
                        "start_sec": round(wi * WINDOW_SEC, 1),
                        "end_sec": round((wi + 1) * WINDOW_SEC, 1),
                        "pred_class": pred,
                        "pred_name": CLASS_NAMES[pred],
                        "p_interictal": round(float(p[0]), 4),
                        "p_preictal": round(float(p[1]), 4),
                        "p_ictal": round(float(p[2]), 4),
                        "preview": previews[wi],
                    })

        counts = {c: sum(1 for w in windows if w["pred_class"] == i)
                  for i, c in enumerate(CLASS_NAMES)}
        return {
            "filename": os.path.basename(edf_path),
            "sampling_freq": sf,
            "duration_sec": round(n_pts / sf, 1),
            "channels_present": len(present),
            "channels_expected": len(COMMON_CHANNELS),
            "n_windows": len(windows),
            "windows": windows,
            "summary": {
                "counts": counts,
                "any_preictal": counts["preictal"] > 0,
                "any_ictal": counts["ictal"] > 0,
                "max_preictal_prob": round(max((w["p_preictal"] for w in windows), default=0), 4),
                "max_ictal_prob": round(max((w["p_ictal"] for w in windows), default=0), 4),
            },
        }
