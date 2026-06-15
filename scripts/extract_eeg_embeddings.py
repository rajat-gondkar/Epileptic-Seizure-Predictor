#!/usr/bin/env python3
"""
Standalone EEG Embedding Extraction
=====================================
Extracts 512-dim embeddings from the trained BiLSTM for all EEG windows.
Caches results to avoid re-extraction.

Usage:
    python scripts/extract_eeg_embeddings.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'seizure_prediction'))

from libModelLSTM import clsLSTM
from libCHBMITDataset import CHBMITDataset


EEG_MODEL = 'seizure_prediction/SavedModels/EEGLSTM_CHB-MIT_all_patients_train_Epoch-15_TLoss-0.0501_VLoss-0.7880_20260605-152850.net'
CSV_TRAIN = 'seizure_prediction/DataCSVs/CHB-MIT/all_patients_train.csv'
CSV_TEST = 'seizure_prediction/DataCSVs/CHB-MIT/all_patients_test.csv'
OUTPUT_DIR = Path('models/fusion')
CACHE_FILE = OUTPUT_DIR / 'eeg_embeddings.npz'

TARGET_SEQ_LEN = 7680  # 30 seconds at 256 Hz (matches -du 30 -rf 256 training params)


def collate_fn(batch):
    """Pad/truncate all windows to fixed length, select 19 channels, and stack."""
    eeg_windows, labels = zip(*batch)
    
    # The 19 common channels the model was trained on
    COMMON_19 = [
        'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
        'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
        'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
        'FZ-CZ', 'CZ-PZ', 'P7-T7'
    ]
    
    padded = []
    for w in eeg_windows:
        w = torch.as_tensor(w, dtype=torch.float32)
        
        # If more than 19 channels, select common 19
        if w.shape[1] > 19:
            # Try to find common channels by name if available
            # Otherwise just take first 19
            w = w[:, :19]
        
        # Pad or truncate to fixed length
        if w.shape[0] > TARGET_SEQ_LEN:
            w = w[:TARGET_SEQ_LEN]
        elif w.shape[0] < TARGET_SEQ_LEN:
            pad = torch.zeros(TARGET_SEQ_LEN - w.shape[0], w.shape[1])
            w = torch.cat([w, pad], dim=0)
        padded.append(w)
    
    eeg_batch = torch.stack(padded, dim=0)
    label_batch = torch.tensor(labels, dtype=torch.float32)
    
    return eeg_batch, label_batch


def load_eeg_model():
    """Load the trained BiLSTM model."""
    print(f"Loading EEG model from {EEG_MODEL}")
    checkpoint = torch.load(EEG_MODEL, map_location='cpu', weights_only=False)
    
    model = clsLSTM(
        checkpoint['intFeaturesDim'],
        checkpoint['intHiddenDim'],
        checkpoint['intNumLayers'],
        checkpoint['intOutputSize'],
        checkpoint['fltDropProb'],
    )
    model.load_state_dict(checkpoint['dctStateDict'])
    model.eval()
    
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  Features: {checkpoint['intFeaturesDim']}, Hidden: {checkpoint['intHiddenDim']}, "
          f"Layers: {checkpoint['intNumLayers']}, Output: {checkpoint['intOutputSize']}")
    return model


def extract_embeddings(model, dataset, batch_size=16):
    """Extract embeddings from a dataset."""
    all_embeddings = []
    all_labels = []
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (eeg_windows, labels) in enumerate(dataloader):
            hidden = model.initHidden(len(eeg_windows), argTrainOnGPU=False)
            embedding, _ = model.get_embedding(eeg_windows, hidden)
            
            all_embeddings.append(embedding.numpy())
            all_labels.append(labels.numpy())
            
            if (batch_idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                pct = (batch_idx + 1) / len(dataloader) * 100
                print(f"    {pct:.1f}% ({(batch_idx+1)*batch_size}/{len(dataset)}) "
                      f"- {elapsed:.0f}s elapsed")
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    return embeddings, labels


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check cache
    if CACHE_FILE.exists():
        print(f"Cache found: {CACHE_FILE}")
        cache = np.load(CACHE_FILE)
        print(f"  Embeddings: {cache['embeddings'].shape}")
        print(f"  Labels: {cache['labels'].shape} ({cache['labels'].sum()} positive)")
        print("  Skipping extraction.")
        return
    
    # Load model
    model = load_eeg_model()
    
    # Load datasets — must match training params: -du 30 -rf 256 -smod 1 -smin -1 -smax 1
    dataset_kwargs = dict(
        resampling_freq=256,
        subseq_duration=30,
        scaling_params=(1, (-1.0, 1.0)),
        bandpass_freqs=(0.5, 45.0),
        zscore_normalize=True,
        argInfo=True,
    )

    print(f"\nLoading train dataset from {CSV_TRAIN}")
    train_dataset = CHBMITDataset(CSV_TRAIN, **dataset_kwargs)
    print(f"  Train windows: {len(train_dataset)}")

    print(f"\nLoading test dataset from {CSV_TEST}")
    test_dataset = CHBMITDataset(CSV_TEST, **dataset_kwargs)
    print(f"  Test windows: {len(test_dataset)}")
    
    # Extract train embeddings
    print(f"\nExtracting train embeddings...")
    start = time.time()
    train_emb, train_labels = extract_embeddings(model, train_dataset)
    print(f"  Done in {time.time()-start:.0f}s")
    
    # Extract test embeddings
    print(f"\nExtracting test embeddings...")
    start = time.time()
    test_emb, test_labels = extract_embeddings(model, test_dataset)
    print(f"  Done in {time.time()-start:.0f}s")
    
    # Combine
    all_embeddings = np.concatenate([train_emb, test_emb], axis=0)
    all_labels = np.concatenate([train_labels, test_labels], axis=0)
    
    # Save
    np.savez(CACHE_FILE, embeddings=all_embeddings, labels=all_labels)
    
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total embeddings: {all_embeddings.shape}")
    print(f"  Total labels: {all_labels.shape}")
    print(f"  Positive rate: {all_labels.mean()*100:.1f}%")
    print(f"  Saved to: {CACHE_FILE}")


if __name__ == '__main__':
    main()
