#!/usr/bin/env python3
"""
Data Loader
=============
Loads preprocessed EEG features and sequences, handles train/val/test splits
with patient-level separation to prevent data leakage.

Usage:
    from src.data_pipeline.data_loader import EEGDataLoader
    
    loader = EEGDataLoader(config)
    train_data, val_data, test_data = loader.get_patient_splits()
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import torch
from torch.utils.data import Dataset, DataLoader


def load_config(config_path="configs/config.yaml"):
    """Load configuration from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class EEGDataset(Dataset):
    """PyTorch Dataset for EEG sequences and labels."""
    
    def __init__(self, sequences, labels, features=None, 
                 augment=False, noise_sigma=0.01, channel_dropout_prob=0.1):
        """
        Args:
            sequences: np.array [N, T, C] - raw EEG time series
            labels: np.array [N] - binary labels
            features: np.array [N, F] - extracted features (optional)
            augment: bool - whether to apply data augmentation
            noise_sigma: float - Gaussian noise std for augmentation
            channel_dropout_prob: float - probability of dropping a channel
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
        self.features = torch.FloatTensor(features) if features is not None else None
        self.augment = augment
        self.noise_sigma = noise_sigma
        self.channel_dropout_prob = channel_dropout_prob
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        if self.augment:
            # Gaussian noise
            noise = torch.randn_like(seq) * self.noise_sigma
            seq = seq + noise
            
            # Random channel dropout
            if torch.rand(1).item() < self.channel_dropout_prob:
                ch_idx = torch.randint(0, seq.shape[1], (1,)).item()
                seq[:, ch_idx] = 0.0
        
        result = {'sequence': seq, 'label': label}
        
        if self.features is not None:
            result['features'] = self.features[idx]
        
        return result


class EEGDataLoader:
    """
    Manages loading, splitting, and batching of EEG data.
    Implements patient-level leave-one-out cross-validation.
    """
    
    def __init__(self, config=None, features_dir=None):
        if config is None:
            config = load_config()
        
        self.config = config
        
        if features_dir is None:
            self.features_dir = Path(config['paths']['data']['processed']['eeg_features'])
        else:
            self.features_dir = Path(features_dir)
        
        self.patients = config['dataset']['chb_mit_patients']
        self.batch_size = config['lstm']['batch_size']
    
    def load_patient_data(self, patient_id):
        """Load all processed data for a single patient."""
        base = self.features_dir / patient_id
        
        data = {}
        
        seq_path = self.features_dir / f'{patient_id}_sequences.npy'
        feat_path = self.features_dir / f'{patient_id}_features.npy'
        label_path = self.features_dir / f'{patient_id}_labels.npy'
        
        if seq_path.exists():
            data['sequences'] = np.load(seq_path)
        if feat_path.exists():
            data['features'] = np.load(feat_path)
        if label_path.exists():
            data['labels'] = np.load(label_path)
        
        return data if data else None
    
    def get_all_data(self) -> Dict[str, dict]:
        """Load data for all patients."""
        all_data = {}
        for patient_id in self.patients:
            data = self.load_patient_data(patient_id)
            if data is not None:
                all_data[patient_id] = data
        return all_data
    
    def get_leave_one_out_splits(self, test_patient_id: str) -> Tuple:
        """
        Create patient-level leave-one-out split.
        
        Args:
            test_patient_id: Patient to hold out for testing
            
        Returns:
            train_data, val_data, test_data: dicts with 'sequences', 'features', 'labels'
        """
        all_data = self.get_all_data()
        
        if test_patient_id not in all_data:
            raise ValueError(f"Patient {test_patient_id} not found in data")
        
        # Test set: held-out patient
        test_data = all_data[test_patient_id]
        
        # Remaining patients for train/val
        remaining = {pid: data for pid, data in all_data.items() 
                    if pid != test_patient_id}
        
        # Use one remaining patient as validation, rest as training
        remaining_ids = sorted(remaining.keys())
        val_patient_id = remaining_ids[-1]  # Last remaining as validation
        
        val_data = remaining[val_patient_id]
        
        # Concatenate remaining for training
        train_sequences = []
        train_features = []
        train_labels = []
        
        for pid in remaining_ids[:-1]:
            data = remaining[pid]
            if 'sequences' in data:
                train_sequences.append(data['sequences'])
            if 'features' in data:
                train_features.append(data['features'])
            if 'labels' in data:
                train_labels.append(data['labels'])
        
        train_data = {}
        if train_sequences:
            train_data['sequences'] = np.concatenate(train_sequences, axis=0)
        if train_features:
            train_data['features'] = np.concatenate(train_features, axis=0)
        if train_labels:
            train_data['labels'] = np.concatenate(train_labels, axis=0)
        
        return train_data, val_data, test_data
    
    def create_dataloaders(self, test_patient_id: str, 
                          augment_train: bool = True) -> Tuple:
        """
        Create PyTorch DataLoaders with patient-level splits.
        
        Args:
            test_patient_id: Patient to hold out
            augment_train: Whether to augment training data
            
        Returns:
            train_loader, val_loader, test_loader: PyTorch DataLoaders
        """
        train_data, val_data, test_data = self.get_leave_one_out_splits(test_patient_id)
        
        aug_config = self.config['lstm'].get('augmentation', {})
        noise_sigma = aug_config.get('gaussian_noise_sigma', 0.01)
        channel_dropout = aug_config.get('channel_dropout_prob', 0.1)
        
        train_dataset = EEGDataset(
            train_data['sequences'], train_data['labels'],
            features=train_data.get('features'),
            augment=augment_train,
            noise_sigma=noise_sigma,
            channel_dropout_prob=channel_dropout
        )
        
        val_dataset = EEGDataset(
            val_data['sequences'], val_data['labels'],
            features=val_data.get('features')
        )
        
        test_dataset = EEGDataset(
            test_data['sequences'], test_data['labels'],
            features=test_data.get('features')
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=0, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=0, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=0, pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def get_class_weights(self, train_data: dict) -> torch.Tensor:
        """
        Compute class weights for imbalanced dataset.
        
        Returns:
            pos_weight: Tensor with positive class weight for BCEWithLogitsLoss
        """
        labels = train_data['labels']
        n_pos = np.sum(labels == 1)
        n_neg = np.sum(labels == 0)
        
        if n_pos == 0:
            return torch.tensor([1.0])
        
        pos_weight = n_neg / n_pos
        return torch.tensor([pos_weight])
    
    def get_data_stats(self) -> pd.DataFrame:
        """Get statistics for all loaded patients."""
        stats = []
        for patient_id in self.patients:
            data = self.load_patient_data(patient_id)
            if data and 'labels' in data:
                labels = data['labels']
                stats.append({
                    'patient_id': patient_id,
                    'total_epochs': len(labels),
                    'interictal': np.sum(labels == 0),
                    'preictal': np.sum(labels == 1),
                    'seq_shape': str(data['sequences'].shape) if 'sequences' in data else 'N/A',
                    'feat_shape': str(data['features'].shape) if 'features' in data else 'N/A'
                })
        
        return pd.DataFrame(stats)


if __name__ == '__main__':
    # Quick test
    loader = EEGDataLoader()
    stats = loader.get_data_stats()
    if not stats.empty:
        print("Data Statistics:")
        print(stats.to_string(index=False))
    else:
        print("No processed data found. Run eeg_preprocessing.py first.")
