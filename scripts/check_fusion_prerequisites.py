#!/usr/bin/env python3
"""
Fusion Training Sanity Check
==============================
Validates that all required models exist and can run inference
before starting the full fusion training pipeline.

Run this first to catch any issues early.

Usage:
    python scripts/check_fusion_prerequisites.py
"""

import sys
import os
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'


def print_status(name, passed, detail=""):
    if passed:
        print(f"  {GREEN}✓{RESET} {name}")
        if detail:
            print(f"    {detail}")
    else:
        print(f"  {RED}✗{RESET} {name}")
        if detail:
            print(f"    {RED}{detail}{RESET}")


def check_file_exists(path, name):
    exists = Path(path).exists()
    print_status(name, exists, f"Path: {path}")
    return exists


def check_eeg_model(model_path):
    """Load EEG model and run a dummy forward pass."""
    print(f"\n{BOLD}--- EEG Model Check ---{RESET}")
    
    if not check_file_exists(model_path, "EEG model file"):
        return False
    
    try:
        sys.path.insert(0, str(PROJECT_ROOT / 'seizure_prediction'))
        from libModelLSTM import clsLSTM
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Extract parameters
        intFeaturesDim = checkpoint['intFeaturesDim']
        intHiddenDim = checkpoint['intHiddenDim']
        intNumLayers = checkpoint['intNumLayers']
        intOutputSize = checkpoint['intOutputSize']
        fltDropProb = checkpoint['fltDropProb']
        
        print_status("Checkpoint loaded", True,
                     f"Features={intFeaturesDim}, Hidden={intHiddenDim}, "
                     f"Layers={intNumLayers}, Output={intOutputSize}")
        
        # Create model
        model = clsLSTM(intFeaturesDim, intHiddenDim, intNumLayers,
                       intOutputSize, fltDropProb)
        model.load_state_dict(checkpoint['dctStateDict'], weights_only=False)
        model.eval()
        
        param_count = sum(p.numel() for p in model.parameters())
        print_status("Model created", True, f"Parameters: {param_count:,}")
        
        # Run dummy forward pass
        batch_size = 2
        time_steps = 10  # Reduced for quick test
        dummy_input = torch.randn(batch_size, time_steps, intFeaturesDim)
        hidden = model.initHidden(batch_size, argTrainOnGPU=False)
        
        with torch.no_grad():
            embedding, risk_score = model.get_embedding(dummy_input, hidden)
        
        print_status("Forward pass (get_embedding)", True,
                     f"Embedding shape: {embedding.shape}, Risk shape: {risk_score.shape}")
        
        # Verify embedding dimensions
        expected_lstm_dim = intHiddenDim * 2  # Bidirectional
        actual_dim = embedding.shape[1]
        dims_match = (actual_dim == expected_lstm_dim)
        print_status(f"Embedding dimension check", dims_match,
                     f"Expected: {expected_lstm_dim}, Got: {actual_dim}")
        
        return dims_match
        
    except Exception as e:
        print_status("EEG model check", False, f"Error: {str(e)}")
        return False


def check_genetic_model(model_path):
    """Load XGBoost model and run a dummy prediction."""
    print(f"\n{BOLD}--- Genetic Model Check ---{RESET}")
    
    if not check_file_exists(model_path, "XGBoost model file"):
        return False
    
    try:
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        
        print_status("XGBoost model loaded", True)
        
        # Run dummy prediction (22 features)
        dummy_features = np.random.randn(2, 22).astype(np.float32)
        probs = model.predict_proba(dummy_features)
        
        print_status("Dummy prediction", True,
                     f"Output shape: {probs.shape}, "
                     f"Sample probs: {probs[0].round(3)}")
        
        return True
        
    except Exception as e:
        print_status("Genetic model check", False, f"Error: {str(e)}")
        return False


def check_genetic_data(data_path):
    """Check genetic training data."""
    print(f"\n{BOLD}--- Genetic Data Check ---{RESET}")
    
    if not check_file_exists(data_path, "Genetic training data"):
        return False
    
    try:
        import pandas as pd
        df = pd.read_csv(data_path)
        
        n_patients = len(df)
        n_cols = len(df.columns)
        has_label = 'has_seizure' in df.columns
        positive_rate = df['has_seizure'].mean() if has_label else 0
        
        print_status("Data loaded", True,
                     f"Patients: {n_patients}, Columns: {n_cols}")
        print_status("Label column exists", has_label,
                     f"Positive rate: {positive_rate*100:.1f}%")
        
        return True
        
    except Exception as e:
        print_status("Genetic data check", False, f"Error: {str(e)}")
        return False


def check_fusion_layer():
    """Check fusion layer can be instantiated."""
    print(f"\n{BOLD}--- Fusion Layer Check ---{RESET}")
    
    try:
        sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'training'))
        from fusion import AttentionGateFusion, FusionDataset
        
        # Create model with 512-dim EEG embedding
        model = AttentionGateFusion(
            eeg_embedding_dim=512,
            genetic_dim=1,
            hidden_dim=128,
            dropout=0.3,
        )
        
        param_count = sum(p.numel() for p in model.parameters())
        print_status("Fusion model created", True, f"Parameters: {param_count:,}")
        
        # Run dummy forward pass
        batch_size = 4
        dummy_eeg = torch.randn(batch_size, 512)
        dummy_genetic = torch.rand(batch_size, 1)
        
        with torch.no_grad():
            risk_score, alpha = model(dummy_eeg, dummy_genetic)
        
        print_status("Forward pass", True,
                     f"Risk shape: {risk_score.shape}, Alpha shape: {alpha.shape}")
        
        # Check alpha range
        alpha_range = (alpha.min().item(), alpha.max().item())
        print_status("Alpha range check", True,
                     f"Range: [{alpha_range[0]:.3f}, {alpha_range[1]:.3f}]")
        
        # Check dataset
        dummy_labels = np.random.randint(0, 2, batch_size).astype(np.float32)
        dataset = FusionDataset(
            dummy_eeg.numpy(), dummy_genetic.numpy(), dummy_labels
        )
        print_status("FusionDataset", True, f"Length: {len(dataset)}")
        
        return True
        
    except Exception as e:
        print_status("Fusion layer check", False, f"Error: {str(e)}")
        return False


def check_output_directories():
    """Check output directories exist or can be created."""
    print(f"\n{BOLD}--- Output Directories Check ---{RESET}")
    
    dirs = [
        PROJECT_ROOT / 'models' / 'fusion',
        PROJECT_ROOT / 'models' / 'fusion' / 'plots',
        PROJECT_ROOT / 'data' / 'processed' / 'fusion',
    ]
    
    all_ok = True
    for d in dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            print_status(str(d.relative_to(PROJECT_ROOT)), True)
        except Exception as e:
            print_status(str(d.relative_to(PROJECT_ROOT)), False, str(e))
            all_ok = False
    
    return all_ok


def main():
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Fusion Training Prerequisites Check{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    
    # Define paths
    eeg_model = "seizure_prediction/SavedModels/EEGLSTM_CHB-MIT_all_patients_train_Epoch-15_TLoss-0.0501_VLoss-0.7880_20260605-152850.net"
    xgb_model = "models/xgboost_genetic/xgboost_genetic_model.pkl"
    genetic_data = "data/processed/genetic_vectors/genetic_training_cohort.csv"
    
    # Run all checks
    results = {
        'EEG Model': check_eeg_model(eeg_model),
        'Genetic Model': check_genetic_model(xgb_model),
        'Genetic Data': check_genetic_data(genetic_data),
        'Fusion Layer': check_fusion_layer(),
        'Output Dirs': check_output_directories(),
    }
    
    # Summary
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Summary{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    
    all_passed = True
    for name, passed in results.items():
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    
    if all_passed:
        print(f"{GREEN}{BOLD}All checks passed! Ready to run fusion training.{RESET}")
        print(f"\nNext step:")
        print(f"  ./scripts/run_fusion_training.sh")
        return 0
    else:
        print(f"{RED}{BOLD}Some checks failed. Fix the issues above before training.{RESET}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
