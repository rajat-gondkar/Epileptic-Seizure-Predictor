#!/usr/bin/env python3
"""
Attention-Gated Fusion Layer
==============================
Combines EEG branch predictions with Genetic branch predictions
through a learned attention mechanism.

Architecture:
    EEG Embedding (64-dim) ──┐
                             ├──► Attention Gate ──► Weighted Sum ──► Risk Score
    Genetic Score (1-dim) ───┘

The attention gate learns a per-patient weight α ∈ [0,1]:
    α = σ(W_e · h_eeg + W_g · h_genetic + b)
    P_final = α · P_eeg + (1 − α) · P_genetic

Where:
    - h_eeg: EEG embedding from BiLSTM (64-dim)
    - h_genetic: Genetic risk score from XGBoost (1-dim, expanded to 64-dim)
    - α: Attention weight (how much to trust EEG vs Genetic)
    - P_final: Final fused seizure risk score

Usage:
    from src.training.fusion import AttentionGateFusion, FusionTrainer
    
    model = AttentionGateFusion(eeg_embedding_dim=64, genetic_dim=1)
    trainer = FusionTrainer(model, eeg_model, genetic_model)
    trainer.train(train_loader, val_loader, epochs=30)
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import xgboost as xgb

# -- Project root --
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))


# ============================================================
# Fusion Layer Model
# ============================================================
class AttentionGateFusion(nn.Module):
    """
    Attention-gated late fusion layer.
    
    Combines EEG embedding with genetic risk score through a learned
    attention mechanism that produces per-patient weighting.
    
    Args:
        eeg_embedding_dim: Dimension of EEG embedding (default: 64)
        genetic_dim: Dimension of genetic input (default: 1)
        hidden_dim: Hidden dimension for attention computation (default: 128)
        dropout: Dropout rate (default: 0.3)
    """
    
    def __init__(
        self,
        eeg_embedding_dim: int = 64,
        genetic_dim: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.eeg_embedding_dim = eeg_embedding_dim
        self.genetic_dim = genetic_dim
        self.hidden_dim = hidden_dim
        
        # EEG projection (64 → hidden)
        self.eeg_projection = nn.Sequential(
            nn.Linear(eeg_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        
        # Genetic projection (1 → hidden)
        # First expand genetic score to match hidden dim
        self.genetic_projection = nn.Sequential(
            nn.Linear(genetic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        
        # Attention gate
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        # Final risk score computation
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        eeg_embedding: torch.Tensor,
        genetic_score: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the fusion layer.
        
        Args:
            eeg_embedding: EEG embedding tensor (batch_size, eeg_embedding_dim)
            genetic_score: Genetic risk score tensor (batch_size, 1) or (batch_size,)
            return_attention: Whether to return attention weights
            
        Returns:
            risk_score: Fused seizure risk score (batch_size, 1) ∈ [0, 1]
            attention_weight: Attention weight α (batch_size, 1) ∈ [0, 1]
        """
        # Ensure genetic_score is 2D
        if genetic_score.dim() == 1:
            genetic_score = genetic_score.unsqueeze(1)
        
        # Project both inputs to same space
        eeg_proj = self.eeg_projection(eeg_embedding)    # (batch, hidden)
        gen_proj = self.genetic_projection(genetic_score)  # (batch, hidden)
        
        # Compute attention weight
        # Concatenate projections for attention
        combined = torch.cat([eeg_proj, gen_proj], dim=1)  # (batch, hidden*2)
        alpha = self.attention(combined)  # (batch, 1)
        
        # Weighted combination of projections
        weighted_eeg = alpha * eeg_proj
        weighted_gen = (1 - alpha) * gen_proj
        fused = weighted_eeg + weighted_gen  # (batch, hidden)
        
        # Compute final risk score
        # Concatenate fused representation with genetic score for final prediction
        risk_input = torch.cat([fused, genetic_score], dim=1)  # (batch, hidden+1)
        risk_score = self.risk_head(risk_input)  # (batch, 1)
        
        if return_attention:
            return risk_score, alpha
        return risk_score, alpha
    
    def get_attention_weights(
        self,
        eeg_embedding: torch.Tensor,
        genetic_score: torch.Tensor,
    ) -> torch.Tensor:
        """Get attention weights without computing final risk score."""
        if genetic_score.dim() == 1:
            genetic_score = genetic_score.unsqueeze(1)
        
        eeg_proj = self.eeg_projection(eeg_embedding)
        gen_proj = self.genetic_projection(genetic_score)
        
        combined = torch.cat([eeg_proj, gen_proj], dim=1)
        alpha = self.attention(combined)
        
        return alpha


# ============================================================
# EEG Branch Wrapper
# ============================================================
class EEGBranchWrapper:
    """
    Wrapper around the trained BiLSTM model to extract embeddings.
    
    Loads the trained model and provides a method to get embeddings
    without the classification head.
    """
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Args:
            model_path: Path to the saved BiLSTM model (.net file)
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.model = None
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        """Load the trained BiLSTM model."""
        # Import the model class
        sys.path.insert(0, str(SCRIPT_DIR / 'seizure_prediction'))
        from libModelLSTM import clsLSTM
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract model parameters
        intFeaturesDim = checkpoint['intFeaturesDim']
        intHiddenDim = checkpoint['intHiddenDim']
        intNumLayers = checkpoint['intNumLayers']
        intOutputSize = checkpoint['intOutputSize']
        fltDropProb = checkpoint['fltDropProb']
        
        # Create model
        self.model = clsLSTM(
            intFeaturesDim, intHiddenDim, intNumLayers,
            intOutputSize, fltDropProb
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['dctStateDict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded EEG model from {self.model_path}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def get_embedding(self, eeg_window: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get EEG embedding and risk score from a window.
        
        Args:
            eeg_window: EEG data tensor (batch_size, time_steps, channels)
            
        Returns:
            embedding: EEG embedding (batch_size, 512) - attention-weighted LSTM output
            risk_score: EEG risk score (batch_size, 1) - preictal probability
        """
        with torch.no_grad():
            eeg_window = eeg_window.to(self.device)
            
            # Initialize hidden state
            batch_size = eeg_window.shape[0]
            hidden = self.model.initHidden(batch_size, 
                                          argTrainOnGPU=(self.device.type == 'cuda'))
            
            # Use the new get_embedding method from libModelLSTM
            embedding, risk_score = self.model.get_embedding(eeg_window, hidden)
            
            # Convert risk score to probabilities
            risk_score = F.softmax(risk_score, dim=1)
            
            # For binary fusion, we want the preictal probability (class 1)
            # or combine interictal vs (preictal + ictal)
            preictal_prob = risk_score[:, 1] if risk_score.shape[1] > 1 else risk_score[:, 0]
            
            return embedding, preictal_prob.unsqueeze(1)


# ============================================================
# Genetic Branch Wrapper
# ============================================================
class GeneticBranchWrapper:
    """
    Wrapper around the trained XGBoost model to extract risk scores.
    
    Loads the trained XGBoost model and provides continuous risk scores.
    """
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: Path to the saved XGBoost model (.pkl)
        """
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        print(f"Loaded Genetic XGBoost model from {model_path}")
    
    def get_risk_score(self, genetic_features: np.ndarray) -> np.ndarray:
        """
        Get genetic risk score from features.
        
        Args:
            genetic_features: Genetic feature matrix (n_patients, n_features)
            
        Returns:
            risk_scores: Genetic risk scores (n_patients, 1) ∈ [0, 1]
        """
        # Get probability of seizure class
        risk_scores = self.model.predict_proba(genetic_features)[:, 1]
        return risk_scores.reshape(-1, 1)


# ============================================================
# Fusion Dataset
# ============================================================
class FusionDataset(Dataset):
    """
    Dataset for training the fusion layer.
    
    Provides EEG embeddings and genetic scores paired with labels.
    """
    
    def __init__(
        self,
        eeg_embeddings: np.ndarray,
        genetic_scores: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Args:
            eeg_embeddings: EEG embeddings (n_samples, 64)
            genetic_scores: Genetic risk scores (n_samples, 1)
            labels: Binary labels (n_samples,)
        """
        self.eeg_embeddings = torch.FloatTensor(eeg_embeddings)
        self.genetic_scores = torch.FloatTensor(genetic_scores)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            self.eeg_embeddings[idx],
            self.genetic_scores[idx],
            self.labels[idx],
        )


# ============================================================
# Fusion Trainer
# ============================================================
class FusionTrainer:
    """
    Trainer for the fusion layer.
    
    Freezes both EEG and Genetic branches, trains only the attention gate.
    """
    
    def __init__(
        self,
        model: AttentionGateFusion,
        device: str = 'cpu',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        l1_alpha: float = 0.01,
    ):
        """
        Args:
            model: AttentionGateFusion model
            device: Device for training
            learning_rate: Learning rate
            weight_decay: L2 regularization
            l1_alpha: L1 regularization on attention weights
        """
        self.model = model.to(device)
        self.device = torch.device(device)
        self.l1_alpha = l1_alpha
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        self.criterion = nn.BCELoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
    
    def _compute_l1_loss(self) -> torch.Tensor:
        """Compute L1 regularization on attention weights."""
        l1_loss = torch.tensor(0.0, device=self.device)
        for name, param in self.model.named_parameters():
            if 'attention' in name:
                l1_loss += torch.sum(torch.abs(param))
        return self.l1_alpha * l1_loss
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for eeg_emb, gen_score, labels in dataloader:
            eeg_emb = eeg_emb.to(self.device)
            gen_score = gen_score.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            risk_score, alpha = self.model(eeg_emb, gen_score)
            loss = self.criterion(risk_score, labels)
            
            # Add L1 regularization
            l1_loss = self._compute_l1_loss()
            total_loss_batch = loss + l1_loss
            
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for eeg_emb, gen_score, labels in dataloader:
                eeg_emb = eeg_emb.to(self.device)
                gen_score = gen_score.to(self.device)
                labels = labels.to(self.device)
                
                risk_score, alpha = self.model(eeg_emb, gen_score)
                loss = self.criterion(risk_score, labels)
                
                total_loss += loss.item()
                all_preds.extend(risk_score.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
        
        avg_loss = total_loss / len(dataloader)
        
        # Compute AUC
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except:
            auc = 0.5
        
        return avg_loss, auc
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 30,
        save_path: Optional[str] = None,
    ) -> Dict:
        """
        Train the fusion layer.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            save_path: Path to save the best model
            
        Returns:
            history: Training history dict
        """
        print("=" * 60)
        print("Training Fusion Layer")
        print("=" * 60)
        
        best_val_loss = float('inf')
        best_val_auc = 0.0
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_auc = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_aucs.append(val_auc)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_auc = val_auc
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    print(f"  Saved best model (val_loss={val_loss:.4f}, val_auc={val_auc:.4f})")
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch+1}/{epochs}: "
                      f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                      f"val_auc={val_auc:.4f}, lr={lr:.6f}")
        
        print(f"\nBest validation: loss={best_val_loss:.4f}, AUC={best_val_auc:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_aucs': self.val_aucs,
            'best_val_loss': best_val_loss,
            'best_val_auc': best_val_auc,
        }


# ============================================================
# Fusion Inference
# ============================================================
class FusionInference:
    """
    Inference class for the trained fusion model.
    
    Combines EEG and Genetic predictions with learned attention weights.
    """
    
    def __init__(
        self,
        fusion_model_path: str,
        eeg_model_path: str,
        genetic_model_path: str,
        device: str = 'cpu',
    ):
        """
        Args:
            fusion_model_path: Path to trained fusion layer (.pt)
            eeg_model_path: Path to trained BiLSTM (.net)
            genetic_model_path: Path to trained XGBoost (.pkl)
            device: Device for inference
        """
        self.device = torch.device(device)
        
        # Load fusion model
        self.fusion_model = AttentionGateFusion()
        self.fusion_model.load_state_dict(torch.load(fusion_model_path, map_location=device))
        self.fusion_model.to(device)
        self.fusion_model.eval()
        
        # Load branch models
        self.eeg_branch = EEGBranchWrapper(eeg_model_path, device)
        self.genetic_branch = GeneticBranchWrapper(genetic_model_path)
        
        print("Fusion inference model loaded successfully")
    
    def predict(
        self,
        eeg_window: torch.Tensor,
        genetic_features: np.ndarray,
    ) -> Dict:
        """
        Get fused prediction for a patient.
        
        Args:
            eeg_window: EEG data (1, time_steps, channels)
            genetic_features: Genetic features (1, n_features)
            
        Returns:
            dict with keys:
                - risk_score: Fused seizure risk score [0, 1]
                - eeg_score: EEG branch prediction
                - genetic_score: Genetic branch prediction
                - attention_weight: α (how much to trust EEG)
        """
        with torch.no_grad():
            # Get EEG embedding and score
            eeg_embedding, eeg_score = self.eeg_branch.get_embedding(eeg_window)
            
            # Get genetic score
            genetic_score = self.genetic_branch.get_risk_score(genetic_features)
            genetic_score_tensor = torch.FloatTensor(genetic_score).to(self.device)
            
            # Get fused prediction
            risk_score, alpha = self.fusion_model(
                eeg_embedding, genetic_score_tensor, return_attention=True
            )
            
            return {
                'risk_score': float(risk_score.cpu().numpy().flatten()[0]),
                'eeg_score': float(eeg_score.cpu().numpy().flatten()[0]),
                'genetic_score': float(genetic_score.flatten()[0]),
                'attention_weight': float(alpha.cpu().numpy().flatten()[0]),
            }
    
    def predict_batch(
        self,
        eeg_windows: torch.Tensor,
        genetic_features: np.ndarray,
    ) -> List[Dict]:
        """Get predictions for a batch of patients."""
        results = []
        for i in range(len(eeg_windows)):
            result = self.predict(
                eeg_windows[i:i+1],
                genetic_features[i:i+1]
            )
            results.append(result)
        return results


# ============================================================
# Main (for testing)
# ============================================================
if __name__ == '__main__':
    # Test the fusion layer
    print("Testing AttentionGateFusion...")
    
    model = AttentionGateFusion(eeg_embedding_dim=64, genetic_dim=1)
    
    # Test forward pass
    batch_size = 8
    eeg_embedding = torch.randn(batch_size, 64)
    genetic_score = torch.rand(batch_size, 1)
    
    risk_score, alpha = model(eeg_embedding, genetic_score)
    
    print(f"  Input: eeg={eeg_embedding.shape}, genetic={genetic_score.shape}")
    print(f"  Output: risk_score={risk_score.shape}, alpha={alpha.shape}")
    print(f"  Risk score range: [{risk_score.min():.3f}, {risk_score.max():.3f}]")
    print(f"  Attention weight range: [{alpha.min():.3f}, {alpha.max():.3f}]")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nFusion layer test passed!")
