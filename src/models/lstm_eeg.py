#!/usr/bin/env python3
"""
CNN-LSTM EEG Seizure Prediction Model
=======================================
1D CNN front-end (3 layers) extracts local spatial-temporal features from
raw EEG, reducing sequence length 1280 → ~160 before the BiLSTM sees it.
This dramatically improves gradient flow compared to feeding raw 1280-step
sequences directly into the LSTM.

Architecture:
    Input [B, T=1280, C=17]
    → Conv1D(17→32, k=5, s=2) + BN + ReLU  → [B, 640, 32]
    → Conv1D(32→64, k=5, s=2) + BN + ReLU  → [B, 320, 64]
    → Conv1D(64→128, k=5, s=2) + BN + ReLU → [B, 160, 128]
    → BiLSTM(1 layer, hidden=128)           → [B, 160, 256]
    → Self-Attention                        → [B, 160, 256]
    → Global Avg Pool                       → [B, 256]
    → FC(256→64) + ReLU + Dropout(0.2)
    → FC(64→1) + Sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Scaled dot-product self-attention over LSTM timesteps."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, hidden_dim]
        Returns:
            attended: [batch, seq_len, hidden_dim]
            weights: [batch, seq_len, seq_len]
        """
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(weights, V)

        return attended, weights


class EEGCNNLSTM(nn.Module):
    """
    CNN-LSTM for EEG seizure prediction.

    Input:  [batch, T=1280, C=17]  (raw EEG sequences)
    Output: [batch, 1]             (seizure probability)
    """

    def __init__(self, input_size=17, hidden_size=128, num_layers=1,
                 dropout=0.2, embedding_dim=64):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        # ── 1D CNN Front-End ──
        # Reduces sequence length: 1280 → 640 → 320 → 160
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=5, stride=2, padding=2)
        self.bn1   = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2   = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn3   = nn.BatchNorm1d(128)

        cnn_output_channels = 128
        cnn_output_length   = 160   # 1280 / 2 / 2 / 2

        # ── Bidirectional LSTM ──
        # Now receives 160 timesteps instead of 1280
        self.lstm = nn.LSTM(
            input_size=cnn_output_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # ── Self-attention over LSTM timesteps ──
        self.attention = SelfAttention(hidden_size * 2)  # *2 for bidirectional

        # ── Classifier head ──
        self.fc1 = nn.Linear(hidden_size * 2, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(embedding_dim, 1)

    def forward(self, x, return_embedding=False):
        """
        Args:
            x: [batch, T, C] — raw EEG sequences
            return_embedding: if True, also return the 64-dim embedding

        Returns:
            logits: [batch, 1]
            embedding: [batch, 64] (only if return_embedding=True)
        """
        # x: [B, T, C] → [B, C, T] for Conv1d
        x = x.transpose(1, 2)   # [B, 17, 1280]

        # Conv block 1: [B, 17, 1280] → [B, 32, 640]
        x = F.relu(self.bn1(self.conv1(x)))
        # Conv block 2: [B, 32, 640] → [B, 64, 320]
        x = F.relu(self.bn2(self.conv2(x)))
        # Conv block 3: [B, 64, 320] → [B, 128, 160]
        x = F.relu(self.bn3(self.conv3(x)))

        # Back to [B, T, C] for LSTM
        x = x.transpose(1, 2)   # [B, 160, 128]

        # LSTM: [B, 160, 128] → [B, 160, 256]
        lstm_out, _ = self.lstm(x)

        # Self-attention: [B, 160, 256] → [B, 160, 256]
        attended, _ = self.attention(lstm_out)

        # Global average pooling: [B, 160, 256] → [B, 256]
        pooled = attended.mean(dim=1)

        # FC layers: [B, 256] → [B, 64] → [B, 1]
        embedding = F.relu(self.fc1(pooled))
        embedding = self.dropout(embedding)

        logits = self.fc2(embedding)

        if return_embedding:
            return logits, embedding
        return logits

    def get_embedding_dim(self):
        return self.embedding_dim
