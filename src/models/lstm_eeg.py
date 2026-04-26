#!/usr/bin/env python3
"""
LSTM EEG Seizure Prediction Model
====================================
Bidirectional LSTM with self-attention for binary seizure prediction
from raw EEG sequences [batch, T=1280, C=17].

Architecture:
    Input → BiLSTM(2 layers, hidden=128) → Self-Attention → AvgPool → FC(256→64) → FC(64→1) → Sigmoid
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


class EEGBiLSTM(nn.Module):
    """
    Bidirectional LSTM for EEG seizure prediction.

    Input:  [batch, T=1280, C=17]  (raw EEG sequences)
    Output: [batch, 1]             (seizure probability)
    """

    def __init__(self, input_size=17, hidden_size=128, num_layers=2,
                 dropout=0.3, embedding_dim=64):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        # Input batch normalization
        self.input_bn = nn.BatchNorm1d(input_size)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Self-attention over timesteps
        self.attention = SelfAttention(hidden_size * 2)  # *2 for bidirectional

        # Classifier head
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
        # Input normalization: [B, T, C] → [B, C, T] → BN → [B, T, C]
        x = x.transpose(1, 2)
        x = self.input_bn(x)
        x = x.transpose(1, 2)

        # LSTM: [B, T, C] → [B, T, 2*hidden]
        lstm_out, _ = self.lstm(x)

        # Self-attention: [B, T, 2*hidden] → [B, T, 2*hidden]
        attended, _ = self.attention(lstm_out)

        # Global average pooling: [B, T, 2*hidden] → [B, 2*hidden]
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
