#!/usr/bin/env python3
"""
STFT-CNN-LSTM EEG Seizure Prediction Model
============================================
Each 5-second EEG epoch is converted to an STFT magnitude spectrogram
(17 channels × 70 freq bins × ~21 time frames). A 2D CNN front-end
compresses the frequency axis while preserving the temporal axis, then
a BiLSTM processes the resulting time sequence.

Architecture:
    Input [B, C=17, F=70, T_stft=21]  (STFT magnitude spectrograms, log-scaled)
    → Conv2D(17→32, k=(3,3), s=(2,1)) + BN + ReLU  → [B, 32, 35, 21]
    → Conv2D(32→64, k=(3,3), s=(2,1)) + BN + ReLU  → [B, 64, 18, 21]
    → Conv2D(64→128, k=(3,3), s=(2,1)) + BN + ReLU → [B, 128, 9, 21]
    → AdaptiveAvgPool2d((1, None))                    → [B, 128, 1, 21]
    → Squeeze + transpose                              → [B, 21, 128]
    → BiLSTM(1 layer, hidden=128)                     → [B, 21, 256]
    → Self-Attention                                   → [B, 21, 256]
    → Global Avg Pool                                  → [B, 256]
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
    STFT-CNN-LSTM for EEG seizure prediction.

    Input:  [batch, C=17, F=70, T_stft]  (STFT magnitude spectrograms)
    Output: [batch, 1]                   (seizure probability)
    """

    def __init__(self, input_channels=17, hidden_size=128, num_layers=1,
                 dropout=0.2, embedding_dim=64):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        # ── 2D CNN Front-End on STFT spectrograms ──
        # Input: [B, 17, 70, T]  (channels, freq, time)
        # Stride=(2,1) downsamples frequency only, preserves time
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=(3, 3),
                               stride=(2, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3),
                               stride=(2, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3),
                               stride=(2, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)

        # Pool frequency dimension to 1, keep time as sequence for LSTM
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

        cnn_output_channels = 128
        # Time dimension flows through unchanged (stride=1 on time axis)

        # ── Bidirectional LSTM ──
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
            x: [batch, C, F, T] — STFT magnitude spectrograms
            return_embedding: if True, also return the 64-dim embedding

        Returns:
            logits: [batch, 1]
            embedding: [batch, 64] (only if return_embedding=True)
        """
        # 2D Conv block 1: [B, 17, 70, T] → [B, 32, 35, T]
        x = F.relu(self.bn1(self.conv1(x)))
        # 2D Conv block 2: [B, 32, 35, T] → [B, 64, 18, T]
        x = F.relu(self.bn2(self.conv2(x)))
        # 2D Conv block 3: [B, 64, 18, T] → [B, 128, 9, T]
        x = F.relu(self.bn3(self.conv3(x)))

        # Pool frequency to 1: [B, 128, 9, T] → [B, 128, 1, T]
        x = self.freq_pool(x)
        # Squeeze frequency: [B, 128, T]
        x = x.squeeze(2)
        # Transpose for LSTM: [B, T, 128]
        x = x.transpose(1, 2)

        # LSTM: [B, T, 128] → [B, T, 256]
        lstm_out, _ = self.lstm(x)

        # Self-attention: [B, T, 256] → [B, T, 256]
        attended, _ = self.attention(lstm_out)

        # Global average pooling: [B, T, 256] → [B, 256]
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
