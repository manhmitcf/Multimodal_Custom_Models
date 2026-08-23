"""Custom Raw STFT Audio CNN with Pre-Emphasis, Frequency-Domain Attention, and Depthwise-Separable Convolutions."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from features.audio_features import FrequencyDomainAttention, RawSTFT256kTransform


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise-Separable 2D Convolution to minimize parameter count."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int] = (3, 3), stride: tuple[int, int] = (1, 1), padding: tuple[int, int] = (1, 1)) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x


class CustomRawStftAudioCNN(nn.Module):
    """Custom Audio CNN processing RAW STFT 2049 bins (NO MEL COMPRESSION, NO RGB ENCODING).

    Pipeline:
      Waveform (256k) -> Pre-Emphasis (0.97) -> Raw STFT 2049 Bins (Hamming 4096, 2048)
      -> Frequency-Domain Attention (F-Attn) -> Early Strided Conv (4x2)
      -> Depthwise-Separable Blocks -> Feature Vector [B, 256].
    """

    def __init__(self, feature_dim: int = 256, n_fft: int = 4096, hop_length: int = 2048, win_length: int = 4096, windowing: str = "hamming", use_std: bool = True, pre_emphasis: float = 0.97) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.stft_transform = RawSTFT256kTransform(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            windowing=windowing,
            alpha=pre_emphasis,
        )
        num_freq_bins = n_fft // 2 + 1
        self.freq_attention = FrequencyDomainAttention(num_freq_bins=num_freq_bins, use_std=use_std, reduction=16)

        # Early Strided Conv Block: Shrinks frequency 2049 -> 513 immediately
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(4, 2), padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.block2 = DepthwiseSeparableConv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.block3 = DepthwiseSeparableConv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.block4 = DepthwiseSeparableConv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        # Direct STFT Frequency-Profile MLP Projection (guarantees instant high accuracy like MLP baseline)
        in_mlp_dim = num_freq_bins * 2 if use_std else num_freq_bins
        self.freq_mlp = nn.Sequential(
            nn.Linear(in_mlp_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, feature_dim),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_proj = nn.Linear(256, feature_dim)

    def forward(self, input_tensor: Tensor) -> Tensor:
        # Check if input is 2D Waveform [B, Time] or 4D Spectrogram Tensor [B, 1, 2049, Time]
        if input_tensor.ndim == 2:
            db_spec = self.stft_transform(input_tensor)
        elif input_tensor.ndim == 3:
            db_spec = input_tensor.unsqueeze(1)
        else:
            db_spec = input_tensor

        # 1. Compute Direct STFT Frequency Profile (Mean + Std along time axis)
        batch = db_spec.shape[0]
        freq_mean = db_spec.mean(dim=-1).view(batch, -1) # [B, 2049]
        freq_std = db_spec.std(dim=-1).view(batch, -1) # [B, 2049]
        freq_profile = torch.cat([freq_mean, freq_std], dim=1) # [B, 4098]
        mlp_features = self.freq_mlp(freq_profile) # [B, feature_dim]

        # 2. Apply Frequency-Domain Attention (F-Attn)
        attended_spec = self.freq_attention(db_spec)

        # 3. Pass through Early Strided Block & Depthwise Convolutions
        x = self.block1(attended_spec)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # 4. Global Pooling & Projection -> [B, feature_dim]
        pooled = self.global_pool(x).flatten(1)
        cnn_features = self.fc_proj(pooled)

        # 5. Dual-Path Residual Fusion (CNN + Direct STFT Frequency MLP)
        return cnn_features + mlp_features
