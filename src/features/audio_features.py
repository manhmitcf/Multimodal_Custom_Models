"""Raw STFT 256k (No Mel Compression, No RGB Image Encoding) with Pre-Emphasis and Frequency-Domain Attention."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PreEmphasisFilter(nn.Module):
    """Pre-Emphasis High-Pass Filter: y[t] = x[t] - alpha * x[t-1].

    Boosts high-frequency acoustics (2 kHz - 8 kHz) of feeding splashes.
    """

    def __init__(self, alpha: float = 0.97) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, waveforms: Tensor) -> Tensor:
        if waveforms.ndim != 2:
            raise ValueError(f"Expected 2D waveforms tensor [Batch, Time], got shape {waveforms.shape}")
        emphasized = torch.cat([waveforms[:, :1], waveforms[:, 1:] - self.alpha * waveforms[:, :-1]], dim=1)
        return emphasized


class RawSTFT256kTransform(nn.Module):
    """Raw STFT Transform (n_fft=4096, win=4096, hop=2048, window=hamming) WITHOUT Mel Compression.

    Output shape: [Batch, 1, 2049, 250] Log-Power Spectrogram in dB.
    """

    def __init__(self, n_fft: int = 4096, hop_length: int = 2048, win_length: int = 4096, windowing: str = "hamming", alpha: float = 0.97) -> None:
        super().__init__()
        self.pre_emphasis = PreEmphasisFilter(alpha=alpha)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.windowing = windowing.lower()

        if self.windowing == "hamming":
            win_tensor = torch.hamming_window(win_length)
        elif self.windowing in ("hann", "hanning"):
            win_tensor = torch.hann_window(win_length)
        else:
            raise ValueError(f"Unsupported windowing: {windowing}")

        self.register_buffer("window", win_tensor)

    def forward(self, waveforms: Tensor) -> Tensor:
        # 1. Apply Pre-Emphasis Filter (alpha=0.97)
        emphasized = self.pre_emphasis(waveforms)

        # 2. Compute Raw STFT
        stft_complex = torch.stft(
            emphasized,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(waveforms.device),
            center=True,
            return_complex=True,
        )

        # 3. Compute Power Spectrogram in Log-dB scale
        power_spec = torch.abs(stft_complex) ** 2
        db_spec = 10.0 * torch.log10(power_spec + 1e-10)

        # 4. Reshape to 4D Tensor [Batch, 1, 2049, Time_Frames]
        db_spec = db_spec.unsqueeze(1)
        return db_spec


class FrequencyDomainAttention(nn.Module):
    """Frequency-Domain Attention (F-Attention) on 2049 Raw STFT Frequency Bins.

    Uses Mean and Std energy profiling along time axis to learn frequency attention weights
    a_F in [0, 1]^2049. Highlights feeding acoustics (2 kHz - 8 kHz) and suppresses motor noise (0 - 500 Hz).
    """

    def __init__(self, num_freq_bins: int = 2049, use_std: bool = True, reduction: int = 16) -> None:
        super().__init__()
        self.num_freq_bins = num_freq_bins
        self.use_std = use_std

        in_dim = num_freq_bins * 2 if use_std else num_freq_bins
        self.freq_gate = nn.Sequential(
            nn.Linear(in_dim, num_freq_bins // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_freq_bins // reduction, num_freq_bins, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, spectrogram: Tensor) -> Tensor:
        # spectrogram shape: [Batch, 1, 2049, Time_Frames]
        batch, channels, freq, time = spectrogram.shape

        # 1. Temporal Energy Profiling: Mean and Std along Time Axis
        freq_mean = spectrogram.mean(dim=-1).view(batch, freq) # [B, 2049]
        if self.use_std:
            freq_std = spectrogram.std(dim=-1).view(batch, freq) # [B, 2049]
            freq_profile = torch.cat([freq_mean, freq_std], dim=1) # [B, 4098]
        else:
            freq_profile = freq_mean

        # 2. Learn Frequency-wise Attention Weights -> [B, 1, 2049, 1]
        freq_weights = self.freq_gate(freq_profile).view(batch, 1, freq, 1)

        # 3. Apply Frequency Attention Weights to Spectrogram Matrix
        return spectrogram * freq_weights
