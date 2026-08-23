"""High-Resolution STFT dB Spectrogram to RGB Image Transformation Module."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class STFTTodBImageTransform(nn.Module):
    """Converts 2s audio waveform [B, 128000] into a 3-channel 224x224 STFT dB Spectrogram Image [B, 3, 224, 224]."""

    def __init__(self, n_fft: int = 2048, hop_length: int = 512, image_size: int = 224) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.image_size = image_size
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: Tensor of shape [batch, 128000] (64 kHz x 2s audio).
        Returns:
            db_image: Normalized 3-channel image Tensor of shape [batch, 3, 224, 224].
        """
        # Ensure waveform is 2D [batch, samples]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # STFT computation
        stft_complex = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(waveform.device),
            return_complex=True,
        )  # Shape: [batch, n_fft // 2 + 1, time_steps]

        # Magnitude & Power-to-dB conversion
        power = torch.abs(stft_complex) ** 2
        db_spec = 10.0 * torch.log10(power + 1e-10)  # Shape: [batch, freq, time]

        # Add channel dimension: [batch, 1, freq, time]
        db_spec = db_spec.unsqueeze(1)

        # Min-Max Normalization per batch sample
        batch_size = db_spec.size(0)
        flat_db = db_spec.reshape(batch_size, -1)
        min_val = flat_db.min(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
        max_val = flat_db.max(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
        norm_spec = (db_spec - min_val) / (max_val - min_val + 1e-6)

        # Interpolate 2D Spectrogram to target Image Size (224x224)
        resized_spec = F.interpolate(
            norm_spec,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )  # Shape: [batch, 1, 224, 224]

        # Replicate 1 channel into 3 RGB channels [batch, 3, 224, 224]
        db_image = resized_spec.repeat(1, 3, 1, 1)
        return db_image
