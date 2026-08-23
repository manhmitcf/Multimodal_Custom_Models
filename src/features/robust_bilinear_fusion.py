"""Method 4 (R-BPMD): Robust Multi-level Factorized Bilinear Pooling with Modality Dropout.

Provides bilinear interaction between audio and visual modalities and Modality Dropout
to ensure robust classification under missing or noisy sensor inputs.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ModalityDropout(nn.Module):
    """Randomly zeroes out audio or visual feature streams during training.

    Simulates sensor degradation (e.g. camera lens glare or water pump audio noise).
    """

    def __init__(self, drop_prob: float = 0.15) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, audio_tokens: Tensor, visual_tokens: Tensor) -> tuple[Tensor, Tensor]:
        if not self.training or self.drop_prob <= 0.0:
            return audio_tokens, visual_tokens

        batch_size = audio_tokens.shape[0]
        device = audio_tokens.device

        # Randomly pick modality to drop per sample: 0=none, 1=audio, 2=visual
        rand_vals = torch.rand(batch_size, device=device)
        audio_mask = (rand_vals > self.drop_prob).float().view(-1, 1, 1)
        visual_mask = (rand_vals <= (1.0 - self.drop_prob)).float().view(-1, 1, 1)

        return audio_tokens * audio_mask, visual_tokens * visual_mask


class MultiFactorizedBilinearPooling(nn.Module):
    """Multi-level Factorized Bilinear Pooling (MFB) layer.

    Computes low-rank Hadamard product interaction between audio and visual projections,
    followed by sum pooling, L2 normalization, and power normalization.
    """

    def __init__(self, audio_dim: int, visual_dim: int, out_dim: int = 256, factor_k: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.factor_k = factor_k

        # Project inputs to factorized dimension (out_dim * factor_k)
        self.audio_proj = nn.Linear(audio_dim, out_dim * factor_k)
        self.visual_proj = nn.Linear(visual_dim, out_dim * factor_k)

        self.dropout = nn.Dropout(dropout)
        self.out_projection = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, audio_feat: Tensor, visual_feat: Tensor) -> Tensor:
        """audio_feat: [batch, audio_dim], visual_feat: [batch, visual_dim]

        Output: fused_feat [batch, out_dim]
        """
        px = self.audio_proj(audio_feat)    # [batch, out_dim * factor_k]
        py = self.visual_proj(visual_feat)  # [batch, out_dim * factor_k]

        # Element-wise product (Hadamard product)
        elementwise_prod = self.dropout(px * py)

        # Sum-pooling across factor_k groups: [batch, out_dim, factor_k] -> [batch, out_dim]
        batch_size = audio_feat.shape[0]
        reshaped = elementwise_prod.reshape(batch_size, self.out_dim, self.factor_k)
        summed = reshaped.sum(dim=-1)

        # Power normalization (signed square root) & L2 normalization
        signed_sqrt = torch.sign(summed) * torch.sqrt(torch.abs(summed) + 1e-8)
        normed = torch.nn.functional.normalize(signed_sqrt, p=2, dim=-1)

        return self.out_projection(normed)
