"""Geometry & Water-Ripple Feature Extractors for Method 3 (GW-AVF).

Computes 2D Wavelet/High-Pass ripple energy maps and Delaunay/Density spatial grids
to enrich spatial visual tokens for audio-visual cross-attention fusion.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class RippleWaveletExtractor(nn.Module):
    """Extracts high-frequency water ripple energy maps on a 14x14 spatial grid.

    Uses trainable 2D high-pass filters (Laplacian & High-frequency Wavelet approximations)
    to detect water surface turbulence and expanding circular ripples.
    """

    def __init__(self, in_channels: int = 3, grid_size: int = 14, out_dim: int = 32) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.out_dim = out_dim

        # High-pass filters for horizontal, vertical, and diagonal high frequencies (Wavelet LH, HL, HH)
        laplacian_kernel = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], dtype=torch.float32)
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32)

        weight = torch.stack([laplacian_kernel, sobel_x, sobel_y]).unsqueeze(1).repeat(1, in_channels, 1, 1) / in_channels
        self.filters = nn.Parameter(weight, requires_grad=True)

        self.projection = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((grid_size, grid_size)),
            nn.Conv2d(16, out_dim, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, images: Tensor) -> Tensor:
        """Input: images [batch, 3, 224, 224] -> Output: ripple_features [batch, 196, out_dim]"""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected input shape [batch, 3, H, W], got {tuple(images.shape)}")

        # Convolve with high-pass ripple filters
        ripple_maps = torch.abs(F.conv2d(images, self.filters, padding=1))  # [batch, 3, H, W]
        projected = self.projection(ripple_maps)  # [batch, out_dim, 14, 14]

        # Reshape to spatial tokens: [batch, 196, out_dim]
        batch_size = images.shape[0]
        return projected.flatten(2).transpose(1, 2).contiguous()


class GeometryFlockingExtractor(nn.Module):
    """Computes spatial fish aggregation density & geometric Delaunay features on a 14x14 grid.

    Extracts intensity variance and spatial concentration maps to quantify fish schooling.
    """

    def __init__(self, in_channels: int = 3, grid_size: int = 14, out_dim: int = 32) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.out_dim = out_dim

        self.density_net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),  # 112x112
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),  # 56x56
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((grid_size, grid_size)),  # 14x14
            nn.Conv2d(32, out_dim, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, images: Tensor) -> Tensor:
        """Input: images [batch, 3, 224, 224] -> Output: geometry_features [batch, 196, out_dim]"""
        density_map = self.density_net(images)  # [batch, out_dim, 14, 14]
        return density_map.flatten(2).transpose(1, 2).contiguous()


class GeometryRippleFeatureExtractor(nn.Module):
    """Combines Visual Spatial Tokens + Water Ripple Energy + Geometry Density into

    Geometry-Enhanced Spatial Tokens for Cross-Attention.
    """

    def __init__(self, visual_dim: int = 384, d_model: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.ripple_extractor = RippleWaveletExtractor(in_channels=3, grid_size=14, out_dim=32)
        self.geometry_extractor = GeometryFlockingExtractor(in_channels=3, grid_size=14, out_dim=32)

        # Combined dimension: visual_dim (384) + 32 (ripple) + 32 (geometry) = 448
        in_dim = visual_dim + 32 + 32
        self.fusion_projection = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, visual_tokens: Tensor, images: Tensor) -> Tensor:
        """visual_tokens: [batch, 196, 384], images: [batch, 3, 224, 224]

        Output: geometry_enhanced_tokens [batch, 196, d_model]
        """
        ripple_tokens = self.ripple_extractor(images)      # [batch, 196, 32]
        geometry_tokens = self.geometry_extractor(images)  # [batch, 196, 32]

        concatenated = torch.cat([visual_tokens, ripple_tokens, geometry_tokens], dim=-1)  # [batch, 196, 448]
        return self.fusion_projection(concatenated)
