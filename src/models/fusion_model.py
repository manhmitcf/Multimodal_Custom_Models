"""Pure Original Spatial Cross-Attention Fusion Architecture (`exp/midframe-to-audio-cross-attn`)."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CrossAttentionHead(nn.Module):
    """Spatial Cross-Attention Head: Video Query (1280d -> 256d) Cross-Attends to Audio Tokens (512d -> 256d)."""

    def __init__(self, audio_dim: int = 512, video_dim: int = 1280, d_model: int = 256, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.audio_projection = nn.Linear(audio_dim, d_model)
        self.video_projection = nn.Linear(video_dim, d_model)

        self.audio_positions = nn.Parameter(torch.randn(1, 6, d_model) * 0.02)
        self.audio_self_attention = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 4)

    def forward(self, audio_tokens: Tensor, video_feature: Tensor) -> tuple[Tensor, Tensor]:
        if audio_tokens.ndim != 3 or audio_tokens.shape[1] != 6:
            raise ValueError("audio tokens must have shape [batch, 6, 512]")

        # 1. Project Audio Tokens (512 -> 256) + Positional Embedding + Self Attention
        audio = self.audio_self_attention(self.audio_projection(audio_tokens) + self.audio_positions)

        # 2. Project Video Feature (1280 -> 256) as Video Query
        query = self.video_projection(video_feature).unsqueeze(1) # [B, 1, 256]

        # 3. Cross-Attention: Video Query attends to 6 Audio Tokens
        attended, attention = self.cross_attention(query, audio, audio, need_weights=True, average_attn_weights=False)

        # 4. Residual LayerNorm + FFN + Classification
        fused = self.cross_norm(query + attended)
        fused = self.ffn_norm(fused + self.ffn(fused))
        logits = self.classifier(fused.squeeze(1))

        return logits, attention


class BaselineSourceMultimodal(nn.Module):
    """End-to-End Multimodal Model: Source Encoders + Spatial Cross-Attention Fusion."""

    def __init__(
        self,
        audio_encoder: nn.Module,
        video_encoder: nn.Module,
        d_model: int = 256,
        num_heads: int = 4,
        encoder_mode: str = "frozen",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.audio_encoder = audio_encoder
        self.video_encoder = video_encoder
        self.encoder_mode = encoder_mode

        audio_dim = getattr(audio_encoder, "feature_dim", 512)
        video_dim = getattr(video_encoder, "feature_dim", 1280)

        self.fusion = CrossAttentionHead(
            audio_dim=audio_dim,
            video_dim=video_dim,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.configure_encoder_mode()

    def configure_encoder_mode(self) -> None:
        for encoder in (self.audio_encoder, self.video_encoder):
            for parameter in encoder.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True) -> BaselineSourceMultimodal:
        super().train(mode)
        if mode:
            # Strictly freeze BatchNorm running stats in eval mode
            self.audio_encoder.eval()
            self.video_encoder.eval()
        return self

    def forward(self, waveforms: Tensor, images: Tensor, return_attention: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        audio_tokens = self.audio_encoder(waveforms) # [B, 6, 512]
        video_feat = self.video_encoder(images) # [B, 1280]
        logits, attention = self.fusion(audio_tokens, video_feat)
        return (logits, attention) if return_attention else logits
