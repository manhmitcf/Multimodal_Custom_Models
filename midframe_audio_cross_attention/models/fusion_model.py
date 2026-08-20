"""The only changed architecture: multimodal fusion over source-baseline encoders."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CrossAttentionHead(nn.Module):
    def __init__(self, audio_dim: int, video_dim: int, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.audio_projection = nn.Linear(audio_dim, d_model)
        self.video_projection = nn.Linear(video_dim, d_model)
        self.audio_positions = nn.Parameter(Tensor(1, 6, d_model).normal_(mean=0.0, std=0.02))
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
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 4, d_model))
        self.ffn_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 4)
        self.concat_classifier = nn.Sequential(
            nn.Linear(audio_dim + video_dim, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 4)
        )

    def forward(self, audio_tokens: Tensor, video_feature: Tensor) -> tuple[Tensor, Tensor]:
        if audio_tokens.ndim != 3 or audio_tokens.shape[1] != 6:
            raise ValueError("audio tokens must have shape [batch, 6, 512]")
        audio = self.audio_self_attention(self.audio_projection(audio_tokens) + self.audio_positions)
        query = self.video_projection(video_feature).unsqueeze(1)
        attended, attention = self.cross_attention(query, audio, audio, need_weights=True, average_attn_weights=False)
        fused = self.cross_norm(query + attended)
        fused = self.ffn_norm(fused + self.ffn(fused))
        return self.classifier(fused.squeeze(1)), attention

    def concat_baseline(self, audio_tokens: Tensor, video_feature: Tensor) -> Tensor:
        return self.concat_classifier(torch.cat((audio_tokens.mean(dim=1), video_feature), dim=1))


class BaselineSourceMultimodal(nn.Module):
    """Source encoders plus only the proposal's new fusion architecture."""

    def __init__(self, audio_encoder: nn.Module, video_encoder: nn.Module, d_model: int, num_heads: int, encoder_mode: str, dropout: float) -> None:
        super().__init__()
        if encoder_mode not in {"frozen", "tune"}:
            raise ValueError("encoder_mode must be frozen or tune")
        self.audio_encoder = audio_encoder
        self.video_encoder = video_encoder
        self.encoder_mode = encoder_mode
        self.fusion = CrossAttentionHead(audio_encoder.feature_dim, video_encoder.feature_dim, d_model, num_heads, dropout)
        self.configure_encoder_mode()

    def _open_prefixes(self) -> tuple[str, ...]:
        video = {
            "densenet121": ("video_encoder.model.backbone.model.features.denseblock4", "video_encoder.model.backbone.model.features.norm5"),
            "efficientnet_b0": ("video_encoder.model.backbone.model.features.7", "video_encoder.model.backbone.model.features.8"),
            "mobilenet_v2": tuple(f"video_encoder.model.backbone.model.features.{index}" for index in range(14, 19)),
            "swin_tiny": ("video_encoder.model.backbone.model.features.7", "video_encoder.model.backbone.model.norm"),
        }
        return ("audio_encoder.model.backbone.conv_block4", "audio_encoder.model.backbone.fc1", *video[self.video_encoder.name])

    def configure_encoder_mode(self) -> None:
        for encoder in (self.audio_encoder, self.video_encoder):
            for parameter in encoder.parameters():
                parameter.requires_grad = False
        if self.encoder_mode == "tune":
            prefixes = self._open_prefixes()
            for name, parameter in self.named_parameters():
                if any(name.startswith(prefix) for prefix in prefixes):
                    parameter.requires_grad = True

    def train(self, mode: bool = True) -> "BaselineSourceMultimodal":
        super().train(mode)
        if mode:
            self.audio_encoder.eval()
            self.video_encoder.eval()
            if self.encoder_mode == "tune":
                for prefix in self._open_prefixes():
                    self.get_submodule(prefix).train()
        return self

    def forward(self, waveforms: Tensor, images: Tensor, return_attention: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        logits, attention = self.fusion(self.audio_encoder(waveforms), self.video_encoder(images))
        return (logits, attention) if return_attention else logits

    def forward_concat_baseline(self, waveforms: Tensor, images: Tensor) -> Tensor:
        return self.fusion.concat_baseline(self.audio_encoder(waveforms), self.video_encoder(images))


class SpatialCrossAttentionHead(nn.Module):
    """Classify stage-3 Swin visual tokens conditioned on six audio tokens."""

    def __init__(self, audio_dim: int, video_dim: int, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.audio_projection = nn.Linear(audio_dim, d_model)
        self.video_projection = nn.Sequential(nn.Linear(video_dim, d_model), nn.LayerNorm(d_model))
        self.audio_positions = nn.Parameter(Tensor(1, 6, d_model).normal_(mean=0.0, std=0.02))
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
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 4, d_model))
        self.ffn_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 4)

    def forward(self, audio_tokens: Tensor, video_tokens: Tensor) -> tuple[Tensor, Tensor]:
        if audio_tokens.ndim != 3 or audio_tokens.shape[1] != 6:
            raise ValueError("audio tokens must have shape [batch, 6, channels]")
        if video_tokens.ndim != 3:
            raise ValueError("video tokens must have shape [batch, tokens, channels]")
        audio = self.audio_self_attention(self.audio_projection(audio_tokens) + self.audio_positions)
        visual = self.video_projection(video_tokens)
        attended, attention = self.cross_attention(visual, audio, audio, need_weights=True, average_attn_weights=False)
        fused = self.cross_norm(visual + attended)
        fused = self.ffn_norm(fused + self.ffn(fused))
        return self.classifier(fused.mean(dim=1)), attention


class SwinSpatialMultimodal(nn.Module):
    """PANNS audio tokens fused with iBOT-adapted Swin stage-3 spatial tokens."""

    def __init__(self, audio_encoder: nn.Module, video_encoder: nn.Module, d_model: int, num_heads: int, encoder_mode: str, dropout: float) -> None:
        super().__init__()
        if encoder_mode not in {"frozen", "tune"}:
            raise ValueError("encoder_mode must be frozen or tune")
        self.audio_encoder = audio_encoder
        self.video_encoder = video_encoder
        self.encoder_mode = encoder_mode
        self.fusion = SpatialCrossAttentionHead(audio_encoder.feature_dim, video_encoder.feature_dim, d_model, num_heads, dropout)
        self.configure_encoder_mode()

    def _open_prefixes(self) -> tuple[str, ...]:
        return (
            "audio_encoder.model.backbone.conv_block4",
            "audio_encoder.model.backbone.fc1",
            "video_encoder.model.backbone.model.features.5",
            "video_encoder.model.backbone.model.features.6",
            "video_encoder.model.backbone.model.features.7",
            "video_encoder.model.backbone.model.norm",
        )

    def configure_encoder_mode(self) -> None:
        for encoder in (self.audio_encoder, self.video_encoder):
            for parameter in encoder.parameters():
                parameter.requires_grad = False
        if self.encoder_mode == "tune":
            for name, parameter in self.named_parameters():
                if any(name.startswith(prefix) for prefix in self._open_prefixes()):
                    parameter.requires_grad = True

    def train(self, mode: bool = True) -> "SwinSpatialMultimodal":
        super().train(mode)
        if mode:
            self.audio_encoder.eval()
            self.video_encoder.eval()
            if self.encoder_mode == "tune":
                for prefix in self._open_prefixes():
                    self.get_submodule(prefix).train()
        return self

    def forward(self, waveforms: Tensor, images: Tensor, return_attention: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        logits, attention = self.fusion(self.audio_encoder(waveforms), self.video_encoder(images))
        return (logits, attention) if return_attention else logits
