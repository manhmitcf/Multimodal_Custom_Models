"""The only changed architecture: multimodal fusion over source-baseline encoders."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _sinusoidal_1d(length: int, d_model: int) -> Tensor:
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
    encoding = torch.zeros(length, d_model, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies)
    return encoding.unsqueeze(0)


def _sinusoidal_2d(grid_size: int, d_model: int) -> Tensor:
    if d_model % 4:
        raise ValueError("d_model must be divisible by 4 for 2D sinusoidal visual positions")
    row_encoding = _sinusoidal_1d(grid_size, d_model // 2).squeeze(0)
    column_encoding = _sinusoidal_1d(grid_size, d_model // 2).squeeze(0)
    positions = [torch.cat((row_encoding[row], column_encoding[column])) for row in range(grid_size) for column in range(grid_size)]
    return torch.stack(positions).unsqueeze(0)


class SpatialCrossAttentionHead(nn.Module):
    """Audio/video token encoders followed by video-query-to-audio cross-attention."""

    def __init__(
        self,
        audio_dim: int,
        video_dim: int,
        d_model: int,
        num_heads: int,
        visual_grid_size: int,
        positional_encoding: str,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if positional_encoding not in {"none", "learned", "sinusoidal"}:
            raise ValueError("positional_encoding must be none, learned, or sinusoidal")
        self.audio_projection = nn.Linear(audio_dim, d_model)
        self.video_projection = nn.Linear(video_dim, d_model)
        self.num_visual_tokens = visual_grid_size * visual_grid_size
        self.positional_encoding = positional_encoding
        if positional_encoding == "learned":
            self.audio_positions = nn.Parameter(Tensor(1, 6, d_model).normal_(mean=0.0, std=0.02))
            self.visual_positions = nn.Parameter(Tensor(1, self.num_visual_tokens, d_model).normal_(mean=0.0, std=0.02))
        elif positional_encoding == "sinusoidal":
            self.register_buffer("audio_positions", _sinusoidal_1d(6, d_model), persistent=False)
            self.register_buffer("visual_positions", _sinusoidal_2d(visual_grid_size, d_model), persistent=False)
        else:
            self.audio_positions = None
            self.visual_positions = None
        self.audio_self_attention = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.visual_self_attention = nn.TransformerEncoderLayer(
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

    def forward(self, audio_tokens: Tensor, video_tokens: Tensor) -> tuple[Tensor, Tensor]:
        if audio_tokens.ndim != 3 or audio_tokens.shape[1] != 6:
            raise ValueError("audio tokens must have shape [batch, 6, 512]")
        if video_tokens.ndim != 3 or video_tokens.shape[1] != self.num_visual_tokens:
            raise ValueError(f"video tokens must have shape [batch, {self.num_visual_tokens}, channels]")
        audio = self.audio_projection(audio_tokens)
        visual = self.video_projection(video_tokens)
        if self.audio_positions is not None:
            audio = audio + self.audio_positions
            visual = visual + self.visual_positions
        audio = self.audio_self_attention(audio)
        visual = self.visual_self_attention(visual)
        attended, attention = self.cross_attention(visual, audio, audio, need_weights=True, average_attn_weights=False)
        fused = self.cross_norm(visual + attended)
        fused = self.ffn_norm(fused + self.ffn(fused))
        return self.classifier(fused.mean(dim=1)), attention

    def concat_baseline(self, audio_tokens: Tensor, video_tokens: Tensor) -> Tensor:
        return self.concat_classifier(torch.cat((audio_tokens.mean(dim=1), video_tokens.mean(dim=1)), dim=1))


class DinoCrossAttentionHead(nn.Module):
    """DINO patch queries attend to contextual PANNS audio tokens."""

    def __init__(
        self,
        audio_dim: int,
        visual_dim: int,
        d_model: int,
        num_heads: int,
        audio_positional_encoding: str,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if audio_positional_encoding not in {"none", "learned", "sinusoidal"}:
            raise ValueError("audio_positional_encoding must be none, learned, or sinusoidal")
        self.audio_projection = nn.Linear(audio_dim, d_model)
        self.visual_projection = nn.Sequential(nn.Linear(visual_dim, d_model), nn.LayerNorm(d_model))
        self.audio_positional_encoding = audio_positional_encoding
        if audio_positional_encoding == "learned":
            self.audio_positions = nn.Parameter(Tensor(1, 6, d_model).normal_(mean=0.0, std=0.02))
        elif audio_positional_encoding == "sinusoidal":
            self.register_buffer("audio_positions", _sinusoidal_1d(6, d_model), persistent=False)
        else:
            self.audio_positions = None
        # DINO patch tokens already contain spatial position information.
        self.visual_positions = None
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

    def forward(self, audio_tokens: Tensor, visual_tokens: Tensor) -> tuple[Tensor, Tensor]:
        if audio_tokens.ndim != 3 or audio_tokens.shape[1] != 6:
            raise ValueError("audio tokens must have shape [batch, 6, channels]")
        if visual_tokens.ndim != 3:
            raise ValueError("DINO patch tokens must have shape [batch, patches, channels]")
        audio = self.audio_projection(audio_tokens)
        visual = self.visual_projection(visual_tokens)
        if self.audio_positions is not None:
            audio = audio + self.audio_positions
        audio = self.audio_self_attention(audio)
        attended, attention = self.cross_attention(visual, audio, audio, need_weights=True, average_attn_weights=False)
        fused = self.cross_norm(visual + attended)
        fused = self.ffn_norm(fused + self.ffn(fused))
        return self.classifier(fused.mean(dim=1)), attention


class BaselineSourceMultimodal(nn.Module):
    """Source encoders plus the spatial-token cross-attention fusion architecture."""

    def __init__(
        self,
        audio_encoder: nn.Module,
        video_encoder: nn.Module,
        d_model: int,
        num_heads: int,
        encoder_mode: str,
        positional_encoding: str,
        dropout: float,
    ) -> None:
        super().__init__()
        if encoder_mode not in {"frozen", "tune"}:
            raise ValueError("encoder_mode must be frozen or tune")
        self.audio_encoder = audio_encoder
        self.video_encoder = video_encoder
        self.encoder_mode = encoder_mode
        self.fusion = SpatialCrossAttentionHead(
            audio_encoder.feature_dim,
            video_encoder.feature_dim,
            d_model,
            num_heads,
            video_encoder.grid_size,
            positional_encoding,
            dropout,
        )
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


class PannsDinoMultimodal(nn.Module):
    """PANNS Cnn6 audio tokens fused with DINOv2 patch-token queries."""

    def __init__(
        self,
        audio_encoder: nn.Module,
        dino_encoder: nn.Module,
        d_model: int,
        num_heads: int,
        audio_encoder_mode: str,
        audio_positional_encoding: str,
        dropout: float,
    ) -> None:
        super().__init__()
        if audio_encoder_mode not in {"frozen", "tune"}:
            raise ValueError("audio_encoder_mode must be frozen or tune")
        self.audio_encoder = audio_encoder
        self.dino_encoder = dino_encoder
        self.audio_encoder_mode = audio_encoder_mode
        self.fusion = DinoCrossAttentionHead(
            audio_dim=audio_encoder.feature_dim,
            visual_dim=dino_encoder.feature_dim,
            d_model=d_model,
            num_heads=num_heads,
            audio_positional_encoding=audio_positional_encoding,
            dropout=dropout,
        )
        self._configure_audio_encoder()

    def _configure_audio_encoder(self) -> None:
        for parameter in self.audio_encoder.parameters():
            parameter.requires_grad = False
        if self.audio_encoder_mode == "tune":
            for name, parameter in self.audio_encoder.named_parameters():
                if name.startswith("model.backbone.conv_block4") or name.startswith("model.backbone.fc1"):
                    parameter.requires_grad = True

    def train(self, mode: bool = True) -> "PannsDinoMultimodal":
        super().train(mode)
        if mode:
            self.audio_encoder.eval()
            self.dino_encoder.eval()
            if self.audio_encoder_mode == "tune":
                self.audio_encoder.model.backbone.conv_block4.train()
                self.audio_encoder.model.backbone.fc1.train()
            if self.dino_encoder.encoder_mode == "full":
                self.dino_encoder.model.train()
        return self

    def forward(self, waveforms: Tensor, images: Tensor, return_attention: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        logits, attention = self.fusion(self.audio_encoder(waveforms), self.dino_encoder(images))
        return (logits, attention) if return_attention else logits
