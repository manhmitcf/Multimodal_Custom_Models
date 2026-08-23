"""Multimodal & Single-Modality fusion architectures: baseline, spatial, Method 3 GW-AVF, Method 4 R-BPMD, and STFT 256k PANNS-MobileNet Advanced Fusion."""

from __future__ import annotations

import torch
from torch import Tensor, nn

try:
    from features.geometry_ripple_features import GeometryRippleFeatureExtractor
except ImportError:
    GeometryRippleFeatureExtractor = None

try:
    from features.robust_bilinear_fusion import RobustBilinearCrossAttentionHead
except ImportError:
    RobustBilinearCrossAttentionHead = None

from features.stft_db_spectrogram import STFTTodBImageTransform


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

    def __init__(self, audio_dim: int, video_dim: int, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.audio_projection = nn.Linear(audio_dim, d_model)
        self.video_projection = nn.Linear(video_dim, d_model)
        self.audio_positions = nn.Parameter(Tensor(1, 6, d_model).normal_(mean=0.0, std=0.02))
        self.video_positions = nn.Parameter(Tensor(1, 196, d_model).normal_(mean=0.0, std=0.02))

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
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 4),
        )

    def forward(self, audio_tokens: Tensor, visual_tokens: Tensor) -> tuple[Tensor, Tensor]:
        if audio_tokens.ndim != 3 or audio_tokens.shape[1] != 6:
            raise ValueError("audio tokens must have shape [batch, 6, 512]")
        if visual_tokens.ndim != 3 or visual_tokens.shape[1] != 196:
            raise ValueError("visual tokens must have shape [batch, 196, 384]")

        audio = self.audio_self_attention(self.audio_projection(audio_tokens) + self.audio_positions)
        queries = self.video_projection(visual_tokens) + self.video_positions

        attended, attention = self.cross_attention(queries, audio, audio, need_weights=True, average_attn_weights=False)
        fused = self.cross_norm(queries + attended)
        fused = self.ffn_norm(fused + self.ffn(fused))
        pooled = fused.mean(dim=1)
        return self.classifier(pooled), attention


class GeometryRippleCrossAttentionHead(nn.Module):
    """Method 3 (GW-AVF): Enhanced Swin visual spatial tokens with Delaunay Geometry & Wavelet Ripples."""

    def __init__(self, audio_dim: int, video_dim: int, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.feature_enricher = GeometryRippleFeatureExtractor(visual_dim=video_dim, d_model=d_model, dropout=dropout)
        self.audio_projection = nn.Linear(audio_dim, d_model)
        self.audio_positions = nn.Parameter(Tensor(1, 6, d_model).normal_(mean=0.0, std=0.02))
        self.video_positions = nn.Parameter(Tensor(1, 196, d_model).normal_(mean=0.0, std=0.02))

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
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 4),
        )

    def forward(self, audio_tokens: Tensor, visual_tokens: Tensor, raw_images: Tensor) -> tuple[Tensor, Tensor]:
        if audio_tokens.ndim != 3 or audio_tokens.shape[1] != 6:
            raise ValueError("audio tokens must have shape [batch, 6, 512]")
        if visual_tokens.ndim != 3 or visual_tokens.shape[1] != 196:
            raise ValueError("visual tokens must have shape [batch, 196, 384]")

        enriched_queries = self.feature_enricher(visual_tokens, raw_images) + self.video_positions
        audio = self.audio_self_attention(self.audio_projection(audio_tokens) + self.audio_positions)

        attended, attention = self.cross_attention(enriched_queries, audio, audio, need_weights=True, average_attn_weights=False)
        fused = self.cross_norm(enriched_queries + attended)
        fused = self.ffn_norm(fused + self.ffn(fused))
        pooled = fused.mean(dim=1)
        return self.classifier(pooled), attention


class GeometryRippleMultimodalModel(nn.Module):
    """Method 3 (GW-AVF) Multimodal Model."""

    def __init__(self, audio_encoder: nn.Module, video_encoder: nn.Module, d_model: int, num_heads: int, encoder_mode: str, dropout: float) -> None:
        super().__init__()
        self.audio_encoder = audio_encoder
        self.video_encoder = video_encoder
        self.encoder_mode = encoder_mode
        self.fusion = GeometryRippleCrossAttentionHead(audio_encoder.feature_dim, video_encoder.feature_dim, d_model, num_heads, dropout)
        self.configure_encoder_mode()

    def configure_encoder_mode(self) -> None:
        for encoder in (self.audio_encoder, self.video_encoder):
            for parameter in encoder.parameters():
                parameter.requires_grad = False

    def forward(self, waveforms: Tensor, images: Tensor, return_attention: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        audio_tokens = self.audio_encoder(waveforms)
        visual_tokens = self.video_encoder(images)
        logits, attention = self.fusion(audio_tokens, visual_tokens, images)
        return (logits, attention) if return_attention else logits


class RobustBilinearMultimodalModel(nn.Module):
    """Method 4 (R-BPMD) Multimodal Model."""

    def __init__(self, audio_encoder: nn.Module, video_encoder: nn.Module, d_model: int, num_heads: int, encoder_mode: str, dropout: float, drop_prob: float = 0.15) -> None:
        super().__init__()
        self.audio_encoder = audio_encoder
        self.video_encoder = video_encoder
        self.encoder_mode = encoder_mode
        self.fusion = RobustBilinearCrossAttentionHead(
            audio_dim=audio_encoder.feature_dim,
            video_dim=video_encoder.feature_dim,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            drop_prob=drop_prob,
        )
        self.configure_encoder_mode()

    def configure_encoder_mode(self) -> None:
        for encoder in (self.audio_encoder, self.video_encoder):
            for parameter in encoder.parameters():
                parameter.requires_grad = False

    def forward(self, waveforms: Tensor, images: Tensor, return_attention: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        audio_tokens = self.audio_encoder(waveforms)
        visual_tokens = self.video_encoder(images)
        logits, attention = self.fusion(audio_tokens, visual_tokens)
        return (logits, attention) if return_attention else logits


class SwinSpatialMultimodal(nn.Module):
    """PANNS audio tokens fused with iBOT-adapted Swin stage-3 spatial tokens."""

    def __init__(self, audio_encoder: nn.Module, video_encoder: nn.Module, d_model: int, num_heads: int, encoder_mode: str, dropout: float) -> None:
        super().__init__()
        self.audio_encoder = audio_encoder
        self.video_encoder = video_encoder
        self.encoder_mode = encoder_mode
        self.fusion = SpatialCrossAttentionHead(audio_encoder.feature_dim, video_encoder.feature_dim, d_model, num_heads, dropout)
        self.configure_encoder_mode()

    def configure_encoder_mode(self) -> None:
        for encoder in (self.audio_encoder, self.video_encoder):
            for parameter in encoder.parameters():
                parameter.requires_grad = False

    def forward(self, waveforms: Tensor, images: Tensor, return_attention: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        logits, attention = self.fusion(self.audio_encoder(waveforms), self.video_encoder(images))
        return (logits, attention) if return_attention else logits


class StftPannsMobileNetAdvancedFusionHead(nn.Module):
    """Advanced Multimodal Fusion Head: Spatial Cross-Attention + Factorized Bilinear MFB + Modality Dropout."""

    def __init__(self, audio_dim: int = 512, video_dim: int = 1280, d_model: int = 256, num_heads: int = 4, dropout: float = 0.1, factor_k: int = 3, drop_prob: float = 0.15) -> None:
        super().__init__()
        self.d_model = d_model
        self.factor_k = factor_k
        self.drop_prob = drop_prob

        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.video_proj = nn.Linear(video_dim, d_model)
        self.stft_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, d_model),
        )
        self.audio_pos = nn.Parameter(Tensor(1, 6, d_model).normal_(mean=0.0, std=0.02))

        # Spatial Cross-Attention
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)

        # Factorized Bilinear MFB Projections
        self.mfb_audio_linear = nn.Linear(d_model, d_model * factor_k)
        self.mfb_video_linear = nn.Linear(d_model, d_model * factor_k)
        self.mfb_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 4),
        )

    def forward(self, audio_tokens: Tensor, video_feature: Tensor, stft_db_img: Tensor) -> tuple[Tensor, Tensor]:
        # Modality Dropout during training for robust sensor fault tolerance
        if self.training and self.drop_prob > 0.0:
            rand_val = torch.rand(1).item()
            if rand_val < self.drop_prob / 2.0:
                audio_tokens = torch.zeros_like(audio_tokens)
                stft_db_img = torch.zeros_like(stft_db_img)
            elif rand_val < self.drop_prob:
                video_feature = torch.zeros_like(video_feature)

        # 1. PANNS Audio Tokens: [B, 6, 512] -> [B, 6, d_model]
        panns_embed = self.audio_proj(audio_tokens) + self.audio_pos

        # 2. STFT dB Image Feature: [B, 3, 224, 224] -> [B, 1, d_model]
        stft_embed = self.stft_conv(stft_db_img).unsqueeze(1)

        # 3. Combined Audio Representation (6 PANNS tokens + 1 STFT token = 7 tokens)
        combined_audio = torch.cat([panns_embed, stft_embed], dim=1) # [B, 7, d_model]

        # 4. Video Query: MobileNetV2 [B, 1280] -> [B, 1, d_model]
        video_embed = self.video_proj(video_feature).unsqueeze(1)

        # 5. Cross Attention: Video Query attends to 7 Audio/STFT Tokens
        attended, attention = self.cross_attn(video_embed, combined_audio, combined_audio, need_weights=True)
        fused_video = self.cross_norm(video_embed + attended).squeeze(1) # [B, d_model]

        # 6. Audio Pooled Feature [B, d_model]
        audio_pooled = combined_audio.mean(dim=1)

        # 7. Factorized Bilinear MFB Hadamard Fusion
        aud_mfb = self.mfb_audio_linear(audio_pooled) # [B, d_model * factor_k]
        vid_mfb = self.mfb_video_linear(fused_video)   # [B, d_model * factor_k]
        mfb_prod = aud_mfb * vid_mfb                  # Hadamard product
        mfb_pooled = mfb_prod.reshape(-1, self.d_model, self.factor_k).sum(dim=2) # Sum pooling
        mfb_power = torch.sign(mfb_pooled) * torch.sqrt(torch.abs(mfb_pooled) + 1e-10) # Power norm
        mfb_out = self.mfb_norm(mfb_power)

        # 8. Concatenate Direct + Bilinear Fused Representations
        final_repr = torch.cat([fused_video, mfb_out], dim=1)
        logits = self.classifier(final_repr)
        return logits, attention


class StftPannsMobileNetAdvancedMultimodalModel(nn.Module):
    """Multimodal Model: Audio (PANNS CNN6 full fine-tune + STFT 256k) + Video (MobileNetV2) + Advanced MFB & Modality Dropout Fusion."""

    def __init__(self, audio_panns_encoder: nn.Module, video_mobilenet_encoder: nn.Module, d_model: int = 256, num_heads: int = 4, encoder_mode: str = "tune_audio_only", dropout: float = 0.1) -> None:
        super().__init__()
        self.stft_transform = STFTTodBImageTransform(n_fft=4096, hop_length=2048, win_length=2048, image_size=224)
        self.audio_panns_encoder = audio_panns_encoder
        self.video_mobilenet_encoder = video_mobilenet_encoder
        self.encoder_mode = encoder_mode
        self.fusion = StftPannsMobileNetAdvancedFusionHead(
            audio_dim=audio_panns_encoder.feature_dim,
            video_dim=video_mobilenet_encoder.feature_dim,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.configure_encoder_mode()

    def configure_encoder_mode(self) -> None:
        # Unfreeze ALL parameters of Audio Encoder (PANNS CNN6) for full fine-tuning
        for parameter in self.audio_panns_encoder.parameters():
            parameter.requires_grad = True

        # Freeze ALL parameters of Video Encoder (MobileNetV2) completely
        for parameter in self.video_mobilenet_encoder.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True) -> StftPannsMobileNetAdvancedMultimodalModel:
        super().train(mode)
        if mode:
            # Audio encoder in train mode for full fine-tuning
            self.audio_panns_encoder.train()
            # Video encoder strictly in eval mode
            self.video_mobilenet_encoder.eval()
        return self

    def forward(self, waveforms: Tensor, images: Tensor, return_attention: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        # 1. PANNS CNN6 Audio Tokens [B, 6, 512]
        audio_tokens = self.audio_panns_encoder(waveforms)

        # 2. STFT dB Spectrogram Image [B, 3, 224, 224]
        stft_db_img = self.stft_transform(waveforms)

        # 3. MobileNetV2 Video Feature [B, 1280]
        video_feature = self.video_mobilenet_encoder(images)

        # 4. Advanced Multimodal Fusion
        logits, attention = self.fusion(audio_tokens, video_feature, stft_db_img)
        return (logits, attention) if return_attention else logits
