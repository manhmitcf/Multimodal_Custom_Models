"""Adapters and loaders for pretrained baseline Video Encoders with Mechanism 1 SE-Recalibration."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from models.reference_bridge import load_video_reference
from settings import RunConfig

VIDEO_FEATURE_DIMS = {
    "densenet121": 1024,
    "efficientnet_b0": 1280,
    "mobilenet_v2": 1280,
    "swin_tiny": 768,
}


def _load_strict(model: nn.Module, checkpoint_path: Path) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)


class SERecalibration1D(nn.Module):
    """Squeeze-and-Excitation 1D Channel Recalibration (Mechanism 1).

    Models channel-wise feature dependencies for the visual vector, adaptively
    amplifying discriminative visual channels (whitewater splashes) and suppressing noise.
    """

    def __init__(self, in_features: int, reduction: int = 16) -> None:
        super().__init__()
        reduced_features = max(in_features // reduction, 32)
        self.fc = nn.Sequential(
            nn.Linear(in_features, reduced_features, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_features, in_features, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C]
        weights = self.fc(x) # [B, C] in [0, 1]
        return x * weights


class SourceVideoFeatureEncoder(nn.Module):
    """Uses original VideoModel; a hook exposes pre-classifier visual feature vector, enhanced with SE-Recalibration."""

    def __init__(self, model: nn.Module, name: str, use_se_recalibration: bool = True) -> None:
        super().__init__()
        self.model = model
        self.name = name
        self.feature_dim = VIDEO_FEATURE_DIMS[name]
        self.use_se_recalibration = use_se_recalibration
        if self.use_se_recalibration:
            self.se = SERecalibration1D(self.feature_dim, reduction=16)

    def _classifier(self) -> nn.Module:
        network = self.model.backbone.model
        if self.name == "densenet121":
            return network.classifier
        if self.name in {"efficientnet_b0", "mobilenet_v2"}:
            return network.classifier[1]
        return network.head

    def forward(self, images: Tensor) -> Tensor:
        captured: list[Tensor] = []

        def save_pre_classifier(_module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
            captured.append(inputs[0])

        hook = self._classifier().register_forward_pre_hook(save_pre_classifier)
        try:
            self.model(images)
        finally:
            hook.remove()

        if len(captured) != 1 or captured[0].shape[-1] != self.feature_dim:
            raise RuntimeError(f"Could not capture expected {self.feature_dim}d {self.name} feature")

        feat = captured[0] # [B, 1280]
        if self.use_se_recalibration:
            feat = self.se(feat) # Recalibrate channels adaptively
        return feat


def build_video_encoder(config: RunConfig) -> SourceVideoFeatureEncoder:
    """Build and load pretrained weights for the video encoder specified in config."""
    video_ref = load_video_reference(config.references.video_repo)
    backbone_name = config.model.video_backbone
    if backbone_name not in video_ref.video_backbones:
        raise ValueError(f"Unsupported video backbone: {backbone_name}")

    backbone_cls = video_ref.video_backbones[backbone_name]
    backbone_instance = backbone_cls(classes_num=4, pretrained=False)
    video_model = video_ref.VideoModel(backbone=backbone_instance)

    if config.video_checkpoint.exists():
        _load_strict(video_model, config.video_checkpoint)

    use_se = config.model.video_se_recalibration
    return SourceVideoFeatureEncoder(video_model, backbone_name, use_se_recalibration=use_se)
