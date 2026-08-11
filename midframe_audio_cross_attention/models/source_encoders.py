"""New token/feature adapters that execute the original baseline models."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.reference_bridge import load_audio_reference, load_video_reference
from settings import RunConfig


VIDEO_SPATIAL_CHANNELS = {"densenet121": 1024, "efficientnet_b0": 1280, "mobilenet_v2": 1280, "swin_tiny": 768}


def _load_strict(model: nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)


def _six_windows(waveforms: Tensor) -> Tensor:
    if waveforms.ndim != 2 or waveforms.shape[1] != 128000:
        raise ValueError(f"source audio contract requires [batch, 128000], got {tuple(waveforms.shape)}")
    return waveforms.unfold(1, size=48000, step=16000).contiguous()


class SourceAudioTokenEncoder(nn.Module):
    """Uses original AudioModel; a new hook exposes one 512d token per window."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.feature_dim = 512

    def forward(self, waveforms: Tensor) -> Tensor:
        windows = _six_windows(waveforms)
        batch_size, tokens, window_samples = windows.shape
        captured: list[Tensor] = []

        def save_pre_classifier(_module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
            captured.append(inputs[0])

        hook = self.model.backbone.fc_audioset.register_forward_pre_hook(save_pre_classifier)
        try:
            self.model(windows.reshape(batch_size * tokens, window_samples))
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError("Could not capture source PANNS feature before fc_audioset")
        return captured[0].reshape(batch_size, tokens, self.feature_dim)


class SourceVideoSpatialEncoder(nn.Module):
    """Uses original VideoModel; a new hook exposes its last spatial feature map."""

    def __init__(self, model: nn.Module, name: str, grid_size: int) -> None:
        super().__init__()
        self.model = model
        self.name = name
        self.feature_dim = VIDEO_SPATIAL_CHANNELS[name]
        self.grid_size = grid_size
        self.num_tokens = grid_size * grid_size

    def _spatial_features(self) -> nn.Module:
        return self.model.backbone.model.features

    def forward(self, images: Tensor) -> Tensor:
        captured: list[Tensor] = []

        def save_spatial_features(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
            captured.append(output)

        hook = self._spatial_features().register_forward_hook(save_spatial_features)
        try:
            self.model(images)
        finally:
            hook.remove()
        if len(captured) != 1 or captured[0].ndim != 4:
            raise RuntimeError(f"Could not capture a 4D spatial feature map from {self.name}")
        feature_map = captured[0]
        if self.name == "swin_tiny":
            feature_map = feature_map.permute(0, 3, 1, 2).contiguous()
        if feature_map.shape[1] != self.feature_dim:
            raise RuntimeError(f"Expected {self.feature_dim} spatial channels from {self.name}, got {feature_map.shape[1]}")
        pooled = F.adaptive_avg_pool2d(feature_map, (self.grid_size, self.grid_size))
        return pooled.flatten(2).transpose(1, 2).contiguous()


def build_source_video_encoder(config: RunConfig) -> SourceVideoSpatialEncoder:
    """Instantiate and strict-load the unchanged source video checkpoint."""
    video_reference = load_video_reference(config.references.video_repo)
    backbone = video_reference.video_backbones[config.model.video_backbone](classes_num=4, pretrained=False)
    video_model = video_reference.VideoModel(backbone=backbone)
    _load_strict(video_model, config.video_checkpoint)
    return SourceVideoSpatialEncoder(
        video_model,
        config.model.video_backbone,
        config.model.visual_grid_size,
    )
