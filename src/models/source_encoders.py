"""New token/feature adapters that execute the original baseline models."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from models.reference_bridge import load_audio_reference, load_video_reference
from settings import RunConfig


VIDEO_FEATURE_DIMS = {"densenet121": 1024, "efficientnet_b0": 1280, "mobilenet_v2": 1280, "swin_tiny": 768}


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


class SourceVideoFeatureEncoder(nn.Module):
    """Uses original VideoModel; a new hook exposes the pre-classifier visual feature."""

    def __init__(self, model: nn.Module, name: str) -> None:
        super().__init__()
        self.model = model
        self.name = name
        self.feature_dim = VIDEO_FEATURE_DIMS[name]

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
        return captured[0]


class SourceSwinSpatialEncoder(nn.Module):
    """Expose SwinTiny stage-3 tokens while retaining source checkpoint weights.

    Torchvision SwinTiny uses channels-last feature maps.  At 224x224,
    stage 3 is a 14x14 grid with 384 channels; this is the representation
    used both by iBOT-inspired pretraining and spatial multimodal fusion.
    """

    stage3_feature_index = 5
    stage3_grid_size = 14
    stage3_feature_dim = 384
    input_grid_size = 56
    input_feature_dim = 96

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.name = "swin_tiny"
        self.feature_dim = self.stage3_feature_dim
        self.num_tokens = self.stage3_grid_size * self.stage3_grid_size
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.input_feature_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    @property
    def network(self) -> nn.Module:
        return self.model.backbone.model

    def _expand_stage3_mask(self, stage3_mask: Tensor) -> Tensor:
        if stage3_mask.shape != (stage3_mask.shape[0], self.num_tokens):
            raise ValueError(
                f"stage-3 mask must have shape [batch, {self.num_tokens}], got {tuple(stage3_mask.shape)}"
            )
        grid = stage3_mask.reshape(-1, self.stage3_grid_size, self.stage3_grid_size)
        return grid.repeat_interleave(4, dim=1).repeat_interleave(4, dim=2)

    def forward_features(self, images: Tensor, stage3_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        features = self.network.features
        x = features[0](images)
        if x.ndim != 4 or x.shape[-1] != self.input_feature_dim:
            raise RuntimeError(f"Unexpected Swin input feature shape: {tuple(x.shape)}")
        if stage3_mask is not None:
            expanded_mask = self._expand_stage3_mask(stage3_mask).unsqueeze(-1)
            if expanded_mask.shape[:3] != x.shape[:3]:
                raise RuntimeError(f"Swin mask shape {tuple(expanded_mask.shape)} does not match features {tuple(x.shape)}")
            x = torch.where(expanded_mask, self.mask_token.to(dtype=x.dtype), x)
        stage3 = None
        for index in range(1, len(features)):
            x = features[index](x)
            if index == self.stage3_feature_index:
                stage3 = x
        if stage3 is None or stage3.shape[1:] != (
            self.stage3_grid_size,
            self.stage3_grid_size,
            self.stage3_feature_dim,
        ):
            actual = None if stage3 is None else tuple(stage3.shape)
            raise RuntimeError(f"Unexpected Swin stage-3 feature shape: {actual}")
        stage4 = self.network.norm(x)
        global_feature = stage4.mean(dim=(1, 2))
        return stage3, global_feature

    def forward(self, images: Tensor) -> Tensor:
        stage3, _ = self.forward_features(images)
        return stage3.reshape(stage3.shape[0], self.num_tokens, self.feature_dim).contiguous()


def build_source_encoders(config: RunConfig) -> tuple[SourceAudioTokenEncoder, SourceSwinSpatialEncoder]:
    """Instantiate and strict-load only the original source model classes."""
    audio_reference = load_audio_reference(config.references.audio_repo)
    frontend_config = audio_reference.AudioFeaturesConfig(
        sample_rate=64000,
        window_size=2048,
        hop_size=1024,
        mel_bins=128,
        fmin=1,
        fmax=32000,
        time_drop_width=64,
        time_stripes_num=2,
        freq_drop_width=8,
        freq_stripes_num=2,
    )
    audio_model = audio_reference.AudioModel(
        frontend=audio_reference.AudioFrontend(config=frontend_config),
        backbone=audio_reference.PANNS_Cnn6(classes_num=4),
    )
    _load_strict(audio_model, config.audio_checkpoint)

    video_reference = load_video_reference(config.references.video_repo)
    if config.model.video_backbone != "swin_tiny":
        raise ValueError("The Swin iBOT spatial pipeline requires model.video_backbone='swin_tiny'.")
    backbone = video_reference.video_backbones[config.model.video_backbone](classes_num=4, pretrained=False)
    video_model = video_reference.VideoModel(backbone=backbone)
    _load_strict(video_model, config.video_checkpoint)
    return SourceAudioTokenEncoder(audio_model), SourceSwinSpatialEncoder(video_model)
