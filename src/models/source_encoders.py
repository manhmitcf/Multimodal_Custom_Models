"""Token/Feature adapters that execute original PANNS CNN6 and MobileNetV2 baseline encoders."""

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
    """Uses original AudioModel; a forward hook exposes six 512d tokens per waveform."""

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
    """Uses original VideoModel; a forward hook exposes the 1280d pre-classifier visual feature vector."""

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


def build_source_encoders(config: RunConfig) -> tuple[SourceAudioTokenEncoder, SourceVideoFeatureEncoder]:
    """Instantiate and strict-load original source PANNS CNN6 and MobileNetV2 models."""
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
    raw_audio_model = audio_reference.AudioModel(
        model_name="panns_cnn6",
        features_config=frontend_config,
        num_classes=4,
        drop_rate=config.model.dropout,
        global_pool="max",
    )
    _load_strict(raw_audio_model, config.audio_checkpoint)

    video_reference = load_video_reference(config.references.video_repo)
    raw_video_model = video_reference.VideoModel(
        backbone_name=config.model.video_backbone,
        num_classes=4,
        drop_rate=config.model.dropout,
    )
    _load_strict(raw_video_model, config.video_checkpoint)

    return SourceAudioTokenEncoder(raw_audio_model), SourceVideoFeatureEncoder(raw_video_model, config.model.video_backbone)
