"""Load the two baseline repositories without letting their top-level names collide."""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterator


_LEGACY_TOP_LEVEL = ("config", "dataset", "features", "models", "tasks", "transforms", "utils")


@contextmanager
def _legacy_import_context(repository_root: Path) -> Iterator[None]:
    """Temporarily reserve legacy absolute names for one read-only source repo."""
    repository_root = Path(repository_root).resolve()
    if not repository_root.is_dir():
        candidates = [
            repository_root,
            Path.cwd() / repository_root.name,
            Path.cwd().parent / repository_root.name,
            Path(__file__).resolve().parent.parent / repository_root.name,
            Path(__file__).resolve().parent.parent.parent / repository_root.name,
        ]
        found = False
        for candidate in candidates:
            if candidate.is_dir():
                repository_root = candidate.resolve()
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Baseline repository not found: {repository_root}")

    previous_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name.split(".", 1)[0] in _LEGACY_TOP_LEVEL
    }
    for name in previous_modules:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(repository_root))
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name.split(".", 1)[0] in _LEGACY_TOP_LEVEL:
                sys.modules.pop(name, None)
        sys.modules.update(previous_modules)
        sys.path.remove(str(repository_root))


@dataclass(frozen=True)
class AudioReference:
    AudioFeaturesConfig: type
    AudioFrontend: type
    AudioModel: type
    PANNS_Cnn6: type
    FishVoiceDataLoader: type


@dataclass(frozen=True)
class VideoReference:
    VideoModel: type
    FishVideoDataLoader: type
    video_loader_module: ModuleType
    video_backbones: dict[str, type]


def load_audio_reference(repository_root: Path | str) -> AudioReference:
    """Return the original audio classes while leaving no legacy module alias behind."""
    with _legacy_import_context(Path(repository_root)):
        config = importlib.import_module("config")
        features = importlib.import_module("features")
        models = importlib.import_module("models")
        loader = importlib.import_module("dataset.dataloader_melspectrogram")
        return AudioReference(
            AudioFeaturesConfig=config.AudioFeaturesConfig,
            AudioFrontend=features.AudioFrontend,
            AudioModel=models.AudioModel,
            PANNS_Cnn6=models.PANNS_Cnn6,
            FishVoiceDataLoader=loader.FishVoiceDataLoader,
        )


def load_video_reference(repository_root: Path | str) -> VideoReference:
    """Return the original video classes while leaving no legacy module alias behind."""
    with _legacy_import_context(Path(repository_root)):
        models = importlib.import_module("models")
        loader = importlib.import_module("dataset.dataloader_videos")
        return VideoReference(
            VideoModel=models.VideoModel,
            FishVideoDataLoader=loader.FishVideoDataLoader,
            video_loader_module=loader,
            video_backbones={
                "densenet121": models.DenseNet121,
                "efficientnet_b0": models.EfficientNetB0,
                "mobilenet_v2": models.MobileNetV2,
                "swin_tiny": models.SwinTiny,
            },
        )
