"""Dynamic Bridge module to load baseline U_FFIA27K_audio and U_FFIA27K_video repository dependencies."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class AudioReference:
    repo_path: Path
    FishVoiceDataLoader: type
    AudioFeaturesConfig: type
    AudioModel: type
    AudioClassifier: type
    audio_loader_module: ModuleType


@dataclass(frozen=True)
class VideoReference:
    repo_path: Path
    FishVideoDataLoader: type
    VideoModel: type
    video_loader_module: ModuleType


def _import_file(module_name: str, file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_audio_reference(repo_path: Path) -> AudioReference:
    resolved_repo = repo_path.resolve()
    if not (resolved_repo / "dataset" / "dataloader_melspectrogram.py").exists():
        candidates = [
            Path.cwd() / "U_FFIA27K_audio",
            Path.cwd().parent / "U_FFIA27K_audio",
            Path("/marimo/Multimodal_Custom_Models/U_FFIA27K_audio"),
        ]
        for cand in candidates:
            if cand.exists() and (cand / "dataset" / "dataloader_melspectrogram.py").exists():
                resolved_repo = cand.resolve()
                break

    sys.path.insert(0, str(resolved_repo))
    try:
        dataloader_mod = _import_file("audio_ref_dataloader", resolved_repo / "dataset" / "dataloader_melspectrogram.py")
        config_mod = _import_file("audio_ref_config", resolved_repo / "config" / "feature_config.py")
        model_mod = _import_file("audio_ref_model", resolved_repo / "models" / "audio_model.py")
    finally:
        sys.path.pop(0)

    return AudioReference(
        repo_path=resolved_repo,
        FishVoiceDataLoader=dataloader_mod.FishVoiceDataLoader,
        AudioFeaturesConfig=config_mod.AudioFeaturesConfig,
        AudioModel=model_mod.AudioModel,
        AudioClassifier=model_mod.AudioClassifier,
        audio_loader_module=dataloader_mod,
    )


def load_video_reference(repo_path: Path) -> VideoReference:
    resolved_repo = repo_path.resolve()
    if not (resolved_repo / "dataset" / "dataloader_videos.py").exists():
        candidates = [
            Path.cwd() / "U_FFIA27K_video",
            Path.cwd().parent / "U_FFIA27K_video",
            Path("/marimo/Multimodal_Custom_Models/U_FFIA27K_video"),
        ]
        for cand in candidates:
            if cand.exists() and (cand / "dataset" / "dataloader_videos.py").exists():
                resolved_repo = cand.resolve()
                break

    sys.path.insert(0, str(resolved_repo))
    try:
        dataloader_mod = _import_file("video_ref_dataloader", resolved_repo / "dataset" / "dataloader_videos.py")
        model_mod = _import_file("video_ref_model", resolved_repo / "models" / "video_model.py")
    finally:
        sys.path.pop(0)

    return VideoReference(
        repo_path=resolved_repo,
        FishVideoDataLoader=dataloader_mod.FishVideoDataLoader,
        VideoModel=model_mod.VideoModel,
        video_loader_module=dataloader_mod,
    )
