"""Configuration loading and validation routines for STFT 256k Raw 2049 Bins Multimodal Experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class STFTConfig:
    sr: int = 256000
    pre_emphasis: float = 0.97
    frame_length: int = 4096
    hop_length: int = 2048
    n_fft: int = 4096
    windowing: str = "hamming"
    use_std: bool = True


@dataclass(frozen=True)
class ReferenceConfig:
    audio_repo: Path
    video_repo: Path


@dataclass(frozen=True)
class DataConfig:
    split_dir: Path
    cache_audio: bool = True
    video_cache_mode: str = "ram"
    num_workers: int = -1
    image_size: int = 224


@dataclass(frozen=True)
class ModelConfig:
    video_backbone: str = "mobilenet_v2"
    encoder_mode: str = "tune"
    d_model: int = 256
    dropout: float = 0.1
    fusion_type: str = "fbgf"


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 100
    fusion_learning_rate: float = 0.001
    encoder_learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    seed: int = 42
    device: str = "auto"
    output_dir: Path = field(default_factory=lambda: Path("checkpoint"))


@dataclass(frozen=True)
class ResultsUploadConfig:
    enabled: bool = True
    repo_id: str = "manhmitcf/fish_result"
    repo_type: str = "dataset"
    path_prefix: str = "stft256k_raw2049_freqattn_mobilenetv2_fbgf"
    create_repo: bool = True


@dataclass(frozen=True)
class RunConfig:
    stft: STFTConfig
    references: ReferenceConfig
    audio_checkpoint: Path
    video_checkpoint: Path
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    results_upload: ResultsUploadConfig

    @classmethod
    def from_json(cls, path: Path | str) -> RunConfig:
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        base = path.parent.parent if path.parent.name == "config" else path.parent

        def resolve(relative: str) -> Path:
            resolved = (base / relative).resolve()
            if resolved.exists():
                return resolved
            cand1 = (Path.cwd() / relative).resolve()
            if cand1.exists():
                return cand1
            cand2 = (Path(__file__).resolve().parent.parent / relative.lstrip("../")).resolve()
            if cand2.exists():
                return cand2
            return resolved

        stft_data = payload.get("stft", {})
        stft = STFTConfig(
            sr=int(stft_data.get("sr", 256000)),
            pre_emphasis=float(stft_data.get("pre_emphasis", 0.97)),
            frame_length=int(stft_data.get("frame_length", 4096)),
            hop_length=int(stft_data.get("hop_length", 2048)),
            n_fft=int(stft_data.get("n_fft", 4096)),
            windowing=str(stft_data.get("windowing", "hamming")),
            use_std=bool(stft_data.get("use_std", True)),
        )

        references = ReferenceConfig(
            audio_repo=resolve(payload["references"]["audio_repo"]),
            video_repo=resolve(payload["references"]["video_repo"]),
        )

        checkpoints = payload["checkpoints"]
        audio_checkpoint = resolve(checkpoints["audio"])
        video_by_backbone = checkpoints.get("video_by_backbone", {})
        video_checkpoint_value = video_by_backbone.get(payload["model"].get("video_backbone"), checkpoints["audio"])
        video_checkpoint = resolve(video_checkpoint_value)

        data = DataConfig(
            split_dir=resolve(payload["data"]["split_dir"]),
            cache_audio=payload["data"].get("cache_audio", True),
            video_cache_mode=payload["data"].get("video_cache_mode", "ram"),
            num_workers=payload["data"].get("num_workers", -1),
            image_size=payload["data"].get("image_size", 224),
        )

        model = ModelConfig(
            video_backbone=payload["model"].get("video_backbone", "mobilenet_v2"),
            encoder_mode=payload["model"].get("encoder_mode", "tune"),
            d_model=payload["model"].get("d_model", 256),
            dropout=payload["model"].get("dropout", 0.1),
            fusion_type=payload["model"].get("fusion_type", "fbgf"),
        )

        training = TrainingConfig(
            batch_size=payload["training"].get("batch_size", 16),
            epochs=payload["training"].get("epochs", 100),
            fusion_learning_rate=payload["training"].get("fusion_learning_rate", 0.001),
            encoder_learning_rate=payload["training"].get("encoder_learning_rate", 0.0001),
            weight_decay=payload["training"].get("weight_decay", 0.0001),
            seed=payload["training"].get("seed", 42),
            device=payload["training"].get("device", "auto"),
            output_dir=resolve(payload["training"].get("output_dir", "checkpoint")),
        )

        results_upload = ResultsUploadConfig(
            enabled=payload["results_upload"].get("enabled", True),
            repo_id=payload["results_upload"].get("repo_id", "manhmitcf/fish_result"),
            repo_type=payload["results_upload"].get("repo_type", "dataset"),
            path_prefix=payload["results_upload"].get("path_prefix", "stft256k_raw2049_freqattn_mobilenetv2_fbgf"),
            create_repo=payload["results_upload"].get("create_repo", True),
        )

        return cls(
            stft=stft,
            references=references,
            audio_checkpoint=audio_checkpoint,
            video_checkpoint=video_checkpoint,
            data=data,
            model=model,
            training=training,
            results_upload=results_upload,
        )
