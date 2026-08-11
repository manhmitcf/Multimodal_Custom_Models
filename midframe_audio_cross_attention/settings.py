"""Single JSON configuration for the source-integrated multimodal experiment."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_VALID_DINO_MODELS = {"dinov2_vits14_reg"}
_VALID_ENCODER_MODES = {"frozen", "tune"}
_VALID_DINO_ENCODER_MODES = {"frozen", "full"}
_VALID_CACHE_MODES = {"disk", "ram", "none"}
_VALID_POSITIONAL_ENCODINGS = {"none", "learned", "sinusoidal"}


def resolve_num_workers(value: int) -> int:
    """Mirror the source baseline's ``-1`` auto-worker convention exactly."""
    value = int(value)
    if value != -1:
        return value
    max_cpu = os.cpu_count()
    if max_cpu is None or max_cpu <= 0:
        return 0
    if max_cpu == 2:
        return max_cpu // 2
    return (max_cpu // 2) + 1


@dataclass(frozen=True)
class ReferenceConfig:
    audio_repo: Path
    video_repo: Path


@dataclass(frozen=True)
class DataConfig:
    split_dir: Path
    cache_audio: bool
    video_cache_mode: str
    num_workers: int
    image_size: int


@dataclass(frozen=True)
class ModelConfig:
    dino_model: str
    dino_encoder_mode: str
    audio_encoder_mode: str
    audio_positional_encoding: str
    d_model: int
    num_heads: int
    dropout: float


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    fusion_learning_rate: float
    encoder_learning_rate: float
    weight_decay: float
    seed: int
    device: str
    output_dir: Path


@dataclass(frozen=True)
class RunConfig:
    references: ReferenceConfig
    audio_checkpoint: Path
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig

    @classmethod
    def from_json(cls, path: Path | str) -> "RunConfig":
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        base = path.parent

        def resolve(value: str) -> Path:
            candidate = Path(value)
            return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()

        references = ReferenceConfig(
            audio_repo=resolve(raw["references"]["audio_repo"]),
            video_repo=resolve(raw["references"]["video_repo"]),
        )
        model = ModelConfig(**raw["model"])
        checkpoints = raw["checkpoints"]
        data = DataConfig(
            split_dir=resolve(raw["data"]["split_dir"]),
            cache_audio=bool(raw["data"]["cache_audio"]),
            video_cache_mode=str(raw["data"]["video_cache_mode"]),
            num_workers=resolve_num_workers(raw["data"]["num_workers"]),
            image_size=int(raw["data"]["image_size"]),
        )
        training_raw: dict[str, Any] = {**raw["training"], "output_dir": resolve(raw["training"]["output_dir"])}
        training = TrainingConfig(**training_raw)
        config = cls(
            references=references,
            audio_checkpoint=resolve(checkpoints["audio"]),
            data=data,
            model=model,
            training=training,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.model.dino_model not in _VALID_DINO_MODELS:
            raise ValueError(f"model.dino_model must be one of {sorted(_VALID_DINO_MODELS)}")
        if self.model.dino_encoder_mode not in _VALID_DINO_ENCODER_MODES:
            raise ValueError(f"model.dino_encoder_mode must be one of {sorted(_VALID_DINO_ENCODER_MODES)}")
        if self.model.audio_encoder_mode not in _VALID_ENCODER_MODES:
            raise ValueError(f"model.audio_encoder_mode must be one of {sorted(_VALID_ENCODER_MODES)}")
        if self.model.audio_positional_encoding not in _VALID_POSITIONAL_ENCODINGS:
            raise ValueError(f"model.audio_positional_encoding must be one of {sorted(_VALID_POSITIONAL_ENCODINGS)}")
        if self.data.video_cache_mode not in _VALID_CACHE_MODES:
            raise ValueError(f"data.video_cache_mode must be one of {sorted(_VALID_CACHE_MODES)}")
        if self.data.num_workers < 0:
            raise ValueError("data.num_workers must be -1 or a non-negative integer")
        if self.model.d_model <= 0 or self.model.d_model % self.model.num_heads:
            raise ValueError("model.d_model must be positive and divisible by model.num_heads")
        if self.model.audio_positional_encoding == "sinusoidal" and self.model.d_model % 2:
            raise ValueError("model.d_model must be even when audio_positional_encoding is sinusoidal")
        for path, description in (
            (self.references.audio_repo, "audio source repo"),
            (self.references.video_repo, "video source repo"),
            (self.audio_checkpoint, "audio checkpoint"),
            (self.data.split_dir, "split directory"),
        ):
            if not path.exists():
                raise FileNotFoundError(f"Missing {description}: {path}")
