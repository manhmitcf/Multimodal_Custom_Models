"""Settings and DataClasses for Spatial Cross-Attention Multimodal Model Configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReferenceConfig:
    audio_repo: Path
    video_repo: Path


@dataclass(frozen=True)
class DataConfig:
    split_dir: Path
    cache_audio: bool = False
    video_cache_mode: str = "ram"
    num_workers: int = -1
    image_size: int = 224


@dataclass(frozen=True)
class ModelConfig:
    video_backbone: str = "mobilenet_v2"
    encoder_mode: str = "frozen"
    d_model: int = 256
    num_heads: int = 4
    dropout: float = 0.1


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 100
    fusion_learning_rate: float = 0.001
    encoder_learning_rate: float = 0.001
    weight_decay: float = 0.0001
    seed: int = 42
    device: str = "auto"
    output_dir: str = "checkpoint"


@dataclass(frozen=True)
class ResultsUploadConfig:
    enabled: bool = True
    repo_id: str = "manhmitcf/fish_result"
    repo_type: str = "dataset"
    path_prefix: str = "midframe_audio_cross_attention"
    create_repo: bool = True


@dataclass(frozen=True)
class RunConfig:
    references: ReferenceConfig
    audio_checkpoint: Path
    video_checkpoint: Path
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    results_upload: ResultsUploadConfig

    @classmethod
    def from_json(cls, json_path: Path) -> RunConfig:
        with open(json_path, encoding="utf-8") as f:
            payload = json.load(f)

        base_dir = json_path.parent

        def resolve(path_str: str) -> Path:
            p = Path(path_str)
            if p.is_absolute():
                return p
            return (base_dir / p).resolve()

        references = ReferenceConfig(
            audio_repo=resolve(payload["references"]["audio_repo"]),
            video_repo=resolve(payload["references"]["video_repo"]),
        )

        audio_ckpt = resolve(payload["checkpoints"]["audio"])

        backbone = payload["model"].get("video_backbone", "mobilenet_v2")
        video_map = payload["checkpoints"].get("video_by_backbone", {})
        if backbone in video_map:
            video_ckpt = resolve(video_map[backbone])
        else:
            video_ckpt = resolve(payload["checkpoints"].get("video", ""))

        data_payload = payload.get("data", {})
        data = DataConfig(
            split_dir=resolve(data_payload.get("split_dir", "../splits")),
            cache_audio=bool(data_payload.get("cache_audio", False)),
            video_cache_mode=str(data_payload.get("video_cache_mode", "ram")),
            num_workers=int(data_payload.get("num_workers", -1)),
            image_size=int(data_payload.get("image_size", 224)),
        )

        model_payload = payload.get("model", {})
        model = ModelConfig(
            video_backbone=str(model_payload.get("video_backbone", "mobilenet_v2")),
            encoder_mode=str(model_payload.get("encoder_mode", "frozen")),
            d_model=int(model_payload.get("d_model", 256)),
            num_heads=int(model_payload.get("num_heads", 4)),
            dropout=float(model_payload.get("dropout", 0.1)),
        )

        train_payload = payload.get("training", {})
        training = TrainingConfig(
            batch_size=int(train_payload.get("batch_size", 64)),
            epochs=int(train_payload.get("epochs", 100)),
            fusion_learning_rate=float(train_payload.get("fusion_learning_rate", 0.001)),
            encoder_learning_rate=float(train_payload.get("encoder_learning_rate", 0.001)),
            weight_decay=float(train_payload.get("weight_decay", 0.0001)),
            seed=int(train_payload.get("seed", 42)),
            device=str(train_payload.get("device", "auto")),
            output_dir=str(train_payload.get("output_dir", "checkpoint")),
        )

        upload_payload = payload.get("results_upload", {})
        results_upload = ResultsUploadConfig(
            enabled=bool(upload_payload.get("enabled", True)),
            repo_id=str(upload_payload.get("repo_id", "manhmitcf/fish_result")),
            repo_type=str(upload_payload.get("repo_type", "dataset")),
            path_prefix=str(upload_payload.get("path_prefix", "midframe_audio_cross_attention")),
            create_repo=bool(upload_payload.get("create_repo", True)),
        )

        return cls(
            references=references,
            audio_checkpoint=audio_ckpt,
            video_checkpoint=video_ckpt,
            data=data,
            model=model,
            training=training,
            results_upload=results_upload,
        )
