"""Pair immutable holdout rows while delegating media processing to baseline datasets."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from models.reference_bridge import load_audio_reference, load_video_reference
from settings import RunConfig


def resolve_num_workers(config_num_workers: int) -> int:
    """Calculate optimal num_workers. If config_num_workers <= 0, use max_cpu_cores // 2 + 1."""
    if config_num_workers <= 0:
        max_cores = os.cpu_count() or 4
        return (max_cores // 2) + 1
    return config_num_workers


def read_immutable_split(split_dir: Path, split: str) -> list[list[Any]]:
    path = Path(split_dir) / f"{split}.csv"
    if not path.exists():
        candidates = [
            path,
            Path.cwd() / "checkpoints" / "PANNS_Cnn6_holdout_random_sample_20260729_153012" / "DL_audio" / "checkpoint" / "panns_cnn6" / "splits" / f"{split}.csv",
            Path.cwd().parent / "checkpoints" / "PANNS_Cnn6_holdout_random_sample_20260729_153012" / "DL_audio" / "checkpoint" / "panns_cnn6" / "splits" / f"{split}.csv",
            Path(__file__).resolve().parent.parent.parent / "checkpoints" / "PANNS_Cnn6_holdout_random_sample_20260729_153012" / "DL_audio" / "checkpoint" / "panns_cnn6" / "splits" / f"{split}.csv",
            Path(__file__).resolve().parent.parent / "checkpoints" / "PANNS_Cnn6_holdout_random_sample_20260729_153012" / "DL_audio" / "checkpoint" / "panns_cnn6" / "splits" / f"{split}.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"audio_path", "video_path", "label"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"{path} must contain {sorted(required)}")
        return [[row["audio_path"], row["video_path"], int(row["label"])] for row in reader]


def _source_parent(loader_class: type, records: list[list[Any]], split: str, **attributes: Any) -> Any:
    """Build only the source loader state needed by its original InnerDataset."""
    parent = object.__new__(loader_class)
    for name, value in attributes.items():
        setattr(parent, name, value)
    parent.train_dict = records if split == "train" else []
    parent.val_dict = records if split == "val" else []
    parent.test_dict = records if split == "test" else []
    return parent


class SourcePairedDataset(Dataset[dict[str, Any]]):
    def __init__(self, config: RunConfig, split: str) -> None:
        super().__init__()
        self.config = config
        self.split = split
        self.records = read_immutable_split(config.data.split_dir, split)

        audio_ref = load_audio_reference(config.references.audio_repo)
        video_ref = load_video_reference(config.references.video_repo)

        frontend_config = audio_ref.AudioFeaturesConfig(
            sample_rate=64000,
            window_size=2048,
            hop_size=1024,
            mel_bins=128,
            fmin=1,
            fmax=32000,
            time_drop_width=16,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2,
        )

        audio_parent = _source_parent(
            audio_ref.FishAudioDataLoader,
            self.records,
            split,
            dataset_dir=Path(config.references.audio_repo).resolve().parent,
            cache_audio=config.data.cache_audio,
            num_workers=config.data.num_workers,
            config=frontend_config,
        )
        audio_parent.inner_dataset = audio_ref.FishAudioDataLoader.InnerDataset(
            audio_parent,
            split=split,
            transform=None,
        )
        self.audio_inner = audio_parent.inner_dataset

        video_parent = _source_parent(
            video_ref.FishVideoDataLoader,
            self.records,
            split,
            dataset_dir=Path(config.references.video_repo).resolve().parent,
            video_cache_mode=config.data.video_cache_mode,
            num_workers=config.data.num_workers,
            image_size=config.data.image_size,
        )

        transform = video_ref.build_transforms(split=split, image_size=config.data.image_size)
        video_parent.inner_dataset = video_ref.FishVideoDataLoader.InnerDataset(
            video_parent,
            split=split,
            transform=transform,
        )
        self.video_inner = video_parent.inner_dataset

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        audio_data = self.audio_inner[index]
        video_data = self.video_inner[index]
        label = int(audio_data["label"])

        waveform = audio_data["waveform"]
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.from_numpy(np.asarray(waveform, dtype=np.float32))

        image = video_data["image"]
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.asarray(image, dtype=np.float32))

        return {
            "waveform": waveform,
            "image": image,
            "label": label,
        }


def paired_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    waveforms = torch.stack([sample["waveform"] for sample in batch], dim=0)
    images = torch.stack([sample["image"] for sample in batch], dim=0)
    labels = torch.tensor([sample["label"] for sample in batch], dtype=torch.long)
    return {
        "waveform": waveforms,
        "image": images,
        "label": labels,
    }
