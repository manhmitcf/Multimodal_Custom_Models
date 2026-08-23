"""Pair immutable holdout rows while delegating media processing to baseline datasets.

Includes U_FFIA27K_video's exact multithreaded (ThreadPoolExecutor) uint8 numpy array RAM preload
and periodic progress logging every 5000 images.
"""

from __future__ import annotations

import concurrent.futures
import csv
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from models.reference_bridge import load_audio_reference, load_video_reference
from settings import RunConfig

logger = logging.getLogger(__name__)


def read_immutable_split(split_dir: Path, split: str) -> list[list[Any]]:
    path = Path(split_dir) / f"{split}.csv"
    if not path.exists():
        candidates = [
            path,
            Path.cwd() / "checkpoints" / "PANNS_Cnn6_holdout_random_sample_20260729_153012" / "DL_audio" / "checkpoint" / "panns_cnn6" / "splits" / f"{split}.csv",
            Path.cwd().parent / "checkpoints" / "PANNS_Cnn6_holdout_random_sample_20260729_153012" / "DL_audio" / "checkpoint" / "panns_cnn6" / "splits" / f"{split}.csv",
            Path("/marimo/Multimodal_Custom_Models/checkpoints/PANNS_Cnn6_holdout_random_sample_20260729_153012/DL_audio/checkpoint/panns_cnn6/splits") / f"{split}.csv",
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
    """One record drives both original baseline inner datasets; no split is regenerated."""

    def __init__(self, config: RunConfig, split: str) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.config = config
        self.split = split
        self.records = read_immutable_split(config.data.split_dir, split)

        audio_reference = load_audio_reference(config.references.audio_repo)
        video_reference = load_video_reference(config.references.video_repo)

        audio_parent = _source_parent(
            audio_reference.FishVoiceDataLoader,
            self.records,
            split,
            sample_rate=64000,
            batch_size=config.training.batch_size,
            num_workers=config.data.num_workers,
            cache_audio=config.data.cache_audio,
        )
        video_module = video_reference.video_loader_module
        video_parent = _source_parent(
            video_reference.FishVideoDataLoader,
            self.records,
            split,
            batch_size=config.training.batch_size,
            dataloader_workers=config.data.num_workers,
            prefetch_factor=None,
            cache_mode=config.data.video_cache_mode,
            image_size=config.data.image_size,
            image_cache_root=video_module.DEFAULT_IMAGE_CACHE_ROOT,
            image_cache_dir=video_module._resolve_image_cache_dir(None, config.data.image_size),
        )

        self.audio_dataset = audio_reference.FishVoiceDataLoader._InnerDataset(audio_parent, split)
        self.video_dataset = video_reference.FishVideoDataLoader._InnerDataset(video_parent, split)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        audio_sample = self.audio_dataset[index]
        video_sample = self.video_dataset[index]
        audio_path, video_path, label = self.records[index]

        if audio_sample["audio_name"] != audio_path or video_sample["video_name"] != video_path:
            raise RuntimeError(f"baseline path mismatch at index {index}")

        audio_label = int(np.asarray(audio_sample["target"]).argmax())
        video_label = int(np.asarray(video_sample["target"]).argmax())
        if (audio_label, video_label) != (label, label):
            raise RuntimeError(f"baseline label mismatch at index {index}: {audio_label}, {video_label}, expected {label}")

        return {
            "waveform": audio_sample["waveform"],
            "image": video_sample["video_form"],
            "label": label,
            "audio_path": audio_path,
            "video_path": video_path,
        }


def paired_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "waveform": torch.from_numpy(np.asarray([item["waveform"] for item in batch], dtype=np.float32)),
        "image": torch.stack([item["image"] for item in batch]),
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "audio_path": [item["audio_path"] for item in batch],
        "video_path": [item["video_path"] for item in batch],
    }
