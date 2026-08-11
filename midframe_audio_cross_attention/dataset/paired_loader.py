"""Pair immutable holdout rows while delegating media processing to baseline datasets."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from models.reference_bridge import load_video_reference
from settings import RunConfig


def read_immutable_split(split_dir: Path, split: str) -> list[list[Any]]:
    path = Path(split_dir) / f"{split}.csv"
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
    """Pair cached STFT tokens with the unchanged source video dataset."""

    def __init__(self, config: RunConfig, split: str, stft_cache_dir: Path) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.records = read_immutable_split(config.data.split_dir, split)
        video_reference = load_video_reference(config.references.video_repo)
        self.stft_cache_dir = Path(stft_cache_dir) / split
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
        self.video_dataset = video_reference.FishVideoDataLoader._InnerDataset(video_parent, split)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        video_sample = self.video_dataset[index]
        audio_path, video_path, label = self.records[index]
        if video_sample["video_name"] != video_path:
            raise RuntimeError(f"baseline path mismatch at index {index}")
        video_label = int(np.asarray(video_sample["target"]).argmax())
        if video_label != label:
            raise RuntimeError(f"baseline label mismatch at index {index}: {video_label}, expected {label}")
        return {
            "audio_features": np.load(self.stft_cache_dir / f"{index}.npy"),
            "image": video_sample["video_form"],
            "label": label,
            "audio_path": audio_path,
            "video_path": video_path,
        }


def paired_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "audio_features": torch.from_numpy(np.asarray([item["audio_features"] for item in batch], dtype=np.float32)),
        "image": torch.stack([item["image"] for item in batch]),
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "audio_path": [item["audio_path"] for item in batch],
        "video_path": [item["video_path"] for item in batch],
    }
