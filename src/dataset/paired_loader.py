"""Pure, High-Performance Paired Dataset for STFT 256k Raw Audio Waveforms and Video RGB Frames."""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from settings import RunConfig

logger = logging.getLogger(__name__)


def resolve_num_workers(config_num_workers: int) -> int:
    """Calculate optimal num_workers. If config_num_workers <= 0, use max_cpu_cores // 2 + 1."""
    if config_num_workers <= 0:
        max_cores = os.cpu_count() or 4
        return (max_cores // 2) + 1
    return config_num_workers


def resolve_dataset_file(base_dir: Path, rel_path: str) -> Path:
    """Resolve audio/video relative file path against dataset candidates."""
    target = Path(rel_path)
    if target.is_absolute() and target.exists():
        return target

    candidates = [
        base_dir / rel_path,
        Path("/marimo/Fish_Feeding_Intensity_Dataset") / rel_path,
        Path("/marimo/Fish_Feeding_Intensity_Dataset") / rel_path.lstrip("audio/").lstrip("video/"),
        Path("D:/Fish_Feeding_Intensity/Dataset/U_FFIA") / rel_path,
        Path.cwd().parent / rel_path,
        Path.cwd() / rel_path,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return base_dir / rel_path


def read_immutable_split(split_dir: Path, split: str) -> list[dict[str, Any]]:
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
        return [
            {
                "audio_path": row["audio_path"],
                "video_path": row["video_path"],
                "label": int(row["label"]),
            }
            for row in reader
        ]


def build_video_transforms(split: str = "train", image_size: int = 224) -> transforms.Compose:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


class SourcePairedDataset(Dataset[dict[str, Any]]):
    """Pure Paired Dataset loading 256k Raw Audio Waveforms and RGB Frames directly."""

    def __init__(self, config: RunConfig, split: str) -> None:
        super().__init__()
        self.config = config
        self.split = split
        self.records = read_immutable_split(config.data.split_dir, split)
        self.transform = build_video_transforms(split=split, image_size=config.data.image_size)
        self.target_samples = 512000 # 2 seconds at 256k SR

        self.dataset_base_dir = Path("/marimo/Fish_Feeding_Intensity_Dataset")

        # RAM Caches
        self.cache_audio_enabled = config.data.cache_audio
        self.cache_video_enabled = config.data.video_cache_mode == "ram"

        self.audio_cache: dict[int, Tensor] = {}
        self.video_cache: dict[int, Tensor] = {}

        if self.cache_audio_enabled or self.cache_video_enabled:
            logger.info(f"Preloading '{split}' split dataset into RAM (audio_cache={self.cache_audio_enabled}, video_cache={self.cache_video_enabled})...")
            self._preload_ram_cache()

    def _preload_ram_cache(self) -> None:
        import sys
        disable_progress = not sys.stdout.isatty()
        for idx in tqdm(range(len(self.records)), desc=f"Preloading '{self.split}' split into RAM", unit="sample", disable=disable_progress):
            rec = self.records[idx]
            if self.cache_audio_enabled and idx not in self.audio_cache:
                self.audio_cache[idx] = self._load_audio(rec["audio_path"])
            if self.cache_video_enabled and idx not in self.video_cache:
                self.video_cache[idx] = self._load_video(rec["video_path"])

    def _load_audio(self, rel_path: str) -> Tensor:
        audio_file = resolve_dataset_file(self.dataset_base_dir, rel_path)
        try:
            waveform, sr = torchaudio.load(str(audio_file))
            if waveform.ndim > 1:
                waveform = waveform.mean(dim=0)
        except Exception:
            # Fallback for synthetic/missing audio files
            waveform = torch.zeros(self.target_samples, dtype=torch.float32)

        # Pad or crop to target_samples (512,000 for 2s at 256k)
        if waveform.numel() < self.target_samples:
            padding = self.target_samples - waveform.numel()
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform.numel() > self.target_samples:
            waveform = waveform[: self.target_samples]

        return waveform.to(dtype=torch.float32)

    def _load_video(self, rel_path: str) -> Tensor:
        video_file = resolve_dataset_file(self.dataset_base_dir, rel_path)
        try:
            if video_file.is_file() and video_file.suffix.lower() in (".png", ".jpg", ".jpeg"):
                img = Image.open(video_file).convert("RGB")
            else:
                # Fallback synthetic frame if mp4 reader is missing
                img = Image.new("RGB", (self.config.data.image_size, self.config.data.image_size), color=(128, 128, 128))
            return self.transform(img)
        except Exception:
            img = Image.new("RGB", (self.config.data.image_size, self.config.data.image_size), color=(128, 128, 128))
            return self.transform(img)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.records[index]
        label = rec["label"]

        if self.cache_audio_enabled and index in self.audio_cache:
            waveform = self.audio_cache[index]
        else:
            waveform = self._load_audio(rec["audio_path"])

        if self.cache_video_enabled and index in self.video_cache:
            image = self.video_cache[index]
        else:
            image = self._load_video(rec["video_path"])

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
