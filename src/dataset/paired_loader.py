"""Pure, High-Performance Paired Dataset for STFT 256k Audio Waveforms and Video RGB Frames matching U_FFIA27K_video architecture."""

from __future__ import annotations

import csv
import logging
import os
import pickle
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
    """RAM-Optimized Paired Dataset with Fast PIL RAM Image Caching for Ultra-Fast Training."""

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
        self.video_cache: dict[int, Image.Image] = {}

        if self.cache_audio_enabled or self.cache_video_enabled:
            logger.info(f"Preloading '{split}' split dataset into RAM (audio_cache={self.cache_audio_enabled}, video_cache={self.cache_video_enabled})...")
            self._preload_ram_cache()

    def _preload_ram_cache(self) -> None:
        import sys

        total_samples = len(self.records)
        disable_progress = not sys.stdout.isatty()
        for idx in tqdm(range(total_samples), desc=f"Preloading '{self.split}' split into RAM", unit="sample", disable=disable_progress):
            rec = self.records[idx]
            if self.cache_audio_enabled and idx not in self.audio_cache:
                self.audio_cache[idx] = self._load_audio(rec["audio_path"])
            if self.cache_video_enabled and idx not in self.video_cache:
                self.video_cache[idx] = self._load_video_pil(rec["video_path"], idx)

            if disable_progress and (idx + 1) % 5000 == 0:
                logger.info(f"Preloading '{self.split}' split: {idx + 1}/{total_samples} samples cached into RAM...")

        if disable_progress:
            logger.info(f"Preloading '{self.split}' split complete! Total cached: {len(self.video_cache)} video frames.")

    def _load_audio(self, rel_path: str) -> Tensor:
        sample_id = Path(rel_path).stem
        npy_candidates = [
            Path("stft256k_features_npy") / f"{sample_id}.npy",
            Path("/marimo/Multimodal_Custom_Models/src/stft256k_features_npy") / f"{sample_id}.npy",
            Path.cwd() / "stft256k_features_npy" / f"{sample_id}.npy",
        ]
        for npy_file in npy_candidates:
            if npy_file.exists():
                try:
                    spec_np = np.load(str(npy_file)) # [2049, 250]
                    return torch.from_numpy(spec_np).unsqueeze(0).to(dtype=torch.float32)
                except Exception:
                    pass

        audio_file = resolve_dataset_file(self.dataset_base_dir, rel_path)
        try:
            waveform, sr = torchaudio.load(str(audio_file))
            if waveform.ndim > 1:
                waveform = waveform.mean(dim=0)
        except Exception:
            waveform = torch.zeros(self.target_samples, dtype=torch.float32)

        if waveform.numel() < self.target_samples:
            padding = self.target_samples - waveform.numel()
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform.numel() > self.target_samples:
            waveform = waveform[: self.target_samples]

        return waveform.to(dtype=torch.float32)

    def _load_video_pil(self, rel_path: str, index: int = 0) -> Image.Image:
        # 1. Check disk cache candidates from U_FFIA27K_video
        cache_candidates = [
            Path(f"/marimo/video_cache/single_frame_size_224/{self.split}/{index}.pkl"),
            Path(f"/marimo/video_cache/{self.split}/{index}.pkl"),
            Path.cwd() / "video_cache" / f"{self.split}_{index}.pkl",
        ]
        for cache_path in cache_candidates:
            if cache_path.exists():
                try:
                    with open(cache_path, "rb") as f:
                        sample = pickle.load(f)
                    image_form = sample.get("image_form")
                    if isinstance(image_form, np.ndarray) and image_form.dtype == np.uint8:
                        if image_form.shape[0] == 3:
                            image_form = image_form.transpose(1, 2, 0)
                        return Image.fromarray(image_form)
                except Exception:
                    pass

        # 2. Decode using decord or cv2
        video_file = resolve_dataset_file(self.dataset_base_dir, rel_path)
        try:
            if video_file.is_file() and video_file.suffix.lower() in (".png", ".jpg", ".jpeg"):
                return Image.open(video_file).convert("RGB")

            # Try decord VideoReader first (as in U_FFIA27K_video)
            try:
                from decord import VideoReader, cpu
                vr = VideoReader(str(video_file), width=self.config.data.image_size, height=self.config.data.image_size, ctx=cpu(0))
                if len(vr) > 0:
                    frame_index = len(vr) // 2
                    frame_rgb = vr.get_batch([frame_index]).asnumpy()[0]
                    return Image.fromarray(frame_rgb)
            except Exception:
                pass

            # Fallback to OpenCV cv2
            import cv2
            cap = cv2.VideoCapture(str(video_file))
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_idx = max(frame_count // 2, 0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    return Image.fromarray(frame_rgb)

            return Image.new("RGB", (self.config.data.image_size, self.config.data.image_size), color=(128, 128, 128))
        except Exception:
            return Image.new("RGB", (self.config.data.image_size, self.config.data.image_size), color=(128, 128, 128))

    def _load_video(self, rel_path: str, index: int = 0) -> Tensor:
        img_pil = self._load_video_pil(rel_path, index)
        return self.transform(img_pil)

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
            image = self.transform(self.video_cache[index])
        else:
            image = self._load_video(rec["video_path"], index)

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
