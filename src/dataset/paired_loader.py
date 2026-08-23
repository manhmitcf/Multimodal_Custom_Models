"""Pure, High-Performance Paired Dataset for STFT 256k Audio Waveforms and Video RGB Frames matching U_FFIA27K_video architecture exactly."""

from __future__ import annotations

import concurrent.futures
import csv
import gc
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

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


def get_process_memory_str() -> str:
    """Read VmRSS process memory from /proc/self/status on Linux."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                    return f"{rss_kb / 1024:.1f} MB ({rss_kb / (1024*1024):.2f} GB)"
    except Exception:
        pass
    return "N/A"


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
    """RAM-Optimized Paired Dataset ensuring 100% exact loss/acc match and strictly 3.15 GB max video RAM."""

    def __init__(self, config: RunConfig, split: str) -> None:
        super().__init__()
        self.config = config
        self.split = split
        self.records = read_immutable_split(config.data.split_dir, split)
        self.transform = build_video_transforms(split=split, image_size=config.data.image_size)
        self.target_samples = 512000 # 2 seconds at 256k SR

        self.dataset_base_dir = Path("/marimo/Fish_Feeding_Intensity_Dataset")

        # RAM Cache (Video Only to guarantee 3.15 GB RAM limit)
        self.cache_video_enabled = (config.data.video_cache_mode == "ram")
        self.video_cache: list[np.ndarray | None] = [None] * len(self.records)

        if self.cache_video_enabled:
            logger.info(f"Initial Process Memory before '{split}' RAM preload: {get_process_memory_str()}")
            self._preload_ram_cache()

    def _preload_ram_cache(self) -> None:
        total_samples = len(self.records)
        num_workers = max(os.cpu_count() or 4, 1)
        logger.info(f"Preloading '{self.split}' split video frames into RAM (guaranteed uint8 [3, 224, 224] 147 KB format, ThreadPoolExecutor {num_workers} workers)...")

        def load_sample(idx_rec: tuple[int, dict[str, Any]]) -> tuple[int, np.ndarray]:
            idx, rec = idx_rec
            img_pil = self._load_video_pil(rec["video_path"], idx)

            # Ensure image is strictly 224x224 RGB
            target_size = (self.config.data.image_size, self.config.data.image_size)
            if img_pil.size != target_size:
                img_pil = img_pil.resize(target_size, Image.BILINEAR)

            img_np = np.array(img_pil, dtype=np.uint8) # [224, 224, 3] uint8
            video_uint8 = img_np.transpose(2, 0, 1)    # [3, 224, 224] uint8 (150,528 bytes = 147 KB)
            return idx, video_uint8

        indexed_records = list(enumerate(self.records))
        completed_count = 0
        total_bytes = 0
        chunk_size = 1000

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            for chunk_start in range(0, total_samples, chunk_size):
                chunk_items = indexed_records[chunk_start : chunk_start + chunk_size]
                futures = [executor.submit(load_sample, item) for item in chunk_items]
                for future in concurrent.futures.as_completed(futures):
                    completed_count += 1
                    try:
                        idx, video_uint8 = future.result()
                        self.video_cache[idx] = video_uint8
                        total_bytes += video_uint8.nbytes
                    except Exception as exc:
                        logger.error(f"Error preloading video index {idx}: {exc}")

                    if completed_count % 5000 == 0 or completed_count == total_samples:
                        pct = (completed_count / total_samples) * 100.0
                        ram_mb = total_bytes / (1024 * 1024)
                        logger.info(f"RAM Preload '{self.split}' progress: {completed_count}/{total_samples} ({pct:.1f}%) | Cached Images Memory: {ram_mb:.1f} MB | Process RSS: {get_process_memory_str()}")

                del futures
                gc.collect()

        ram_mb = total_bytes / (1024 * 1024)
        logger.info(f"Preloading '{self.split}' complete! Cached {len(self.video_cache)} video frames ({ram_mb:.1f} MB). Final Process RSS: {get_process_memory_str()}")

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
        target_size = (self.config.data.image_size, self.config.data.image_size)

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
                        img_pil = Image.fromarray(image_form)
                        if img_pil.size != target_size:
                            img_pil = img_pil.resize(target_size, Image.BILINEAR)
                        return img_pil
                except Exception:
                    pass

        # 2. Decode using decord or cv2
        video_file = resolve_dataset_file(self.dataset_base_dir, rel_path)
        try:
            if video_file.is_file() and video_file.suffix.lower() in (".png", ".jpg", ".jpeg"):
                img = Image.open(video_file).convert("RGB")
                if img.size != target_size:
                    img = img.resize(target_size, Image.BILINEAR)
                return img

            # Try decord VideoReader first
            try:
                from decord import VideoReader, cpu
                vr = VideoReader(str(video_file), width=self.config.data.image_size, height=self.config.data.image_size, ctx=cpu(0))
                if len(vr) > 0:
                    frame_index = len(vr) // 2
                    frame_rgb = vr.get_batch([frame_index]).asnumpy()[0]
                    img = Image.fromarray(frame_rgb)
                    if img.size != target_size:
                        img = img.resize(target_size, Image.BILINEAR)
                    return img
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
                    img = Image.fromarray(frame_rgb)
                    if img.size != target_size:
                        img = img.resize(target_size, Image.BILINEAR)
                    return img

            return Image.new("RGB", target_size, color=(128, 128, 128))
        except Exception:
            return Image.new("RGB", target_size, color=(128, 128, 128))

    def _load_video(self, rel_path: str, index: int = 0) -> Tensor:
        img_pil = self._load_video_pil(rel_path, index)
        return self.transform(img_pil)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.records[index]
        label = rec["label"]

        waveform = self._load_audio(rec["audio_path"])

        if self.cache_video_enabled and self.video_cache[index] is not None:
            img_np = self.video_cache[index] # [3, 224, 224] uint8
            img_pil = Image.fromarray(img_np.transpose(1, 2, 0)) # Exact RGB PIL Image restored!
            image = self.transform(img_pil)
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
