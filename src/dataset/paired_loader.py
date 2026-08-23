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
import torchvision.transforms.functional as TF
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

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


class ImageToPIL:
    """Convert one RGB image in [C, H, W] or [H, W, C] format to PIL Image matching U_FFIA27K_video exactly."""
    def __call__(self, image: np.ndarray | Image.Image | Tensor) -> Image.Image:
        if isinstance(image, np.ndarray):
            if image.ndim != 3:
                raise ValueError(f"Expected image with 3 dimensions, got shape {tuple(image.shape)}")
            if image.shape[0] == 3:
                image = image.transpose(1, 2, 0)
            return TF.to_pil_image(image)
        if isinstance(image, torch.Tensor):
            if image.ndim != 3:
                raise ValueError(f"Expected image with 3 dimensions, got shape {tuple(image.shape)}")
            if image.shape[0] != 3 and image.shape[-1] == 3:
                image = image.permute(2, 0, 1)
            return TF.to_pil_image(image)
        return image


def build_video_transforms(split: str = "train", image_size: int = 224) -> transforms.Compose:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if split == "train":
        return transforms.Compose(
            [
                ImageToPIL(),
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            ImageToPIL(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


def _load_image_from_disk_cache(
    cache_path: Optional[Path],
    video_path: str,
    image_size: int,
    label: Any
) -> Optional[Dict[str, Any]]:
    """Load pre-processed center frame sample from disk cache matching U_FFIA27K_video."""
    if cache_path is None or not cache_path.exists():
        return None

    try:
        with open(cache_path, "rb") as f:
            sample = pickle.load(f)
    except Exception:
        return None

    image_form = sample.get("image_form")
    expected_shape_chw = (3, image_size, image_size)
    expected_shape_hwc = (image_size, image_size, 3)

    if not isinstance(image_form, np.ndarray) or image_form.dtype != np.uint8:
        return None

    if image_form.shape == expected_shape_hwc:
        image_form = image_form.transpose(2, 0, 1)

    if image_form.shape != expected_shape_chw:
        return None

    return {
        "video_name": video_path,
        "image_form": image_form,
        "target": label,
    }


def _decode_center_image(video_file: Path, label: Any, image_size: int) -> Dict[str, Any]:
    """Decode center frame from video using decord or cv2 matching U_FFIA27K_video."""
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(str(video_file), width=image_size, height=image_size, ctx=cpu(0))
        if len(vr) > 0:
            frame_index = len(vr) // 2
            image = vr.get_batch([frame_index]).asnumpy()[0]  # [H, W, C] RGB
            image_uint8 = image.transpose(2, 0, 1).astype(np.uint8)  # [C, H, W]
            return {"video_name": str(video_file), "image_form": image_uint8, "target": label}
    except Exception:
        pass

    import cv2
    cap = cv2.VideoCapture(str(video_file))
    if cap.isOpened():
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = max(frame_count // 2, 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            image_uint8 = frame_resized.transpose(2, 0, 1).astype(np.uint8)
            return {"video_name": str(video_file), "image_form": image_uint8, "target": label}

    image_uint8 = np.zeros((3, image_size, image_size), dtype=np.uint8)
    return {"video_name": str(video_file), "image_form": image_uint8, "target": label}


def _load_or_create_image_sample(
    index: int,
    rel_video_path: str,
    label: Any,
    image_size: int,
    split: str,
    dataset_base_dir: Path
) -> Dict[str, Any]:
    cache_candidates = [
        Path(f"/marimo/video_cache/single_frame_size_224/{split}/{index}.pkl"),
        Path(f"/marimo/video_cache/{split}/{index}.pkl"),
        Path.cwd() / "video_cache" / f"{split}_{index}.pkl",
    ]
    for cache_path in cache_candidates:
        cached_sample = _load_image_from_disk_cache(cache_path, rel_video_path, image_size, label)
        if cached_sample is not None:
            return cached_sample

    video_file = resolve_dataset_file(dataset_base_dir, rel_video_path)
    return _decode_center_image(video_file, label, image_size)


class SourcePairedDataset(Dataset[dict[str, Any]]):
    """RAM-Optimized Paired Dataset strictly matching U_FFIA27K_video RAM caching architecture."""

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
        self.cache_video_enabled = (config.data.video_cache_mode == "ram")

        self.audio_cache: dict[int, Tensor] = {}
        self.video_cache: list[dict[str, Any] | None] = [None] * len(self.records)

        if self.cache_audio_enabled or self.cache_video_enabled:
            logger.info(f"Initial Process Memory before '{split}' RAM preload: {get_process_memory_str()}")
            self._preload_ram_cache()

    def _preload_ram_cache(self) -> None:
        total_samples = len(self.records)
        num_workers = max(os.cpu_count() or 4, 1)
        logger.info(f"Preloading '{self.split}' split into RAM matching U_FFIA27K_video (uint8 [3, 224, 224] format, ThreadPoolExecutor {num_workers} workers)...")

        def load_sample(idx_rec: tuple[int, dict[str, Any]]) -> tuple[int, Tensor | None, dict[str, Any] | None]:
            idx, rec = idx_rec
            audio_wave = self._load_audio(rec["audio_path"]) if self.cache_audio_enabled else None
            sample_dict = _load_or_create_image_sample(
                index=idx,
                rel_video_path=rec["video_path"],
                label=rec["label"],
                image_size=self.config.data.image_size,
                split=self.split,
                dataset_base_dir=self.dataset_base_dir,
            ) if self.cache_video_enabled else None

            return idx, audio_wave, sample_dict

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
                        idx, audio_wave, sample_dict = future.result()
                        if audio_wave is not None:
                            self.audio_cache[idx] = audio_wave
                        if sample_dict is not None:
                            self.video_cache[idx] = sample_dict
                            total_bytes += sample_dict["image_form"].nbytes
                    except Exception as exc:
                        logger.error(f"Error preloading sample index: {exc}")

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

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.records[index]
        label = rec["label"]

        if self.cache_audio_enabled and index in self.audio_cache:
            waveform = self.audio_cache[index]
        else:
            waveform = self._load_audio(rec["audio_path"])

        if self.cache_video_enabled and self.video_cache[index] is not None:
            sample_dict = self.video_cache[index]
            image_uint8 = sample_dict["image_form"] # np.ndarray [3, 224, 224] uint8
            image = self.transform(image_uint8)
        else:
            sample_dict = _load_or_create_image_sample(
                index=index,
                rel_video_path=rec["video_path"],
                label=label,
                image_size=self.config.data.image_size,
                split=self.split,
                dataset_base_dir=self.dataset_base_dir,
            )
            image = self.transform(sample_dict["image_form"])

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
