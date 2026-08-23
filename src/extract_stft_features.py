"""Parallel STFT Feature Extraction Script saving 2049 Bins Spectrograms to Numpy .npy files."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stft_extractor")


def apply_pre_emphasis(waveform: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    if waveform.size == 0:
        return waveform
    return np.append(waveform[0], waveform[1:] - alpha * waveform[:-1]).astype(np.float32)


def compute_stft_spectrogram(
    audio_path: str,
    n_fft: int = 4096,
    hop_length: int = 2048,
    win_length: int = 4096,
    windowing: str = "hamming",
    alpha: float = 0.97,
    target_samples: int = 512000,
) -> np.ndarray:
    """Compute Pre-Emphasis + Raw STFT 2049 Bins Spectrogram [1, 2049, 250]."""
    try:
        waveform_tensor, sr = torchaudio.load(audio_path)
        if waveform_tensor.ndim > 1:
            waveform_tensor = waveform_tensor.mean(dim=0)
        signal = waveform_tensor.numpy().astype(np.float32)
    except Exception:
        signal = np.zeros(target_samples, dtype=np.float32)

    # Pad or crop to target_samples (512,000 for 2s at 256k)
    if signal.size < target_samples:
        signal = np.pad(signal, (0, target_samples - signal.size), mode="constant")
    elif signal.size > target_samples:
        signal = signal[:target_samples]

    # 1. Apply Pre-Emphasis (0.97)
    signal = apply_pre_emphasis(signal, alpha=alpha)

    # 2. Get Window
    if windowing.lower() == "hamming":
        window = torch.hamming_window(win_length)
    else:
        window = torch.hann_window(win_length)

    # 3. Compute STFT via PyTorch
    signal_tensor = torch.from_numpy(signal).unsqueeze(0)
    stft_complex = torch.stft(
        signal_tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )

    power_spec = torch.abs(stft_complex) ** 2
    db_spec = 10.0 * torch.log10(power_spec + 1e-10)

    # Min-Max Normalization to [-1.0, 1.0]
    db_min = db_spec.min()
    db_max = db_spec.max()
    db_norm = 2.0 * (db_spec - db_min) / (db_max - db_min + 1e-6) - 1.0

    return db_norm.squeeze(0).numpy().astype(np.float32) # [2049, 250]


def process_single_audio_file(task: dict[str, Any], dataset_dir: Path, output_dir: Path, config_dict: dict[str, Any]) -> tuple[bool, str]:
    rel_path = task["audio_path"]
    audio_file = dataset_dir / rel_path
    if not audio_file.exists():
        cand = Path("/marimo/Fish_Feeding_Intensity_Dataset") / rel_path
        if cand.exists():
            audio_file = cand

    if not audio_file.exists():
        return False, f"Missing audio: {rel_path}"

    try:
        sample_id = Path(rel_path).stem
        save_path = output_dir / f"{sample_id}.npy"
        if not save_path.exists():
            spec = compute_stft_spectrogram(
                str(audio_file),
                n_fft=config_dict["n_fft"],
                hop_length=config_dict["hop_length"],
                win_length=config_dict["win_length"],
                windowing=config_dict["windowing"],
                alpha=config_dict["pre_emphasis"],
            )
            np.save(save_path, spec)
        return True, str(save_path)
    except Exception as e:
        return False, f"Error processing {rel_path}: {e}"


def extract_all_stft_features(
    dataset_dir: str = "/marimo/Fish_Feeding_Intensity_Dataset",
    split_dir: str = "../checkpoints/PANNS_Cnn6_holdout_random_sample_20260729_153012/DL_audio/checkpoint/panns_cnn6/splits",
    output_dir: str = "stft256k_features_npy",
    num_workers: int = -1,
) -> Path:
    dataset_path = Path(dataset_dir)
    split_path = Path(split_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "sr": 256000,
        "pre_emphasis": 0.97,
        "win_length": 4096,
        "hop_length": 2048,
        "n_fft": 4096,
        "windowing": "hamming",
    }
    (out_path / "config.json").write_text(json.dumps(config_dict, indent=4), encoding="utf-8")

    all_tasks = []
    for split in ("train", "val", "test"):
        csv_file = split_path / f"{split}.csv"
        if not csv_file.exists():
            continue
        with csv_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_tasks.append(dict(row))

    max_workers = num_workers if num_workers > 0 else ((os.cpu_count() or 4) // 2 + 1)
    logger.info(f"Extracting STFT 2049 Bins to Numpy (.npy) for {len(all_tasks)} files using {max_workers} parallel workers...")

    failed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_audio_file, task, dataset_path, out_path, config_dict): task for task in all_tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting STFT Numpy Features", unit="file"):
            success, msg = future.result()
            if not success:
                logger.warning(msg)
                failed += 1

    logger.info(f"Extraction Completed! Total: {len(all_tasks)}, Failed: {failed}, Output Dir: {out_path.resolve()}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="/marimo/Fish_Feeding_Intensity_Dataset")
    parser.add_argument("--split_dir", type=str, default="../checkpoints/PANNS_Cnn6_holdout_random_sample_20260729_153012/DL_audio/checkpoint/panns_cnn6/splits")
    parser.add_argument("--output_dir", type=str, default="stft256k_features_npy")
    parser.add_argument("--num_workers", type=int, default=-1)
    args = parser.parse_args()

    extract_all_stft_features(
        dataset_dir=args.dataset_dir,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
    )
