"""STFT 256k Feature Extractor Utility."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_stft")


def extract_all_stft_features(
    dataset_dir: str = "/marimo/Fish_Feeding_Intensity_Dataset",
    split_dir: str = "../checkpoints/PANNS_Cnn6_holdout_random_sample_20260729_153012/DL_audio/checkpoint/panns_cnn6/splits",
    output_dir: str = "stft256k_features_npy",
    num_workers: int = -1,
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    base_path = Path(dataset_dir)
    split_path = Path(split_dir)

    all_rows = []
    for split in ("train", "val", "test"):
        csv_file = split_path / f"{split}.csv"
        if csv_file.exists():
            with open(csv_file, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_rows.append(row["audio_path"])

    unique_paths = list(set(all_rows))
    logger.info(f"Extracting Raw STFT 2049 Bins for {len(unique_paths)} unique audio files into {out_path}...")

    # Pre-emphasis + STFT Transform
    pre_emphasis_alpha = 0.97
    window = torch.hamming_window(4096)

    for rel_path in tqdm(unique_paths, desc="STFT Extraction"):
        sample_id = Path(rel_path).stem
        save_file = out_path / f"{sample_id}.npy"
        if save_file.exists():
            continue

        audio_file = base_path / rel_path
        if not audio_file.exists():
            continue

        try:
            waveform, sr = torchaudio.load(str(audio_file))
            if waveform.ndim > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.squeeze(0)

            # Pre-emphasis filter
            emphasized = torch.cat([waveform[:1], waveform[1:] - pre_emphasis_alpha * waveform[:-1]], dim=0)

            # Compute Raw STFT
            stft_complex = torch.stft(
                emphasized,
                n_fft=4096,
                hop_length=2048,
                win_length=4096,
                window=window,
                center=True,
                return_complex=True,
            )

            # Compute Power Spectrogram in dB scale
            power_spec = torch.abs(stft_complex) ** 2
            db_spec = 10.0 * torch.log10(power_spec + 1e-10) # [2049, 250]

            # Normalize to [-1, 1]
            db_min = db_spec.min()
            db_max = db_spec.max()
            db_norm = 2.0 * (db_spec - db_min) / (db_max - db_min + 1e-6) - 1.0

            np.save(str(save_file), db_norm.numpy().astype(np.float32))
        except Exception as e:
            logger.warning(f"Error processing {rel_path}: {e}")

    logger.info("STFT 2049 Bins feature extraction complete!")
