"""Librosa STFT tokens matching the established classical ML feature recipe."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
from dataclasses import asdict
from tqdm import tqdm


@dataclass(frozen=True)
class StftConfig:
    sample_rate: int = 256000
    duration_seconds: float = 2.0
    window_seconds: float = 0.75
    hop_seconds: float = 0.25
    pre_emphasis: float = 0.97
    frame_length: int = 4096
    hop_length: int = 2048
    n_fft: int = 4096
    windowing: str = "hamming"
    use_std: bool = False

    @property
    def target_samples(self) -> int:
        return int(self.sample_rate * self.duration_seconds)

    @property
    def window_samples(self) -> int:
        return int(self.sample_rate * self.window_seconds)

    @property
    def token_hop_samples(self) -> int:
        return int(self.sample_rate * self.hop_seconds)

    @property
    def feature_dim(self) -> int:
        return (self.n_fft // 2 + 1) * (2 if self.use_std else 1)

    @property
    def num_tokens(self) -> int:
        return 1 + (self.target_samples - self.window_samples) // self.token_hop_samples


def _pre_emphasis(signal: np.ndarray, coefficient: float) -> np.ndarray:
    if signal.size == 0 or coefficient == 0:
        return signal.astype(np.float32, copy=False)
    return np.append(signal[0], signal[1:] - coefficient * signal[:-1]).astype(np.float32)


def _normalise_duration(signal: np.ndarray, target_samples: int) -> np.ndarray:
    signal = signal[:target_samples]
    if signal.size < target_samples:
        signal = np.pad(signal, (0, target_samples - signal.size))
    return signal.astype(np.float32, copy=False)


def _extract_one_window(signal: np.ndarray, config: StftConfig) -> np.ndarray:
    stft = librosa.stft(
        y=_pre_emphasis(signal, config.pre_emphasis),
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.frame_length,
        window=config.windowing,
    )
    log_magnitude = np.log(np.abs(stft) + 1e-10)
    mean = np.mean(log_magnitude, axis=1)
    if not config.use_std:
        return mean.astype(np.float32)
    return np.concatenate((mean, np.std(log_magnitude, axis=1))).astype(np.float32)


def extract_stft_tokens(signal: np.ndarray, config: StftConfig) -> np.ndarray:
    """Return one classical mean-log-STFT vector per overlapping audio window."""
    signal = _normalise_duration(np.asarray(signal, dtype=np.float32), config.target_samples)
    tokens = [
        _extract_one_window(signal[start : start + config.window_samples], config)
        for start in range(0, config.target_samples - config.window_samples + 1, config.token_hop_samples)
    ]
    output = np.stack(tokens).astype(np.float32, copy=False)
    if output.shape != (config.num_tokens, config.feature_dim):
        raise RuntimeError(f"Unexpected STFT token shape: {output.shape}")
    return output


def extract_stft_tokens_from_file(audio_path: str, config: StftConfig) -> np.ndarray:
    signal, _ = librosa.load(audio_path, sr=config.sample_rate, mono=True)
    return extract_stft_tokens(signal, config)


def cache_directory(cache_root: Path, config: StftConfig) -> Path:
    fingerprint = hashlib.md5(json.dumps(asdict(config), sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return Path(cache_root) / f"stft_sr{config.sample_rate}_nfft{config.n_fft}_hop{config.hop_length}_{fingerprint}"


def _cache_one(audio_path: str, output_path: str, config: StftConfig) -> None:
    feature = extract_stft_tokens_from_file(audio_path, config)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, feature)


def prepare_stft_cache(records_by_split: dict[str, list[list[object]]], cache_root: Path, config: StftConfig, workers: int) -> Path:
    """Precompute missing Librosa features once, then reuse them as immutable NPY files."""
    root = cache_directory(cache_root, config)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    for split, records in records_by_split.items():
        missing = [
            (str(record[0]), str(root / split / f"{index}.npy"))
            for index, record in enumerate(records)
            if not (root / split / f"{index}.npy").exists()
        ]
        if not missing:
            continue
        with ProcessPoolExecutor(max_workers=workers or None) as executor:
            futures = [executor.submit(_cache_one, audio_path, output_path, config) for audio_path, output_path in missing]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Caching {split} STFT", unit="file"):
                future.result()
    return root
