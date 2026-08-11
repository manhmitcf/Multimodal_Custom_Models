"""Single baseline-style entry point: train, select by validation, then test holdout."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.paired_loader import SourcePairedDataset, paired_collate, read_immutable_split
from features.stft_features import StftConfig, prepare_stft_cache
from models.fusion_model import StftSpatialMultimodal
from models.source_encoders import build_source_video_encoder
from settings import RunConfig
from tasks.trainer import MultimodalTrainer


CONFIG_PATH = Path(__file__).parent / "config" / "train_config.json"


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def paired_loader_workers() -> int:
    """Keep RAM-cached paired samples in one process to avoid cache duplication."""
    return 0


def main() -> None:
    config = RunConfig.from_json(CONFIG_PATH)
    random.seed(config.training.seed)
    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    device = resolve_device(config.training.device)
    stft_config = StftConfig(
        sample_rate=config.audio.sample_rate,
        duration_seconds=config.audio.duration_seconds,
        window_seconds=config.audio.window_seconds,
        hop_seconds=config.audio.token_hop_seconds,
        pre_emphasis=config.audio.pre_emphasis,
        frame_length=config.audio.frame_length,
        hop_length=config.audio.hop_length,
        n_fft=config.audio.n_fft,
        windowing=config.audio.windowing,
        use_std=config.audio.use_std,
    )
    records = {split: read_immutable_split(config.data.split_dir, split) for split in ("train", "val", "test")}
    stft_cache_dir = prepare_stft_cache(records, config.data.stft_cache_dir, stft_config, config.data.num_workers)
    video_encoder = build_source_video_encoder(config)
    model = StftSpatialMultimodal(
        video_encoder,
        audio_dim=stft_config.feature_dim,
        d_model=config.model.d_model,
        num_heads=config.model.num_heads,
        encoder_mode=config.model.encoder_mode,
        audio_positional_encoding=config.model.audio_positional_encoding,
        visual_positional_encoding=config.model.visual_positional_encoding,
        dropout=config.model.dropout,
    ).to(device)
    loaders = {
        split: DataLoader(
            SourcePairedDataset(config, split, stft_cache_dir),
            batch_size=config.training.batch_size,
            shuffle=split == "train",
            num_workers=paired_loader_workers(),
            pin_memory=torch.cuda.is_available(),
            collate_fn=paired_collate,
        )
        for split in ("train", "val", "test")
    }
    trainer = MultimodalTrainer(
        model,
        loaders["train"],
        loaders["val"],
        loaders["test"],
        device,
        config.training.output_dir,
        config.training.fusion_learning_rate,
        config.training.encoder_learning_rate,
        config.training.weight_decay,
    )
    trainer.fit_then_test(config.training.epochs)


if __name__ == "__main__":
    main()
