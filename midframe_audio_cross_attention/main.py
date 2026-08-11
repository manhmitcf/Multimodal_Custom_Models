"""Single baseline-style entry point: train, select by validation, then test holdout."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.paired_loader import SourcePairedDataset, paired_collate
from models.fusion_model import BaselineSourceMultimodal
from models.source_encoders import build_source_encoders
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
    audio_encoder, video_encoder = build_source_encoders(config)
    model = BaselineSourceMultimodal(
        audio_encoder,
        video_encoder,
        d_model=config.model.d_model,
        num_heads=config.model.num_heads,
        encoder_mode=config.model.encoder_mode,
        positional_encoding=config.model.positional_encoding,
        dropout=config.model.dropout,
    ).to(device)
    loaders = {
        split: DataLoader(
            SourcePairedDataset(config, split),
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
