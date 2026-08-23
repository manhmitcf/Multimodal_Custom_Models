"""Method 3 (GW-AVF) & Multimodal entry point: train, select by validation, then test holdout."""

from __future__ import annotations

import argparse
import random
import gc
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config.artifact_upload_config import ArtifactUploadConfig
from dataset.paired_loader import SourcePairedDataset, SourceUnlabeledVideoDataset, paired_collate, unlabeled_video_collate
from models.fusion_model import GeometryRippleMultimodalModel, SwinSpatialMultimodal
from models.ibot_pretraining import SwinIbotPretrainer
from models.source_encoders import build_source_encoders
from settings import RunConfig
from tasks.ibot_trainer import IbotTrainer
from tasks.trainer import MultimodalTrainer
from utils.huggingface_results import upload_artifact_if_enabled, upload_result_files


CONFIG_PATH = Path(__file__).parent / "config" / "train_config.json"
UPLOAD_CONFIG_PATH = Path(__file__).parent / "config" / "artifact_upload_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GW-AVF Geometry Water-Ripple Cross-Attention & Multimodal Fusion.")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to a training JSON file. Defaults to config/train_config.json.",
    )
    parser.add_argument(
        "--use-geometry-ripple",
        action="store_true",
        default=True,
        help="Enable Method 3 (GW-AVF) Geometry & Water-Ripple feature enrichment.",
    )
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def paired_loader_workers() -> int:
    """Keep RAM-cached paired samples in one process to avoid cache duplication."""
    return 0


def main() -> None:
    args = parse_args()
    config = RunConfig.from_json(args.config)
    upload_config = ArtifactUploadConfig.from_json(UPLOAD_CONFIG_PATH)

    random.seed(config.training.seed)
    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    device = resolve_device(config.training.device)

    audio_encoder, video_encoder = build_source_encoders(config)

    if config.ibot_pretraining.enabled:
        unlabeled_loader = DataLoader(
            SourceUnlabeledVideoDataset(config),
            batch_size=config.ibot_pretraining.batch_size,
            shuffle=True,
            num_workers=paired_loader_workers(),
            pin_memory=torch.cuda.is_available(),
            collate_fn=unlabeled_video_collate,
        )
        ibot_pretrainer = SwinIbotPretrainer(video_encoder, config.ibot_pretraining).to(device)
        ibot_trainer = IbotTrainer(
            ibot_pretrainer,
            unlabeled_loader,
            device,
            config.ibot_pretraining.output_dir,
        )
        ibot_trainer.fit(config.ibot_pretraining.epochs)
        video_encoder = ibot_pretrainer.student_encoder
        del ibot_trainer
        del ibot_pretrainer
        del unlabeled_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.use_geometry_ripple:
        model = GeometryRippleMultimodalModel(
            audio_encoder,
            video_encoder,
            d_model=config.model.d_model,
            num_heads=config.model.num_heads,
            encoder_mode=config.model.encoder_mode,
            dropout=config.model.dropout,
        ).to(device)
    else:
        model = SwinSpatialMultimodal(
            audio_encoder,
            video_encoder,
            d_model=config.model.d_model,
            num_heads=config.model.num_heads,
            encoder_mode=config.model.encoder_mode,
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

    # Upload CSV summaries and full artifact directory to Hugging Face
    upload_result_files(config.results_upload, config.training.output_dir)
    upload_artifact_if_enabled(upload_config, config.training.output_dir)


if __name__ == "__main__":
    main()
