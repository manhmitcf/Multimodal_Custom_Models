"""STFT dB Image SwinTiny (Audio) + MobileNetV2 (Video) Multimodal Fusion Entry Point."""

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
from models.fusion_model import StftSwinMobileNetMultimodalModel
from models.source_encoders import build_source_encoders, SourceSwinSpatialEncoder, SourceVideoFeatureEncoder
from models.reference_bridge import load_video_reference
from settings import RunConfig
from tasks.ibot_trainer import IbotTrainer
from tasks.trainer import MultimodalTrainer
from utils.huggingface_results import upload_artifact_if_enabled, upload_result_files


CONFIG_PATH = Path(__file__).parent / "config" / "train_config.json"
UPLOAD_CONFIG_PATH = Path(__file__).parent / "config" / "artifact_upload_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run STFT-dB Image SwinTiny (Audio) + MobileNetV2 (Video) Multimodal Fusion.")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to a training JSON file. Defaults to config/train_config.json.",
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

    # Load SwinTiny for Audio STFT dB Image Branch & MobileNetV2 for Video Branch
    video_ref = load_video_reference(config.references.video_repo)
    
    # 1. SwinTiny Backbone for Audio STFT-dB Spectrogram Image
    swin_inner = video_ref.FishVideoDataLoader.build_model("swin_tiny")
    swin_ckpt = config.references.video_repo.parent / "checkpoints/SwinTiny_holdout_random_sample_20260729_153012/DL_video/checkpoint/swin_tiny/video_best.pt"
    if swin_ckpt.exists():
        ckpt = torch.load(swin_ckpt, map_location="cpu", weights_only=False)
        swin_inner.load_state_dict(ckpt["model_state_dict"], strict=True)
    audio_swin_encoder = SourceSwinSpatialEncoder(swin_inner)

    # 2. MobileNetV2 Backbone for Middle RGB Video Frame
    mobilenet_inner = video_ref.FishVideoDataLoader.build_model("mobilenet_v2")
    mobilenet_ckpt = config.references.video_repo.parent / "checkpoints/MobileNetV2_holdout_random_sample_20260729_153012/DL_video/checkpoint/mobilenet_v2/video_best.pt"
    if mobilenet_ckpt.exists():
        ckpt = torch.load(mobilenet_ckpt, map_location="cpu", weights_only=False)
        mobilenet_inner.load_state_dict(ckpt["model_state_dict"], strict=True)
    video_mobilenet_encoder = SourceVideoFeatureEncoder(mobilenet_inner, "mobilenet_v2")

    # 3. Instantiate STFT dB Swin-MobileNet Multimodal Model
    model = StftSwinMobileNetMultimodalModel(
        audio_swin_encoder=audio_swin_encoder.to(device),
        video_mobilenet_encoder=video_mobilenet_encoder.to(device),
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
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        test_loader=loaders["test"],
        device=device,
        output_dir=config.training.output_dir,
        fusion_lr=config.training.fusion_learning_rate,
        encoder_lr=config.training.encoder_learning_rate,
        weight_decay=config.training.weight_decay,
        split_dir=config.data.split_dir,
    )
    trainer.fit_then_test(config.training.epochs)

    # Upload CSV summaries and full artifact directory to Hugging Face
    upload_result_files(config.results_upload, config.training.output_dir)
    upload_artifact_if_enabled(upload_config, config.training.output_dir)


if __name__ == "__main__":
    main()
