"""Single-Modality Audio Only (STFT-dB + PANNS CNN6 Unfrozen) Entry Point."""

from __future__ import annotations

import argparse
import random
import gc
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config.artifact_upload_config import ArtifactUploadConfig
from dataset.paired_loader import SourcePairedDataset, paired_collate
from models.fusion_model import AudioStftPannsOnlyModel
from models.source_encoders import SourceAudioTokenEncoder
from models.reference_bridge import load_audio_reference
from settings import RunConfig
from tasks.trainer import MultimodalTrainer
from utils.huggingface_results import upload_artifact_if_enabled, upload_result_files


CONFIG_PATH = Path(__file__).parent / "config" / "train_config.json"
UPLOAD_CONFIG_PATH = Path(__file__).parent / "config" / "artifact_upload_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Single-Modality Audio Only (STFT-dB + PANNS CNN6 Unfrozen).")
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

    # Build PANNS CNN6 Audio Encoder directly
    audio_ref = load_audio_reference(config.references.audio_repo)
    audio_inner = audio_ref.FishAudioDataLoader.build_model("panns_cnn6")
    if config.checkpoints.audio.exists():
        ckpt = torch.load(config.checkpoints.audio, map_location="cpu", weights_only=False)
        audio_inner.load_state_dict(ckpt["model_state_dict"], strict=True)
    audio_encoder = SourceAudioTokenEncoder(audio_inner)

    # Instantiate Single-Modality Audio STFT + PANNS CNN6 Model
    model = AudioStftPannsOnlyModel(
        audio_panns_encoder=audio_encoder.to(device),
        d_model=config.model.d_model,
        num_heads=config.model.num_heads,
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
