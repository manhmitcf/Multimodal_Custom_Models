"""Main Single Entry Point for Spatial Cross-Attention Multimodal Model."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import torch
from torch.utils.data import DataLoader

from dataset.paired_loader import SourcePairedDataset, paired_collate
from models.fusion_model import BaselineSourceMultimodal
from models.source_encoders import build_source_encoders
from settings import RunConfig
from tasks.trainer import MultimodalTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def main() -> None:
    config_path = src_dir / "config" / "train_config.json"
    if not config_path.exists():
        config_path = Path("config/train_config.json")

    logger.info(f"Loading configuration from {config_path}...")
    config = RunConfig.from_json(config_path)

    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)

    device = resolve_device(config.training.device)

    logger.info(f"Building Source Encoders (Audio=PANNS CNN6, Video={config.model.video_backbone})...")
    audio_encoder, video_encoder = build_source_encoders(config)

    logger.info(f"Building Spatial Cross-Attention Multimodal Model (d_model={config.model.d_model}, heads={config.model.num_heads})...")
    model = BaselineSourceMultimodal(
        audio_encoder=audio_encoder,
        video_encoder=video_encoder,
        d_model=config.model.d_model,
        num_heads=config.model.num_heads,
        encoder_mode=config.model.encoder_mode,
        dropout=config.model.dropout,
    )

    logger.info("Building DataLoaders for 'train', 'val', and 'test' splits...")
    loaders = {
        split: DataLoader(
            SourcePairedDataset(config, split),
            batch_size=config.training.batch_size,
            shuffle=(split == "train"),
            num_workers=0, # Use 0 in main process to share multithreaded RAM cache efficiently
            pin_memory=torch.cuda.is_available(),
            collate_fn=paired_collate,
        )
        for split in ("train", "val", "test")
    }

    logger.info("Initializing MultimodalTrainer...")
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
        upload_config=config.results_upload,
    )

    trainer.fit_then_test(config.training.epochs)


if __name__ == "__main__":
    main()
