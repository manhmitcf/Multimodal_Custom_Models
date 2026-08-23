"""Main Single Entry Point for STFT 256k Raw 2049 Bins MobileNetV2 Multimodal Model."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

from models.custom_audio_cnn import CustomRawStftAudioCNN
from models.fusion_model import CustomSTFT256kMobileNetMultimodalModel
from models.reference_bridge import load_video_reference
from settings import RunConfig
from tasks.trainer import MultimodalTrainer
from utils.huggingface_results import upload_results_artifact

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")


def main() -> None:
    config_path = Path(__file__).resolve().parent / "config" / "train_config.json"
    if not config_path.exists():
        config_path = Path("config/train_config.json")

    logger.info(f"Loading configuration from {config_path}...")
    config = RunConfig.from_json(config_path)

    # Set random seed
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)

    # Build Audio CNN (Raw STFT 2049 + F-Attn + Depthwise Blocks)
    logger.info(f"Building Custom Raw STFT Audio CNN (n_fft={config.stft.n_fft}, pre_emphasis={config.stft.pre_emphasis})...")
    audio_cnn = CustomRawStftAudioCNN(
        feature_dim=config.model.d_model,
        n_fft=config.stft.n_fft,
        hop_length=config.stft.hop_length,
        win_length=config.stft.frame_length,
        windowing=config.stft.windowing,
        use_std=config.stft.use_std,
        pre_emphasis=config.stft.pre_emphasis,
    )

    # Build Video Backbone (MobileNetV2)
    logger.info(f"Building Video Backbone ({config.model.video_backbone}) from {config.video_checkpoint}...")
    video_ref = load_video_reference(config.references.video_repo)
    video_encoder = video_ref.build_video_backbone(
        backbone_name=config.model.video_backbone,
        num_classes=4,
        checkpoint_path=config.video_checkpoint,
    )

    # Build Multimodal Model
    logger.info(f"Building Custom STFT 256k Multimodal Model (fusion_type={config.model.fusion_type})...")
    model = CustomSTFT256kMobileNetMultimodalModel(
        audio_cnn=audio_cnn,
        video_encoder=video_encoder,
        d_model=config.model.d_model,
        dropout=config.model.dropout,
        fusion_type=config.model.fusion_type,
        encoder_mode=config.model.encoder_mode,
    )

    # Train and Evaluate Model
    logger.info("Initializing MultimodalTrainer...")
    trainer = MultimodalTrainer(config=config, model=model)
    test_metrics = trainer.fit_then_test()

    # Upload Artifacts to Hugging Face
    upload_config_path = Path(__file__).resolve().parent / "config" / "artifact_upload_config.json"
    if upload_config_path.exists():
        logger.info("Triggering automatic Hugging Face artifact upload...")
        upload_results_artifact(upload_config_path)

    logger.info(f"ALL DONE SUCCESSFULLY! Final Test Macro-F1: {test_metrics['test_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
