"""Training loop for label-free iBOT-inspired Swin adaptation."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.ibot_pretraining import SwinIbotPretrainer


logger = logging.getLogger(__name__)


class IbotTrainer:
    """Optimize the student encoder on immutable train frames and save best loss."""

    def __init__(self, pretrainer: SwinIbotPretrainer, loader: DataLoader, device: torch.device, output_dir: Path) -> None:
        self.pretrainer = pretrainer
        self.loader = loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.optimizer = torch.optim.AdamW(
            pretrainer.trainable_parameters(),
            lr=pretrainer.config.learning_rate,
            weight_decay=pretrainer.config.weight_decay,
        )

    def _checkpoint_path(self) -> Path:
        return self.output_dir / "ibot_pretrain_best.pt"

    def fit(self, epochs: int) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        best_loss = float("inf")
        logger.info("Starting label-free iBOT-inspired Swin adaptation from immutable train.csv frames only.")
        logger.info("Validation and holdout-test video frames are excluded from iBOT pretraining.")
        for epoch in range(1, epochs + 1):
            self.pretrainer.train()
            total_loss = 0.0
            total_samples = 0
            progress = tqdm(self.loader, desc=f"iBOT epoch {epoch}/{epochs}")
            for batch in progress:
                images = batch["image"].to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                losses = self.pretrainer.compute_loss(images, images)
                losses.total.backward()
                self.optimizer.step()
                self.pretrainer.update_teacher()
                total_loss += float(losses.total.detach()) * images.shape[0]
                total_samples += images.shape[0]
                progress.set_postfix(
                    {
                        "total": f"{float(losses.total.detach()):.4f}",
                        "patch": f"{float(losses.patch.detach()):.4f}",
                        "global": f"{float(losses.global_feature.detach()):.4f}",
                    }
                )
            epoch_loss = total_loss / max(total_samples, 1)
            logger.info("iBOT epoch %d/%d: mean training loss = %.6f", epoch, epochs, epoch_loss)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "training_loss": best_loss,
                        "pretrainer_state_dict": self.pretrainer.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "config": self.pretrainer.config,
                    },
                    self._checkpoint_path(),
                )
                logger.info("Saved best iBOT adaptation checkpoint: '%s'", self._checkpoint_path())
        checkpoint = torch.load(self._checkpoint_path(), map_location=self.device, weights_only=False)
        self.pretrainer.load_state_dict(checkpoint["pretrainer_state_dict"], strict=True)
        return self._checkpoint_path()
