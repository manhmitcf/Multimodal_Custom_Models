"""Multimodal Trainer for Spatial Cross-Attention Model."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.huggingface_results import upload_results_artifact

logger = logging.getLogger(__name__)


class MultimodalTrainer:
    """Trainer managing training epochs, validation selection, holdout evaluation, and HF upload."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        output_dir: str = "checkpoint",
        fusion_lr: float = 0.001,
        encoder_lr: float = 0.001,
        weight_decay: float = 0.0001,
        upload_config: Any = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.upload_config = upload_config

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=fusion_lr,
            weight_decay=weight_decay,
        )

    def train_epoch(self, epoch: int, total_epochs: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)
        for i, batch in enumerate(pbar):
            waveforms = batch["waveform"].to(self.device)
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(waveforms, images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Mean Loss": f"{total_loss / (i + 1):.4f}"})

        mean_loss = total_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch}/{total_epochs}: Train Loss = {mean_loss:.4f}")
        return mean_loss

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split_name: str = "val") -> dict[str, float]:
        self.model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0.0

        for batch in loader:
            waveforms = batch["waveform"].to(self.device)
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            logits = self.model(waveforms, images)
            loss = self.criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())

        all_preds_np = np.array(all_preds)
        all_targets_np = np.array(all_targets)

        acc = float(accuracy_score(all_targets_np, all_preds_np))
        macro_f1 = float(f1_score(all_targets_np, all_preds_np, average="macro"))
        per_class_f1 = f1_score(all_targets_np, all_preds_np, average=None)

        logger.info(f"--- {split_name.upper()} Evaluation Results ---")
        logger.info(f"{split_name.capitalize()} Accuracy : {acc:.4f}")
        logger.info(f"{split_name.capitalize()} Macro-F1: {macro_f1:.4f}")
        for cls_idx, f1_val in enumerate(per_class_f1):
            logger.info(f"  Class {cls_idx} F1: {f1_val:.4f}")

        return {
            "loss": total_loss / max(len(loader), 1),
            "accuracy": acc,
            "macro_f1": macro_f1,
            "f1_class_0": float(per_class_f1[0]) if len(per_class_f1) > 0 else 0.0,
            "f1_class_1": float(per_class_f1[1]) if len(per_class_f1) > 1 else 0.0,
            "f1_class_2": float(per_class_f1[2]) if len(per_class_f1) > 2 else 0.0,
            "f1_class_3": float(per_class_f1[3]) if len(per_class_f1) > 3 else 0.0,
        }

    def fit_then_test(self, epochs: int) -> None:
        best_val_f1 = -1.0
        best_ckpt_path = self.output_dir / "best_model.pt"

        logger.info(f"Starting Spatial Cross-Attention Multimodal Training ({epochs} epochs)...")
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch, epochs)
            val_metrics = self.evaluate(self.val_loader, split_name="val")

            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                logger.info(f"🔥 New Best Validation Macro-F1: {best_val_f1:.4f}! Saving checkpoint to {best_ckpt_path}...")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_metrics": val_metrics,
                    },
                    best_ckpt_path,
                )

        logger.info("\n==================================================")
        logger.info(f"Training Complete! Best Val Macro-F1 = {best_val_f1:.4f}")
        logger.info(f"Loading best checkpoint from {best_ckpt_path} for Holdout Test set evaluation...")

        if best_ckpt_path.exists():
            checkpoint = torch.load(best_ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        test_metrics = self.evaluate(self.test_loader, split_name="test")

        # Save test results summary JSON
        results_file = self.output_dir / "results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump({"best_val_f1": best_val_f1, "test_metrics": test_metrics}, f, indent=2)

        # Upload to HuggingFace if enabled
        if self.upload_config and getattr(self.upload_config, "enabled", False):
            logger.info("Uploading results and best checkpoint to HuggingFace...")
            try:
                upload_results_artifact(
                    checkpoint_dir=str(self.output_dir),
                    repo_id=getattr(self.upload_config, "repo_id", "manhmitcf/fish_result"),
                    repo_type=getattr(self.upload_config, "repo_type", "dataset"),
                    path_prefix=getattr(self.upload_config, "path_prefix", "midframe_audio_cross_attention"),
                )
            except Exception as e:
                logger.warning(f"Failed to upload artifact to HuggingFace: {e}")
