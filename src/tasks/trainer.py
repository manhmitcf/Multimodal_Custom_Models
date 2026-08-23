"""Trainer for Custom STFT 256k MobileNetV2 Multimodal Model."""

from __future__ import annotations

import csv
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset.paired_loader import SourcePairedDataset, paired_collate, resolve_num_workers
from models.fusion_model import CustomSTFT256kMobileNetMultimodalModel
from settings import RunConfig
from utils.model_profile import estimate_flops

logger = logging.getLogger(__name__)


class MultimodalTrainer:
    def __init__(self, config: RunConfig, model: CustomSTFT256kMobileNetMultimodalModel) -> None:
        self.config = config
        self.model = model

        self.device = self._resolve_device(config.training.device)
        self.model.to(self.device)

        # Profile parameters
        dummy_wave = torch.randn(2, 512000, device=self.device)
        dummy_img = torch.randn(2, 3, 224, 224, device=self.device)
        self.profile = estimate_flops(self.model, dummy_wave, dummy_img)

        # Separate learning rates for fusion head vs pretrained encoders
        fusion_params = list(self.model.fusion.parameters()) + list(self.model.audio_cnn.parameters())
        encoder_params = list(self.model.video_encoder.parameters())

        self.optimizer = torch.optim.AdamW(
            [
                {"params": fusion_params, "lr": config.training.fusion_learning_rate},
                {"params": encoder_params, "lr": config.training.encoder_learning_rate},
            ],
            weight_decay=config.training.weight_decay,
        )

        self.criterion = nn.CrossEntropyLoss()
        self.output_dir = Path(config.training.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.splits_dir = self.output_dir / "splits"
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        self._copy_splits()

    def _resolve_device(self, requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)

    def _copy_splits(self) -> None:
        split_src = Path(self.config.data.split_dir)
        for split in ("train", "val", "test"):
            src_file = split_src / f"{split}.csv"
            if src_file.exists():
                shutil.copy(src_file, self.splits_dir / f"{split}.csv")

    def _build_dataloader(self, dataset: SourcePairedDataset, shuffle: bool) -> DataLoader:
        is_ram_cached = getattr(dataset, "cache_audio_enabled", False) and getattr(dataset, "cache_video_enabled", False)
        num_workers = resolve_num_workers(self.config.data.num_workers, is_ram_cached=is_ram_cached)
        use_persistent = num_workers > 0
        use_pin = torch.cuda.is_available()

        logger.info(f"Building DataLoader (shuffle={shuffle}): num_workers={num_workers}, pin_memory={use_pin}, persistent_workers={use_persistent}")
        return DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=paired_collate,
            pin_memory=use_pin,
            persistent_workers=use_persistent,
        )

    def fit_then_test(self) -> dict[str, Any]:
        train_ds = SourcePairedDataset(self.config, "train")
        val_ds = SourcePairedDataset(self.config, "val")
        test_ds = SourcePairedDataset(self.config, "test")

        train_loader = self._build_dataloader(train_ds, shuffle=True)
        val_loader = self._build_dataloader(val_ds, shuffle=False)
        test_loader = self._build_dataloader(test_ds, shuffle=False)

        history_path = self.output_dir / "history.csv"
        history_fields = [
            "epoch",
            "train_loss",
            "train_acc",
            "train_macro_f1",
            "val_loss",
            "val_acc",
            "val_macro_f1",
        ] + [f"val_cm_{i}_{j}" for i in range(4) for j in range(4)]

        best_val_f1 = -1.0
        best_val_metrics = {}

        with history_path.open("w", encoding="utf-8", newline="") as h_file:
            writer = csv.DictWriter(h_file, fieldnames=history_fields)
            writer.writeheader()

            for epoch in range(1, self.config.training.epochs + 1):
                train_loss, train_acc, train_f1 = self._train_epoch(train_loader, epoch=epoch)
                val_loss, val_acc, val_f1, val_cm = self._evaluate(val_loader, desc=f"Epoch {epoch:03d} Validation")

                row = {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "train_acc": f"{train_acc:.6f}",
                    "train_macro_f1": f"{train_f1:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                    "val_acc": f"{val_acc:.6f}",
                    "val_macro_f1": f"{val_f1:.6f}",
                }
                for i in range(4):
                    for j in range(4):
                        row[f"val_cm_{i}_{j}"] = int(val_cm[i, j])
                writer.writerow(row)
                h_file.flush()

                logger.info(f"Epoch {epoch:03d}/{self.config.training.epochs:03d} | Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} (Best: {max(best_val_f1, val_f1):.4f})")

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_val_metrics = {
                        "epoch": epoch,
                        "val_loss": float(val_loss),
                        "val_acc": float(val_acc),
                        "val_macro_f1": float(val_f1),
                        "profile": self.profile,
                    }
                    torch.save(self.model.state_dict(), self.output_dir / "best.pt")
                    np.savetxt(self.output_dir / "best_val_confusion_matrix.csv", val_cm, fmt="%d", delimiter=",")

        # Save Best Val Metrics
        (self.output_dir / "best_val_metrics.json").write_text(json.dumps(best_val_metrics, indent=4), encoding="utf-8")

        # Load Best Model for Testing
        self.model.load_state_dict(torch.load(self.output_dir / "best.pt", map_location=self.device))
        test_loss, test_acc, test_f1, test_cm = self._evaluate(test_loader)

        test_metrics = {
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "test_macro_f1": float(test_f1),
            "profile": self.profile,
        }
        (self.output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=4), encoding="utf-8")
        np.savetxt(self.output_dir / "test_confusion_matrix.csv", test_cm, fmt="%d", delimiter=",")

        # Summary Results CSV
        summary_path = self.output_dir / "summary_results.csv"
        with summary_path.open("w", encoding="utf-8", newline="") as s_file:
            s_writer = csv.DictWriter(
                s_file,
                fieldnames=[
                    "model_name",
                    "fusion_type",
                    "best_val_macro_f1",
                    "test_macro_f1",
                    "test_acc",
                    "total_params",
                    "trainable_params",
                ],
            )
            s_writer.writeheader()
            s_writer.writerow(
                {
                    "model_name": "CustomSTFT256kMobileNetMultimodalModel",
                    "fusion_type": self.config.model.fusion_type,
                    "best_val_macro_f1": f"{best_val_f1:.6f}",
                    "test_macro_f1": f"{test_f1:.6f}",
                    "test_acc": f"{test_acc:.6f}",
                    "total_params": self.profile.get("total_params", 0),
                    "trainable_params": self.profile.get("trainable_params", 0),
                }
            )

        logger.info(f"FIT & TEST COMPLETED! Test Macro-F1: {test_f1:.4f} | Total Params: {self.profile.get('total_params', 0):,}")
        return test_metrics

    def _train_epoch(self, dataloader: DataLoader, epoch: int = 1) -> tuple[float, float, float]:
        import sys
        from tqdm import tqdm

        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        disable_progress = not sys.stdout.isatty()
        for batch in tqdm(dataloader, desc=f"Epoch {epoch:03d} Training", leave=False, unit="batch", disable=disable_progress):
            waveforms = batch["waveform"].to(self.device, non_blocking=True)
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(waveforms, images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / len(all_labels)
        acc, f1, _ = compute_metrics(all_preds, all_labels)
        return avg_loss, acc, f1

    def _evaluate(self, dataloader: DataLoader, desc: str = "Evaluating") -> tuple[float, float, float, np.ndarray]:
        import sys
        from tqdm import tqdm

        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        disable_progress = not sys.stdout.isatty()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc, leave=False, unit="batch", disable=disable_progress):
                waveforms = batch["waveform"].to(self.device, non_blocking=True)
                images = batch["image"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)

                logits = self.model(waveforms, images)
                loss = self.criterion(logits, labels)

                total_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / len(all_labels)
        acc, f1, cm = compute_metrics(all_preds, all_labels)
        return avg_loss, acc, f1, cm


def compute_metrics(preds: list[int], labels: list[int]) -> tuple[float, float, np.ndarray]:
    p = np.array(preds)
    l = np.array(labels)
    acc = float((p == l).mean())

    cm = np.zeros((4, 4), dtype=int)
    for pred, label in zip(preds, labels):
        cm[label, pred] += 1

    f1s = []
    for c in range(4):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        f1s.append(f1)

    macro_f1 = float(np.mean(f1s))
    return acc, macro_f1, cm
