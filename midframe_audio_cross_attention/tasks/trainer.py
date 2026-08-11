"""Local multimodal trainer: train/validation selection, then one final holdout test."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    macro_f1: float
    per_class_f1: list[float]
    confusion_matrix: list[list[int]]


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, progress_desc: str = "Running model evaluation...") -> EvaluationResult:
    was_training = model.training
    model.eval()
    prediction_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=progress_desc):
            logits = model(batch["audio_features"].to(device), batch["image"].to(device))
            prediction_batches.append(logits.argmax(dim=1).cpu())
            target_batches.append(batch["label"].cpu())
    if was_training:
        model.train()
    targets = torch.cat(target_batches).numpy()
    predictions = torch.cat(prediction_batches).numpy()
    _, _, f1, _ = precision_recall_fscore_support(targets, predictions, labels=[0, 1, 2, 3], zero_division=0)
    return EvaluationResult(
        accuracy=float(accuracy_score(targets, predictions)),
        macro_f1=float(f1.mean()),
        per_class_f1=[float(value) for value in f1],
        confusion_matrix=confusion_matrix(targets, predictions, labels=[0, 1, 2, 3]).tolist(),
    )


class MultimodalTrainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, device: torch.device, output_dir: Path, fusion_lr: float, encoder_lr: float, weight_decay: float) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = Path(output_dir)
        fusion = [parameter for name, parameter in model.named_parameters() if name.startswith("fusion.") and parameter.requires_grad]
        encoder = [parameter for name, parameter in model.named_parameters() if not name.startswith("fusion.") and parameter.requires_grad]
        groups: list[dict[str, object]] = [{"params": fusion, "lr": fusion_lr, "weight_decay": weight_decay}]
        if encoder:
            groups.append({"params": encoder, "lr": encoder_lr, "weight_decay": weight_decay})
        self.optimizer = torch.optim.AdamW(groups)
        self.loss = nn.CrossEntropyLoss()

    def _save_result(self, name: str, result: EvaluationResult) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / f"{name}_metrics.json").write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
        with (self.output_dir / f"{name}_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(result.confusion_matrix)

    def fit_then_test(self, epochs: int) -> EvaluationResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        best_score = float("-inf")
        logger.info("Starting multimodal training pipeline (monitor metric: validation macro-F1)...")
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            total_samples = 0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{epochs}")
            for batch in pbar:
                self.optimizer.zero_grad(set_to_none=True)
                labels = batch["label"].to(self.device)
                logits = self.model(batch["audio_features"].to(self.device), batch["image"].to(self.device))
                loss = self.loss(logits, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)
                pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Mean Loss": f"{total_loss / total_samples:.4f}"})
            train_loss = total_loss / total_samples
            validation = evaluate(self.model, self.val_loader, self.device, progress_desc=f"Validation Epoch {epoch}/{epochs}")
            logger.info(
                "Epoch %d/%d: Train Loss = %.5f | Val Accuracy = %.4f | Val Macro-F1 = %.4f | Val Per-class F1 = %s",
                epoch,
                epochs,
                train_loss,
                validation.accuracy,
                validation.macro_f1,
                [round(value, 4) for value in validation.per_class_f1],
            )
            if validation.macro_f1 > best_score:
                best_score = validation.macro_f1
                torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(), "val_macro_f1": best_score}, self.output_dir / "best.pt")
                self._save_result("best_val", validation)
                logger.info("Saved best fusion checkpoint: '%s' (validation macro-F1 = %.4f)", self.output_dir / "best.pt", best_score)
        checkpoint = torch.load(self.output_dir / "best.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        logger.info("Training complete. Starting final holdout-test evaluation using best validation checkpoint...")
        test = evaluate(self.model, self.test_loader, self.device, progress_desc="Final holdout test")
        self._save_result("test", test)
        logger.info(
            "Holdout Test: Accuracy = %.4f | Macro-F1 = %.4f | Per-class F1 = %s | Confusion Matrix = %s",
            test.accuracy,
            test.macro_f1,
            [round(value, 4) for value in test.per_class_f1],
            test.confusion_matrix,
        )
        return test
