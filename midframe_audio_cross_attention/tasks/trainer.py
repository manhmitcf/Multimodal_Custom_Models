"""Local multimodal trainer: train/validation selection, then one final holdout test."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    macro_f1: float
    per_class_f1: list[float]
    confusion_matrix: list[list[int]]


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> EvaluationResult:
    was_training = model.training
    model.eval()
    prediction_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["waveform"].to(device), batch["image"].to(device))
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
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            total_samples = 0
            for batch in self.train_loader:
                self.optimizer.zero_grad(set_to_none=True)
                labels = batch["label"].to(self.device)
                logits = self.model(batch["waveform"].to(self.device), batch["image"].to(self.device))
                loss = self.loss(logits, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)
            validation = evaluate(self.model, self.val_loader, self.device)
            print(json.dumps({"epoch": epoch, "train_loss": total_loss / total_samples, "val": asdict(validation)}, ensure_ascii=False))
            if validation.macro_f1 > best_score:
                best_score = validation.macro_f1
                torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(), "val_macro_f1": best_score}, self.output_dir / "best.pt")
                self._save_result("best_val", validation)
        checkpoint = torch.load(self.output_dir / "best.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        test = evaluate(self.model, self.test_loader, self.device)
        self._save_result("test", test)
        print(json.dumps({"test": asdict(test)}, ensure_ascii=False, indent=2))
        return test
