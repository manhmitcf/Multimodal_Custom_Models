"""Trainer for Custom STFT 256k MobileNet Multimodal Model strictly matching baseline standards."""

from __future__ import annotations

import csv
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sklearn.metrics as sklearn_metrics
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.paired_loader import SourcePairedDataset, paired_collate, resolve_num_workers
from settings import RunConfig

logger = logging.getLogger(__name__)

CLASS_NAMES = ["unfed", "low", "medium", "high"]


@dataclass
class EvaluationResult:
    loss: float
    accuracy: float
    macro_f1: float
    mAP: float
    per_class_f1: list[float]
    per_class_ap: list[float]
    per_class_auc: list[float]
    confusion_matrix: list[list[int]]


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
    progress_desc: str = "Evaluating",
) -> EvaluationResult:
    """Run model evaluation over dataloader and return loss, accuracy, macro-F1, mAP, AUC, and confusion matrix."""
    was_training = model.training
    model.eval()

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    target_batches: list[torch.Tensor] = []
    prediction_batches: list[torch.Tensor] = []
    prob_batches: list[torch.Tensor] = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=progress_desc, leave=False)
        for batch in pbar:
            waveforms = batch["waveform"].to(device, non_blocking=True)
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            logits = model(waveforms, images)
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            prediction_batches.append(preds.cpu())
            prob_batches.append(probs.cpu())
            target_batches.append(labels.cpu())

    if was_training:
        model.train()

    targets = torch.cat(target_batches).numpy()
    predictions = torch.cat(prediction_batches).numpy()
    probs = torch.cat(prob_batches).numpy()
    val_loss = total_loss / max(total_samples, 1)

    # One-hot encode targets for mAP and AUC calculation
    targets_onehot = np.eye(4)[targets]

    # mAP & AP per class
    try:
        ap_scores = sklearn_metrics.average_precision_score(targets_onehot, probs, average=None)
        mAP = float(np.mean(ap_scores))
        per_class_ap = [float(val) for val in ap_scores]
    except Exception:
        mAP = 0.0
        per_class_ap = [0.0] * 4

    # AUC per class
    try:
        auc_scores = sklearn_metrics.roc_auc_score(targets_onehot, probs, average=None)
        per_class_auc = [float(val) for val in auc_scores]
    except Exception:
        per_class_auc = [0.0] * 4

    _, _, f1, _ = precision_recall_fscore_support(targets, predictions, labels=[0, 1, 2, 3], zero_division=0)
    acc = float(accuracy_score(targets, predictions))
    cm = confusion_matrix(targets, predictions, labels=[0, 1, 2, 3]).tolist()

    return EvaluationResult(
        loss=val_loss,
        accuracy=acc,
        macro_f1=float(f1.mean()),
        mAP=mAP,
        per_class_f1=[float(val) for val in f1],
        per_class_ap=per_class_ap,
        per_class_auc=per_class_auc,
        confusion_matrix=cm,
    )


def plot_learning_curves(history_csv_path: Path, output_dir: Path) -> None:
    """Generate 3-panel learning curves (Loss, Accuracy, mAP) matching U_FFIA27K_video."""
    if not history_csv_path.exists():
        return

    epochs, train_losses, val_losses = [], [], []
    val_accs, val_f1s, val_maps = [], [], []

    with history_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epochs.append(int(row["epoch"]))
                train_losses.append(float(row["train_loss"]))
                val_losses.append(float(row.get("val_loss", 0.0)))
                val_accs.append(float(row.get("val_accuracy", 0.0)))
                val_f1s.append(float(row.get("val_macro_f1", 0.0)))
                val_maps.append(float(row.get("val_mAP", 0.0)))
            except Exception:
                continue

    if not epochs:
        return

    import matplotlib
    matplotlib.use("Agg")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Multimodal Fish Feeding Intensity Training History", fontsize=16, fontweight="bold", y=0.98)

    # 1. Loss Panel
    axes[0].plot(epochs, train_losses, label="Train Loss", color="#1f77b4", linewidth=2, linestyle="--")
    axes[0].plot(epochs, val_losses, label="Val Loss", color="#ff7f0e", linewidth=2)
    axes[0].set_title("Loss Curves (Train vs Val)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Loss", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(frameon=True)

    # 2. Accuracy & Macro-F1 Panel
    axes[1].plot(epochs, val_accs, label="Val Accuracy", color="#2ca02c", linewidth=2)
    axes[1].plot(epochs, val_f1s, label="Val Macro-F1", color="#d62728", linewidth=2, linestyle="--")
    axes[1].set_title("Accuracy & Macro-F1 Curves", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Score", fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(frameon=True)

    # 3. mAP Panel
    axes[2].plot(epochs, val_maps, label="Val mAP", color="#9467bd", linewidth=2)
    axes[2].set_title("Mean Average Precision (mAP)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch", fontsize=10)
    axes[2].set_ylabel("mAP Score", fontsize=10)
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend(frameon=True)

    plt.tight_layout()
    plot_path = output_dir / "learning_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Generated learning curve reference plot: '{plot_path}'")


def plot_confusion_matrix_heatmap(cm: list[list[int]], title: str, output_path: Path) -> None:
    """Generate annotated heatmap image for Confusion Matrix matching U_FFIA27K_video."""
    import matplotlib
    matplotlib.use("Agg")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    import matplotlib.pyplot as plt

    cm_np = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm_np, cmap="Blues")
    fig.colorbar(cax)

    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm_np[i, j]), va="center", ha="center", color="black" if cm_np[i, j] < cm_np.max()/2 else "white", fontsize=12, fontweight="bold")

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    plt.title(title, fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("Actual Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Generated confusion matrix reference image: '{output_path}'")


class MultimodalTrainer:
    def __init__(self, config: RunConfig, model: nn.Module) -> None:
        self.config = config
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.output_dir = Path(config.training.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_csv_path = self.output_dir / "history.csv"

        # Setup Optimizer with parameter groups (Fusion vs Encoder)
        fusion_params = [p for n, p in model.named_parameters() if n.startswith("fusion_head.") and p.requires_grad]
        encoder_params = [p for n, p in model.named_parameters() if not n.startswith("fusion_head.") and p.requires_grad]

        groups: list[dict[str, Any]] = [
            {"params": fusion_params, "lr": config.training.fusion_learning_rate, "weight_decay": config.training.weight_decay}
        ]
        if encoder_params:
            groups.append({"params": encoder_params, "lr": config.training.encoder_learning_rate, "weight_decay": config.training.weight_decay})

        self.optimizer = torch.optim.AdamW(groups)
        self.criterion = nn.CrossEntropyLoss()

    def _init_output_directory(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        split_src = Path(self.config.data.split_dir)
        if split_src.exists():
            target_splits = self.output_dir / "splits"
            target_splits.mkdir(parents=True, exist_ok=True)
            for split_file in ("train.csv", "val.csv", "test.csv"):
                src = split_src / split_file
                if src.exists():
                    shutil.copy2(src, target_splits / split_file)

        headers = [
            "epoch",
            "train_loss",
            "val_loss",
            "val_accuracy",
            "val_macro_f1",
            "val_mAP",
            "val_auc_unfed", "val_auc_low", "val_auc_medium", "val_auc_high",
            "val_ap_unfed", "val_ap_low", "val_ap_medium", "val_ap_high",
            "val_f1_unfed", "val_f1_low", "val_f1_medium", "val_f1_high",
        ] + [f"cm_{i}_{j}" for i in range(4) for j in range(4)]

        with self.history_csv_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(headers)

    def _save_result(self, name: str, result: EvaluationResult) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / f"{name}_metrics.json").write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (self.output_dir / f"{name}_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Actual\\Predicted"] + CLASS_NAMES)
            for idx, class_name in enumerate(CLASS_NAMES):
                writer.writerow([class_name] + result.confusion_matrix[idx])

        # Generate annotated image plot for Confusion Matrix
        plot_confusion_matrix_heatmap(
            cm=result.confusion_matrix,
            title=f"Multimodal Confusion Matrix ({name.upper()})",
            output_path=self.output_dir / f"confusion_matrix_{name}.png",
        )

    def _log_history_epoch(self, epoch: int, train_loss: float, validation: EvaluationResult) -> None:
        cm_flat = [val for row in validation.confusion_matrix for val in row]
        row = [
            epoch,
            round(train_loss, 5),
            round(validation.loss, 5),
            round(validation.accuracy, 4),
            round(validation.macro_f1, 4),
            round(validation.mAP, 4),
            round(validation.per_class_auc[0], 4), round(validation.per_class_auc[1], 4), round(validation.per_class_auc[2], 4), round(validation.per_class_auc[3], 4),
            round(validation.per_class_ap[0], 4), round(validation.per_class_ap[1], 4), round(validation.per_class_ap[2], 4), round(validation.per_class_ap[3], 4),
            round(validation.per_class_f1[0], 4), round(validation.per_class_f1[1], 4), round(validation.per_class_f1[2], 4), round(validation.per_class_f1[3], 4),
        ] + cm_flat
        with self.history_csv_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(row)

    def _build_dataloader(self, dataset: SourcePairedDataset, shuffle: bool, split: str = "train") -> DataLoader:
        if split in ("val", "test"):
            num_workers = 2
            use_persistent = False
        else:
            num_workers = min(4, resolve_num_workers(self.config.data.num_workers))
            use_persistent = True

        use_pin = torch.cuda.is_available()
        logger.info(f"Building DataLoader for '{split}' (shuffle={shuffle}): num_workers={num_workers}, pin_memory={use_pin}, persistent_workers={use_persistent}")

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
        self._init_output_directory()
        train_ds = SourcePairedDataset(self.config, "train")
        val_ds = SourcePairedDataset(self.config, "val")
        test_ds = SourcePairedDataset(self.config, "test")

        train_loader = self._build_dataloader(train_ds, shuffle=True, split="train")
        val_loader = self._build_dataloader(val_ds, shuffle=False, split="val")
        test_loader = self._build_dataloader(test_ds, shuffle=False, split="test")

        epochs = self.config.training.epochs
        best_score = float("-inf")
        logger.info("Starting multimodal training pipeline (monitoring validation macro-F1)...")

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            total_samples = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

            for batch in pbar:
                self.optimizer.zero_grad(set_to_none=True)
                waveforms = batch["waveform"].to(self.device, non_blocking=True)
                images = batch["image"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)

                logits = self.model(waveforms, images)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)
                pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Mean Loss": f"{total_loss / total_samples:.4f}"})

            train_loss = total_loss / total_samples
            validation = evaluate(self.model, val_loader, self.device, criterion=self.criterion, progress_desc=f"Validation Epoch {epoch}/{epochs}")
            self._log_history_epoch(epoch, train_loss, validation)
            plot_learning_curves(self.history_csv_path, self.output_dir)

            logger.info(
                "Epoch %d/%d: Train Loss = %.5f | Val Loss = %.5f | Val Acc = %.4f | Val Macro-F1 = %.4f | Val mAP = %.4f | Val F1 = %s",
                epoch,
                epochs,
                train_loss,
                validation.loss,
                validation.accuracy,
                validation.macro_f1,
                validation.mAP,
                [round(val, 4) for val in validation.per_class_f1],
            )

            if validation.macro_f1 > best_score:
                best_score = validation.macro_f1
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_macro_f1": best_score,
                    },
                    self.output_dir / "best.pt",
                )
                self._save_result("best_val", validation)
                logger.info("Saved best fusion checkpoint: '%s' (val macro-F1 = %.4f)", self.output_dir / "best.pt", best_score)

        logger.info("Training complete. Evaluating best checkpoint on holdout test set...")
        best_ckpt = torch.load(self.output_dir / "best.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(best_ckpt["model_state_dict"], strict=True)

        test = evaluate(self.model, test_loader, self.device, criterion=self.criterion, progress_desc="Final holdout test")
        self._save_result("test", test)

        summary_path = self.output_dir / "summary_results.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "val_macro_f1",
                "val_accuracy",
                "val_mAP",
                "test_macro_f1",
                "test_accuracy",
                "test_mAP",
                "test_f1_unfed",
                "test_f1_low",
                "test_f1_medium",
                "test_f1_high",
            ])
            writer.writerow([
                round(best_score, 4),
                round(validation.accuracy, 4),
                round(validation.mAP, 4),
                round(test.macro_f1, 4),
                round(test.accuracy, 4),
                round(test.mAP, 4),
                round(test.per_class_f1[0], 4),
                round(test.per_class_f1[1], 4),
                round(test.per_class_f1[2], 4),
                round(test.per_class_f1[3], 4),
            ])

        logger.info(
            "Holdout Test: Accuracy = %.4f | Macro-F1 = %.4f | mAP = %.4f | Per-class F1 = %s",
            test.accuracy,
            test.macro_f1,
            test.mAP,
            [round(val, 4) for val in test.per_class_f1],
        )

        return {
            "test_macro_f1": test.macro_f1,
            "test_accuracy": test.accuracy,
            "test_mAP": test.mAP,
            "best_val_macro_f1": best_score,
        }
