"""iBOT-inspired masked spatial adaptation for the source SwinTiny encoder."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.source_encoders import SourceSwinSpatialEncoder
from settings import IbotPretrainingConfig


def make_block_mask(
    batch_size: int,
    grid_size: int,
    mask_ratio: float,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """Create an independently shuffled boolean mask for each spatial grid."""
    if batch_size <= 0 or grid_size <= 0:
        raise ValueError("batch_size and grid_size must be positive.")
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must be between 0 and 1.")
    token_count = grid_size * grid_size
    masked_count = max(1, min(token_count - 1, round(token_count * mask_ratio)))
    scores = torch.rand(batch_size, token_count, generator=generator, device=device)
    selected = scores.topk(masked_count, dim=1, largest=False).indices
    mask = torch.zeros(batch_size, token_count, dtype=torch.bool, device=device)
    return mask.scatter_(1, selected, True)


def masked_token_cross_entropy(
    student_logits: Tensor,
    teacher_logits: Tensor,
    mask: Tensor,
    student_temperature: float,
    teacher_temperature: float,
    center: Tensor,
) -> Tensor:
    """Cross-entropy at masked positions only, using centered teacher targets."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Student and teacher patch logits must have the same shape.")
    if student_logits.ndim != 3 or mask.shape != student_logits.shape[:2]:
        raise ValueError("Expected logits [batch, tokens, prototypes] and mask [batch, tokens].")
    if not bool(mask.any()):
        raise ValueError("At least one token must be masked.")
    with torch.no_grad():
        targets = F.softmax((teacher_logits - center) / teacher_temperature, dim=-1)
    per_token_loss = -(targets * F.log_softmax(student_logits / student_temperature, dim=-1)).sum(dim=-1)
    return per_token_loss.masked_select(mask).mean()


def global_cross_entropy(
    student_logits: Tensor,
    teacher_logits: Tensor,
    student_temperature: float,
    teacher_temperature: float,
    center: Tensor,
) -> Tensor:
    """DINO-style global feature matching for a Swin encoder without a CLS token."""
    with torch.no_grad():
        targets = F.softmax((teacher_logits - center) / teacher_temperature, dim=-1)
    return -(targets * F.log_softmax(student_logits / student_temperature, dim=-1)).sum(dim=-1).mean()


class IbotProjectionHead(nn.Module):
    """Small train-only head mapping Swin features to teacher/student prototypes."""

    def __init__(self, input_dim: int, num_prototypes: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, num_prototypes),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(features)


@dataclass(frozen=True)
class IbotLoss:
    total: Tensor
    patch: Tensor
    global_feature: Tensor


class SwinIbotPretrainer(nn.Module):
    """Student/EMA-teacher iBOT-inspired adaptation over Swin stage-3 tokens."""

    def __init__(self, student_encoder: SourceSwinSpatialEncoder, config: IbotPretrainingConfig) -> None:
        super().__init__()
        self.config = config
        self.student_encoder = student_encoder
        self.teacher_encoder = copy.deepcopy(student_encoder)
        self.student_patch_head = IbotProjectionHead(student_encoder.feature_dim, config.num_prototypes)
        self.teacher_patch_head = copy.deepcopy(self.student_patch_head)
        self.student_global_head = IbotProjectionHead(768, config.num_prototypes)
        self.teacher_global_head = copy.deepcopy(self.student_global_head)
        self.register_buffer("patch_center", torch.zeros(1, 1, config.num_prototypes))
        self.register_buffer("global_center", torch.zeros(1, config.num_prototypes))
        self._freeze_teacher()

    def _freeze_teacher(self) -> None:
        self.teacher_encoder.eval()
        self.teacher_patch_head.eval()
        self.teacher_global_head.eval()
        for module in (self.teacher_encoder, self.teacher_patch_head, self.teacher_global_head):
            for parameter in module.parameters():
                parameter.requires_grad = False

    def trainable_parameters(self):
        for module in (self.student_encoder, self.student_patch_head, self.student_global_head):
            yield from module.parameters()

    def train(self, mode: bool = True) -> "SwinIbotPretrainer":
        super().train(mode)
        self._freeze_teacher()
        return self

    @torch.no_grad()
    def update_teacher(self) -> None:
        momentum = self.config.ema_momentum
        pairs = (
            (self.student_encoder, self.teacher_encoder),
            (self.student_patch_head, self.teacher_patch_head),
            (self.student_global_head, self.teacher_global_head),
        )
        for student, teacher in pairs:
            for student_parameter, teacher_parameter in zip(student.parameters(), teacher.parameters(), strict=True):
                teacher_parameter.mul_(momentum).add_(student_parameter, alpha=1.0 - momentum)

    @torch.no_grad()
    def _update_centers(self, teacher_patch_logits: Tensor, teacher_global_logits: Tensor) -> None:
        momentum = self.config.ema_momentum
        patch_batch_center = teacher_patch_logits.mean(dim=(0, 1), keepdim=True)
        global_batch_center = teacher_global_logits.mean(dim=0, keepdim=True)
        self.patch_center.mul_(momentum).add_(patch_batch_center, alpha=1.0 - momentum)
        self.global_center.mul_(momentum).add_(global_batch_center, alpha=1.0 - momentum)

    def compute_loss(self, student_images: Tensor, teacher_images: Tensor) -> IbotLoss:
        grid_size = self.student_encoder.stage3_grid_size
        mask = make_block_mask(
            batch_size=student_images.shape[0],
            grid_size=grid_size,
            mask_ratio=self.config.mask_ratio,
            device=student_images.device,
        )
        student_stage3, student_global = self.student_encoder.forward_features(student_images, stage3_mask=mask)
        student_patch_logits = self.student_patch_head(student_stage3.flatten(1, 2))
        student_global_logits = self.student_global_head(student_global)
        with torch.no_grad():
            teacher_stage3, teacher_global = self.teacher_encoder.forward_features(teacher_images)
            teacher_patch_logits = self.teacher_patch_head(teacher_stage3.flatten(1, 2))
            teacher_global_logits = self.teacher_global_head(teacher_global)
        patch_loss = masked_token_cross_entropy(
            student_patch_logits,
            teacher_patch_logits,
            mask,
            self.config.student_temperature,
            self.config.teacher_temperature,
            self.patch_center,
        )
        global_loss = global_cross_entropy(
            student_global_logits,
            teacher_global_logits,
            self.config.student_temperature,
            self.config.teacher_temperature,
            self.global_center,
        )
        self._update_centers(teacher_patch_logits, teacher_global_logits)
        return IbotLoss(total=patch_loss + global_loss, patch=patch_loss, global_feature=global_loss)
