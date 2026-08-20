"""Unit contracts for Swin iBOT-inspired spatial pretraining.

Run on Marimo from ``midframe_audio_cross_attention`` with:
``python -m pytest tests/test_ibot_components.py -v``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from models.ibot_pretraining import make_block_mask, masked_token_cross_entropy
from models.fusion_model import SpatialCrossAttentionHead
from settings import IbotPretrainingConfig
from utils.huggingface_results import result_path_in_repo


def test_block_mask_marks_the_requested_number_of_stage3_tokens() -> None:
    """Would fail if mask construction changes its configured mask ratio."""
    generator = torch.Generator().manual_seed(7)

    mask = make_block_mask(batch_size=2, grid_size=4, mask_ratio=0.5, generator=generator)

    assert mask.shape == (2, 16)
    assert mask.dtype is torch.bool
    assert mask.sum(dim=1).tolist() == [8, 8]


def test_masked_token_loss_ignores_unmasked_student_predictions() -> None:
    """Would fail if a changed unmasked token can affect the iBOT patch loss."""
    teacher_logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    student_logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]], requires_grad=True)
    mask = torch.tensor([[True, False]])

    reference_loss = masked_token_cross_entropy(
        student_logits,
        teacher_logits,
        mask,
        student_temperature=1.0,
        teacher_temperature=1.0,
        center=torch.zeros(1, 1, 2),
    )
    changed_unmasked = student_logits.detach().clone()
    changed_unmasked[:, 1] = torch.tensor([50.0, -50.0])
    changed_loss = masked_token_cross_entropy(
        changed_unmasked,
        teacher_logits,
        mask,
        student_temperature=1.0,
        teacher_temperature=1.0,
        center=torch.zeros(1, 1, 2),
    )

    assert torch.isfinite(reference_loss)
    assert torch.allclose(reference_loss, changed_loss)


def test_ibot_config_rejects_a_non_swin_backbone() -> None:
    """Would fail if iBOT pretraining is accidentally enabled for unsupported encoders."""
    config = IbotPretrainingConfig(
        enabled=True,
        epochs=1,
        batch_size=2,
        learning_rate=1e-4,
        weight_decay=0.04,
        ema_momentum=0.996,
        mask_ratio=0.4,
        num_prototypes=32,
        student_temperature=0.1,
        teacher_temperature=0.07,
        output_dir=Path("runs/test"),
    )

    with pytest.raises(ValueError, match="SwinTiny"):
        config.validate(video_backbone="mobilenet_v2")


def test_spatial_fusion_classifies_each_visual_token_set() -> None:
    """Would fail if spatial fusion drops the batch or stops returning four logits."""
    head = SpatialCrossAttentionHead(audio_dim=5, video_dim=7, d_model=8, num_heads=2, dropout=0.0)

    logits, attention = head(torch.randn(3, 6, 5), torch.randn(3, 4, 7))

    assert logits.shape == (3, 4)
    assert attention.shape == (3, 2, 4, 6)


def test_result_upload_path_keeps_runs_and_timestamps_separate() -> None:
    """Would fail if separate Marimo runs overwrite a previously uploaded CSV."""
    destination = result_path_in_repo(
        path_prefix="swin_tiny_ibot_spatial",
        run_name="02_frozen_standard",
        timestamp="20260821_153012",
        filename="test_confusion_matrix.csv",
    )

    assert destination == "swin_tiny_ibot_spatial/02_frozen_standard/20260821_153012/test_confusion_matrix.csv"
