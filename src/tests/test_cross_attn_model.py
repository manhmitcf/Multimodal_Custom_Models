"""PyTest unit tests for Spatial Cross-Attention Multimodal Model (`exp/midframe-to-audio-cross-attn`)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytest

src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from models.fusion_model import CrossAttentionHead


def test_cross_attention_head() -> None:
    fusion_head = CrossAttentionHead(
        audio_dim=512,
        video_dim=1280,
        d_model=256,
        num_heads=4,
        dropout=0.1,
    )
    audio_tokens = torch.randn(4, 6, 512) # [Batch, 6 tokens, 512d]
    video_feature = torch.randn(4, 1280) # [Batch, 1280d]

    logits, attention = fusion_head(audio_tokens, video_feature)

    assert logits.shape == (4, 4) # 4 classes
    assert attention.shape == (4, 4, 1, 6) # MultiheadAttention weights [Batch, num_heads, Query_len=1, Key_len=6]
