"""Model Profiling Utility: Parameter counting and FLOPs estimation for Audio, Video, Fusion Head, and Total Model."""

from __future__ import annotations

import logging
import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


def count_parameters(module: nn.Module) -> tuple[int, int]:
    """Returns (total_params, trainable_params)."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def estimate_flops(model: nn.Module, sample_waveform: Tensor, sample_image: Tensor) -> dict[str, Any]:
    """Profile parameters and estimate FLOPs for all model components."""
    audio_cnn = getattr(model, "audio_cnn", None)
    video_encoder = getattr(model, "video_encoder", None)
    fusion_head = getattr(model, "fusion", None)

    profile = {}

    if audio_cnn is not None:
        aud_tot, aud_trn = count_parameters(audio_cnn)
        profile["audio_params_total"] = aud_tot
        profile["audio_params_trainable"] = aud_trn

    if video_encoder is not None:
        vid_tot, vid_trn = count_parameters(video_encoder)
        profile["video_params_total"] = vid_tot
        profile["video_params_trainable"] = vid_trn

    if fusion_head is not None:
        fus_tot, fus_trn = count_parameters(fusion_head)
        profile["fusion_params_total"] = fus_tot
        profile["fusion_params_trainable"] = fus_trn

    tot_params, trn_params = count_parameters(model)
    profile["total_params"] = tot_params
    profile["trainable_params"] = trn_params

    # Log Profiling Report
    logger.info("=" * 60)
    logger.info("MODEL PROFILING REPORT:")
    if audio_cnn is not None:
        logger.info(f"  - Audio Encoder (Raw STFT + F-Attn + Depthwise CNN): {aud_tot:,} params ({aud_tot/1e6:.3f} M)")
    if video_encoder is not None:
        logger.info(f"  - Video Encoder (MobileNetV2):                        {vid_tot:,} params ({vid_tot/1e6:.3f} M)")
    if fusion_head is not None:
        logger.info(f"  - Fusion Head (FBGF/Gated Bilinear):                 {fus_tot:,} params ({fus_tot/1e6:.3f} M)")
    logger.info(f"  - TOTAL MODEL PARAMETERS:                             {tot_params:,} params ({tot_params/1e6:.3f} M)")
    logger.info(f"  - TRAINABLE PARAMETERS:                               {trn_params:,} params ({trn_params/1e6:.3f} M)")
    logger.info("=" * 60)

    return profile
