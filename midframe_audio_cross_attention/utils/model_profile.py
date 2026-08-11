"""Parameter and FLOPs profiling for the two-input multimodal model."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelProfile:
    total_params: int
    trainable_params: int
    flops: int

    @property
    def frozen_params(self) -> int:
        return self.total_params - self.trainable_params


def _count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def _hook_flops(model: nn.Module, waveforms: torch.Tensor, images: torch.Tensor) -> int:
    """Fallback estimate when torch.profiler cannot report operator FLOPs."""
    flops = 0
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal flops
        if not inputs:
            return
        result = output[0] if isinstance(output, tuple) else output
        if not isinstance(result, torch.Tensor):
            return
        if isinstance(module, nn.Linear):
            flops += result.numel() * (2 * module.in_features - 1)
            if module.bias is not None:
                flops += result.numel()
        elif isinstance(module, nn.Conv2d):
            kernel_height, kernel_width = module.kernel_size
            per_output = 2 * kernel_height * kernel_width * (module.in_channels // module.groups) - 1
            flops += result.numel() * per_output
            if module.bias is not None:
                flops += result.numel()

    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            handles.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            model(waveforms, images)
    finally:
        for handle in handles:
            handle.remove()
    return flops


def _estimate_flops(model: nn.Module, waveforms: torch.Tensor, images: torch.Tensor) -> int:
    try:
        with torch.inference_mode(), torch.profiler.profile(with_flops=True) as profiler:
            model(waveforms, images)
        profiled_flops = sum(event.flops for event in profiler.key_averages() if event.flops is not None)
        if profiled_flops > 0:
            return int(profiled_flops)
    except Exception as error:
        logger.warning("torch.profiler FLOPs estimation failed; using module-hook estimate: %s", error)
    return _hook_flops(model, waveforms, images)


def profile_model(model: nn.Module, waveforms: torch.Tensor, images: torch.Tensor) -> ModelProfile:
    """Profile one multimodal sample while preserving the caller's train/eval state."""
    was_training = model.training
    model.eval()
    try:
        total_params, trainable_params = _count_parameters(model)
        flops = _estimate_flops(model, waveforms, images)
    finally:
        model.train(was_training)
    return ModelProfile(total_params=total_params, trainable_params=trainable_params, flops=flops)


def log_model_profile(model: nn.Module, waveforms: torch.Tensor, images: torch.Tensor) -> ModelProfile:
    profile = profile_model(model, waveforms, images)
    logger.info("==================================================")
    logger.info("Multimodal Model Profile (one audio/video sample)")
    logger.info("  - Total Parameters:         %d (%.3f M)", profile.total_params, profile.total_params / 1e6)
    logger.info("  - Trainable Parameters:     %d (%.3f M)", profile.trainable_params, profile.trainable_params / 1e6)
    logger.info("  - Frozen Parameters:        %d (%.3f M)", profile.frozen_params, profile.frozen_params / 1e6)
    logger.info("  - Estimated FLOPs:          %d (%.3f GFLOPs)", profile.flops, profile.flops / 1e9)
    logger.info("==================================================")
    return profile
