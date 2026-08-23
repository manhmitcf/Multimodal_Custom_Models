"""PyTest unit tests for Custom STFT 256k MobileNetV2 Multimodal Model."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytest

# Add src to sys.path
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from features.audio_features import FrequencyDomainAttention, PreEmphasisFilter, RawSTFT256kTransform
from models.custom_audio_cnn import CustomRawStftAudioCNN
from models.fusion_model import FactorizedBilinearGatedFusionHead, CustomSTFT256kMobileNetMultimodalModel


def test_pre_emphasis_filter() -> None:
    filter_mod = PreEmphasisFilter(alpha=0.97)
    waveforms = torch.randn(4, 512000)
    out = filter_mod(waveforms)
    assert out.shape == (4, 512000)


def test_raw_stft_transform() -> None:
    stft_mod = RawSTFT256kTransform(n_fft=4096, hop_length=2048, win_length=4096, windowing="hamming", alpha=0.97)
    waveforms = torch.randn(2, 512000)
    out = stft_mod(waveforms)
    # Output shape: [Batch, 1, 2049, Time_Frames]
    assert out.ndim == 4
    assert out.shape[0] == 2
    assert out.shape[1] == 1
    assert out.shape[2] == 2049


def test_frequency_domain_attention() -> None:
    f_attn = FrequencyDomainAttention(num_freq_bins=2049, use_std=True, reduction=16)
    spec = torch.randn(2, 1, 2049, 250)
    out = f_attn(spec)
    assert out.shape == (2, 1, 2049, 250)


def test_custom_raw_stft_audio_cnn() -> None:
    audio_cnn = CustomRawStftAudioCNN(feature_dim=256, n_fft=4096, hop_length=2048, win_length=4096)
    waveforms = torch.randn(2, 512000)
    out = audio_cnn(waveforms)
    assert out.shape == (2, 256)


def test_fusion_head_fbgf_and_gmf() -> None:
    for fusion_type in ("fbgf", "gmf"):
        fusion = FactorizedBilinearGatedFusionHead(audio_dim=256, video_dim=1280, d_model=256, fusion_type=fusion_type)
        audio_feat = torch.randn(2, 256)
        video_feat = torch.randn(2, 1280)
        logits = fusion(audio_feat, video_feat)
        assert logits.shape == (2, 4)
