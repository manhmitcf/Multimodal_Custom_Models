# STFT 256k Raw 2049 Bins with Frequency-Domain Attention & MobileNetV2 Factorized Bilinear Gated Fusion

**Branch**: `exp/stft256k-raw2049-freqattn-mobilenetv2-fbgf`

## 🌟 Quick Start Guide

To run the training and testing pipeline on Marimo Cloud Server or local GPU environment:

```bash
cd src
python main.py
```

## 🛠️ Key Features
1. **Pre-Emphasis Filter ($\alpha=0.97$)**: Amplifies high-frequency acoustic splash signals ($2\text{ kHz} - 8\text{ kHz}$) and attenuates low-frequency motor noise ($0 - 500\text{ Hz}$).
2. **Raw STFT 2049 Bins (No Mel, No RGB)**: Computes high-resolution $2049$ frequency bins ($\Delta f = 62.5\text{ Hz}$) using Hamming windowing ($4096, 2048$), feeding raw float32 magnitude tensors directly to Conv2d layers.
3. **Frequency-Domain Attention (F-Attention)**: Profiles temporal energy via `Mean` & `Std` across 2049 frequency bins, learning adaptive spectral attention weights $\mathbf{a}_F \in [0, 1]^{2049}$.
4. **Depthwise-Separable Audio CNN**: Uses early strided convolution (`stride=(4, 2)`) to shrink frequency dimensions $2049 \rightarrow 513$ instantly, followed by depthwise-separable blocks to extract a 256d audio feature vector (~1.4M params).
5. **Factorized Bilinear Gated Fusion (FBGF / GMF)**: Fuses 256d Audio with MobileNetV2 Video features using low-rank Multi-level Factorized Bilinear pooling ($k=3$), power & L2 normalization, and dynamic gated routing.
6. **Optimized DataLoader**: Uses `cache_audio: true`, `video_cache_mode: "ram"`, `pin_memory = True`, `persistent_workers = True`, and dynamic worker calculation `num_workers = max_cpu_cores // 2 + 1`.
