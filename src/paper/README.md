# STFT 256k Raw 2049 Bins with Frequency-Domain Attention & MobileNetV2 Factorized Bilinear Gated Fusion

## 1. Abstract
This paper introduces an end-to-end multimodal deep learning architecture designed for automated fish feeding intensity assessment on the U-FFIA27K dataset (27,000 paired audio-visual samples).

Key innovations:
1. **Pre-Emphasis High-Pass Filter ($\alpha=0.97$)**: Boosts high-frequency acoustic splash signals ($2\text{ kHz} - 8\text{ kHz}$) of fish snapping at pellets while attenuating low-frequency water pump motor noise ($0 - 500\text{ Hz}$).
2. **Raw STFT 2049 Bins Spectrogram (No Mel Compression, No RGB Encoding)**: Computes high-resolution $2049$ frequency bins ($\Delta f = 62.5\text{ Hz}$) with Hamming windowing ($4096, 2048$), preserving exact float32 acoustic precision without lossy Mel filterbank compression or 8-bit image rendering.
3. **Frequency-Domain Attention (F-Attention)**: Dynamically profiles temporal energy using `Mean` and `Std` across 2049 frequency bins, learning an adaptive spectral gate $\mathbf{a}_F \in [0, 1]^{2049}$ to highlight feeding bands and suppress ambient noise.
4. **Depthwise-Separable Audio CNN**: Employs an early strided convolution (`stride=(4, 2)`) to shrink frequency dimensions $2049 \rightarrow 513$ instantly, followed by depthwise-separable blocks to extract a 256d audio feature vector with minimal parameter overhead (~1.4M params).
5. **Factorized Bilinear Gated Fusion (FBGF / GMF)**: Fuses 256d Audio features with MobileNetV2 Video features using low-rank Multi-level Factorized Bilinear pooling ($k=3$), power & L2 normalization, and dynamic gated routing.

## 2. Experimental Benchmark Results
- Target Dataset: Full U-FFIA27K (27,000 samples)
- Baseline Macro-F1: 85.24%
- True SOTA Benchmark: 95.40%
- Target Macro-F1: **>95.93%**
- Total Parameters: **~5.1M**
