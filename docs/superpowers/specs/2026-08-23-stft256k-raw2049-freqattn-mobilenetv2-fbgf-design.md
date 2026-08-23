# Design Specification: STFT 256k Raw 2049 Bins with Frequency-Domain Attention & MobileNetV2 Factorized Bilinear Gated Fusion

**Branch**: `exp/stft256k-raw2049-freqattn-mobilenetv2-fbgf`  
**Date**: 2026-08-23  
**Target Dataset**: U-FFIA27K (27,000 paired audio-visual aquaculture samples)  
**Primary Goal**: Achieve >95.9% Macro-F1 with an ultra-lightweight (~5.1M params) end-to-end model using Raw STFT 2049 Bins (No Mel, No RGB encoding), Pre-Emphasis filtering (alpha=0.97), Frequency-Domain Attention (F-Attention), Depthwise-Separable Audio CNN, MobileNetV2 visual backbone, and Factorized Bilinear Gated Fusion (FBGF / GMF).

---

## 1. Executive Architecture Summary

```text
       RAW AUDIO WAVEFORM 2s (SR 256 kHz)                                     MIDDLE FRAME RGB (224x224)
                   │                                                                      │
       Pre-Emphasis Filter (alpha=0.97)                                               MobileNetV2
                   │                                                               (Feature 1280d)
       Raw STFT 256k (4096, 2048, 2048)                                                       │
       (NO MEL COMPRESSION - NO RGB ENCODING)                                        Linear Projection
                   │                                                                (1280 -> 256d)
       Spectrogram dB [B, 1, 2049, 250]                                                       │
                   │                                                                Video Feature (256d)
       [FREQUENCY-DOMAIN ATTENTION (F-ATTN)]                                              │
       (Learns attention weights a_F in [0, 1]^2049)                                      │
                   │                                                                      │
       [DEPTHWISE-SEPARABLE AUDIO CNN]                                                    │
       (Early Strided Conv 4x2 -> Depthwise Blocks)                                       │
                   │                                                                      │
         Audio Feature (256d)                                                             │
                   │                                                                      │
                   └───────────────────────────┬──────────────────────────────────────────┘
                                               │
                                [FACTORIZED BILINEAR GATED FUSION]
                                 1. MFB Bilinear Hadamard Product (k=3)
                                 2. Power & L2 Normalization
                                 3. Dynamic Gating Gate g = Sigmoid(W[x_a, x_v])
                                               │
                                     [CLASSIFIER 4 CLASSES]
                                  Linear(256 -> 4) -> Logits
```

---

## 2. Mathematical Component Specifications

### 2.1 Audio Processing Pipeline (`src/features/audio_features.py`)
1. **Pre-Emphasis High-Pass Filter**:
   $$y[t] = x[t] - 0.97 \cdot x[t-1]$$
   Boosts high-frequency acoustics ($2\text{ kHz} - 8\text{ kHz}$) of feeding splashes and attenuates low-frequency water pump motor noise ($0 - 500\text{ Hz}$).

2. **Raw STFT Transform (No Mel Compression)**:
   - `n_fft = 4096`
   - `win_length = 4096`
   - `hop_length = 2048`
   - `window = "hamming"`
   - Computes complex STFT $\mathbf{Z} \in \mathbb{C}^{B \times 2049 \times 250}$.
   - Computes Log-Power Spectrogram in dB scale: $\mathbf{S} = 10 \cdot \log_{10}(|\mathbf{Z}|^2 + 10^{-10}) \in \mathbb{R}^{B \times 1 \times 2049 \times 250}$.

3. **Frequency-Domain Attention (F-Attention Module)**:
   - Collapses the time axis $T=250$ via `Mean` and `Std` energy profiling:
     $$\mathbf{v}_F = \left[ \text{Mean}_T(\mathbf{S}) \,\|\, \text{Std}_T(\mathbf{S}) \right] \in \mathbb{R}^{B \times 4098}$$
   - Pass through 2-layer MLP with Sigmoid activation:
     $$\mathbf{a}_F = \sigma \left( \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \mathbf{v}_F) \right) \in [0, 1]^{B \times 1 \times 2049 \times 1}$$
   - Apply frequency-wise attention weights:
     $$\mathbf{S}_{\text{attended}} = \mathbf{a}_F \odot \mathbf{S}$$

### 2.2 Depthwise-Separable Audio CNN (`src/models/custom_audio_cnn.py`)
- **Block 1 (Early Strided Conv)**: `Conv2d(1 -> 32, kernel=(5, 5), stride=(4, 2), padding=2)`
  - Reduces frequency dimension from $2049 \rightarrow 513$ instantly to prevent VRAM inflation.
- **Block 2**: `DepthwiseSeparableConv2d(32 -> 64, stride=(2, 2))` $\rightarrow [B, 64, 257, 63]$.
- **Block 3**: `DepthwiseSeparableConv2d(64 -> 128, stride=(2, 2))` $\rightarrow [B, 128, 129, 32]$.
- **Block 4**: `DepthwiseSeparableConv2d(128 -> 256, stride=(2, 2))` $\rightarrow [B, 256, 65, 16]$.
- **Global Pooling**: `AdaptiveAvgPool2d((1, 1))` $\rightarrow$ Audio Vector $\mathbf{x}_a \in \mathbb{R}^{B \times 256}$. Total params ~1.4M.

### 2.3 Visual Encoder (`MobileNetV2`)
- Input: Middle RGB Video Frame $[B, 3, 224, 224]$.
- Backbone: `MobileNetV2` ($\sim 3.5\text{M}$ params).
- Linear Projection: `Linear(1280 -> 256)` $\rightarrow$ Video Vector $\mathbf{x}_v \in \mathbb{R}^{B \times 256}$.

### 2.4 Multimodal Fusion Engine (`src/models/fusion_model.py`)
Configurable via `"fusion_type": "fbgf"` (default) or `"fusion_type": "gmf"`:

#### Mode 1: Factorized Bilinear Gated Fusion (FBGF)
1. **Low-Rank Bilinear Factorization ($k=3$)**:
   $$\tilde{\mathbf{x}}_a = \mathbf{W}_a \mathbf{x}_a \in \mathbb{R}^{256 \times 3}, \quad \tilde{\mathbf{x}}_v = \mathbf{W}_v \mathbf{x}_v \in \mathbb{R}^{256 \times 3}$$
   $$\mathbf{z} = \text{SumPool}_3 \Big( \tilde{\mathbf{x}}_a \circ \tilde{\mathbf{x}}_v \Big) \in \mathbb{R}^{256}$$
2. **Power & L2 Normalization**:
   $$\mathbf{z}_{\text{norm}} = \text{L2Norm} \left( \text{Sign}(\mathbf{z}) \odot \sqrt{|\mathbf{z}| + 1e-10} \right)$$
3. **Dynamic Gating Gate**:
   $$\mathbf{g} = \sigma \left( \mathbf{W}_g [\mathbf{x}_a \,\|\, \mathbf{x}_v] + \mathbf{b}_g \right)$$
4. **Fused Feature Representation**:
   $$\mathbf{h}_{\text{fused}} = \mathbf{g} \odot \mathbf{z}_{\text{norm}} + (\mathbf{1} - \mathbf{g}) \odot \text{ReLU}(\mathbf{W}_v' \mathbf{x}_v)$$

#### Mode 2: Gated Multimodal Fusion (GMF)
$$\mathbf{g} = \sigma \left( \mathbf{W}_g [\mathbf{x}_a \,\|\, \mathbf{x}_v] + \mathbf{b}_g \right)$$
$$\mathbf{h}_{\text{fused}} = \mathbf{g} \odot \text{ReLU}(\mathbf{W}_a \mathbf{x}_a) + (\mathbf{1} - \mathbf{g}) \odot \text{ReLU}(\mathbf{W}_v \mathbf{x}_v)$$

### 2.5 Classifier
$$\text{Logits} = \text{Linear}(256 \rightarrow 4) \in \mathbb{R}^{B \times 4}$$

---

## 3. Directory Layout & Standards

All code is strictly placed inside `src/`:
- `src/main.py`: Single entry point.
- `src/settings.py`: Configuration & path resolver.
- `src/config/train_config.json`: Configuration specifying `stft` parameters (`sr: 256000`, `pre_emphasis: 0.97`, `frame_length: 4096`, `hop_length: 2048`, `n_fft: 4096`, `windowing: "hamming"`, `use_std: true`) and `"fusion_type": "fbgf"`.
- `src/config/artifact_upload_config.json`: Hugging Face auto-upload config for repo `manhmitcf/fish_result`.
- `src/dataset/paired_loader.py`: RAM dataset loader supporting `cache_audio: true` and `video_cache_mode: "ram"`.
- `src/features/audio_features.py`: Pre-Emphasis, Raw STFT 2049, FrequencyDomainAttention.
- `src/models/custom_audio_cnn.py`: Depthwise-Separable Audio CNN.
- `src/models/fusion_model.py`: Multimodal Model & Fusion Head.
- `src/tasks/trainer.py`: MultimodalTrainer saving all standard evaluation metrics to `src/checkpoint/`.
- `src/paper/README.md`: Research paper documentation.
- `src/tests/test_custom_multimodal_model.py`: PyTest unit tests.

---

## 4. Verification & Testing Plan

1. **Unit Test Execution**:
   - Run `python -m pytest tests/` in `src/` to verify forward pass shapes, gradient flow, and parameter counts.
2. **Output Checkpoint Standard**:
   - Confirm `src/checkpoint/` generates `splits/`, `best.pt`, `history.csv`, `summary_results.csv`, `best_val_metrics.json`, `test_metrics.json`, `best_val_confusion_matrix.csv`, and `test_confusion_matrix.csv`.
3. **Artifact Upload**:
   - Zip `STFT256k_Raw2049_FreqAttn_MobileNetV2_Artifacts.zip` and upload to Hugging Face dataset repo `manhmitcf/fish_result`.
