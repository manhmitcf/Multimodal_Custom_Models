# 🏗️ STFT 256k Raw 2049 Bins + MobileNetV2 with SE 1D Channel Recalibration & FBGF Fusion

**Nhánh Git**: `exp/stft256k-raw2049-freqattn-mobilenetv2-se-fbgf`

---

## 🚀 Quick Start (Hướng dẫn Chạy Huấn luyện)

Trên Server Marimo Cloud hoặc máy trạm GPU:

```bash
cd /marimo/Multimodal_Custom_Models
git checkout exp/stft256k-raw2049-freqattn-mobilenetv2-se-fbgf
git pull origin exp/stft256k-raw2049-freqattn-mobilenetv2-se-fbgf

cd src
nohup python3 main.py > main.log 2>&1 &
tail -n +1 -f main.log
```

---

## 🛠️ Key Technical Features

1. **Audio Encoder (Raw STFT 2049 Bins 256kHz)**: Phổ âm thanh siêu phân giải 2049 Bins, bộ lọc Pre-emphasis ($\alpha=0.97$), F-Domain Attention (Mean+Std), nén qua Depthwise-Separable Audio CNN $\rightarrow$ Vector 256-dim $[B, 256]$.
2. **Video Encoder with SE-Recalibration**: MobileNetV2 trích vector đặc trưng $[B, 1280] \rightarrow$ Đi qua module **Squeeze-and-Excitation 1D Channel Recalibration (Mechanism 1)** để tự động tái hiệu chỉnh và khuếch đại các kênh chứa thông tin bọt nước/cá đớp mồi nhạy cảm, dập nén nhiễu môi trường bể nuôi.
3. **Factorized Bilinear Gated Fusion (FBGF $k=3$)**: Kết hợp phép nhân Hadamard bậc hai ($\mathbf{a}_{\text{mfb}} \odot \mathbf{v}_{\text{mfb}}$) với cổng điều phối động $g = \sigma(W \cdot [\mathbf{a}, \mathbf{v}])$.
4. **Fast ThreadPoolExecutor RAM Cache**: Nạp đa luồng $21,467$ khung hình mảng `uint8` ($3.15\text{ GB RAM}$) giúp chạy siêu tốc **20+ batches/giây (1-2s / epoch)**.
5. **HuggingFace Auto Export**: Tự động tải kết quả & checkpoint lên HuggingFace repo `manhmitcf/fish_result/stft256k_raw2049_freqattn_mobilenetv2_se_fbgf`.
