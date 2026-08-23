# 🏗️ STFT 256k Raw 2049 Bins Audio CNN + EfficientNetB0 Factorized Bilinear Gated Fusion (FBGF)

**Branch**: `exp/stft256k-raw2049-freqattn-efficientnetb0-fbgf`

---

## 🚀 Quick Start (Hướng dẫn Chạy Huấn luyện)

Trên Server Marimo Cloud hoặc máy trạm GPU:

```bash
cd /marimo/Multimodal_Custom_Models
git checkout exp/stft256k-raw2049-freqattn-efficientnetb0-fbgf
git pull origin exp/stft256k-raw2049-freqattn-efficientnetb0-fbgf

cd src
nohup python3 main.py > main.log 2>&1 &
tail -n +1 -f main.log
```

---

## ⚙️ CẤU HÌNH TRONG `src/config/train_config.json`

* **Audio Encoder**: Custom Raw STFT 2049 Bins ($256\text{kHz}$, Pre-Emphasis $\alpha=0.97$, F-Domain Attention).
* **Video Backbone**: **EfficientNetB0** (`video_backbone: "efficientnet_b0"`) pre-trained từ `checkpoints/EfficientNetB0_holdout_random_sample_20260804_181745/U_FFIA_video/checkpoint/efficientnet_b0/video_best.pt`.
* **Fusion Head**: Factorized Bilinear Gated Fusion (`fusion_type: "fbgf"`, $LR = 0.0001$).
* **Video Cache**: Fast `ThreadPoolExecutor` PIL/uint8 RAM cache ($3.15\text{ GB RAM}$, 5s startup).
