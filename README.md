# 🏗️ STFT 256k Raw 2049 Bins Audio CNN + EfficientNetB0 Factorized Bilinear Gated Fusion (FBGF)

**Nhánh Git**: `exp/stft256k-raw2049-freqattn-efficientnetb0-fbgf`

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

## ⚙️ CẤU HÌNH CHI TIẾT TRONG `src/config/train_config.json`

Mô hình được điều khiển qua tệp [`src/config/train_config.json`](file:///C:/Users/manhm/Desktop/Multimodal_Custom_Models/src/config/train_config.json):

```json
{
  "stft": {
    "sr": 256000,
    "pre_emphasis": 0.97,
    "frame_length": 4096,
    "hop_length": 2048,
    "n_fft": 4096,
    "windowing": "hamming",
    "use_std": true
  },
  "references": {
    "audio_repo": "../U_FFIA27K_audio",
    "video_repo": "../U_FFIA27K_video"
  },
  "checkpoints": {
    "audio": "../checkpoints/PANNS_Cnn6_holdout_random_sample_20260729_153012/DL_audio/checkpoint/panns_cnn6/audio_best.pt",
    "video_by_backbone": {
      "mobilenet_v2": "../checkpoints/MobileNetV2_holdout_random_sample_20260729_153012/DL_video/checkpoint/mobilenet_v2/video_best.pt",
      "efficientnet_b0": "../checkpoints/EfficientNetB0_holdout_random_sample_20260804_181745/U_FFIA_video/checkpoint/efficientnet_b0/video_best.pt"
    }
  },
  "data": {
    "split_dir": "../checkpoints/PANNS_Cnn6_holdout_random_sample_20260729_153012/DL_audio/checkpoint/panns_cnn6/splits",
    "cache_audio": false,
    "video_cache_mode": "ram",
    "num_workers": -1,
    "image_size": 224
  },
  "model": {
    "video_backbone": "efficientnet_b0",
    "encoder_mode": "frozen",
    "d_model": 256,
    "dropout": 0.1,
    "fusion_type": "fbgf"
  },
  "training": {
    "batch_size": 64,
    "epochs": 100,
    "fusion_learning_rate": 0.0001,
    "encoder_learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "seed": 42,
    "device": "auto",
    "output_dir": "checkpoint"
  },
  "results_upload": {
    "enabled": true,
    "repo_id": "manhmitcf/fish_result",
    "repo_type": "dataset",
    "path_prefix": "stft256k_raw2049_freqattn_efficientnetb0_fbgf",
    "create_repo": true
  }
}
```

---

## 🛠️ Key Technical Features

1. **Audio Encoder (Raw STFT 2049 Bins 256kHz)**: Phổ âm thanh siêu phân giải 2049 Bins, bộ lọc Pre-emphasis ($\alpha=0.97$), F-Domain Attention (Mean+Std), nén qua Depthwise-Separable Audio CNN $\rightarrow$ Vector 256-dim $[B, 256]$.
2. **Video Encoder (EfficientNetB0)**: Sử dụng **EfficientNetB0** pre-trained từ checkpoint `EfficientNetB0_holdout_random_sample_20260804_181745` trích xuất 1280-dim $[B, 1280]$.
3. **Factorized Bilinear Gated Fusion (FBGF $k=3$)**: Kết hợp phép nhân Hadamard bậc hai ($\mathbf{a}_{\text{mfb}} \odot \mathbf{v}_{\text{mfb}}$) với cổng điều phối động $g = \sigma(W \cdot [\mathbf{a}, \mathbf{v}])$.
4. **Fast ThreadPoolExecutor RAM Cache**: Nạp đa luồng $21,467$ khung hình mảng `uint8` ($3.15\text{ GB RAM}$) giúp chạy siêu tốc **20+ batches/giây (1-2s / epoch)**.
5. **HuggingFace Auto Export**: Tự động tải kết quả & checkpoint lên HuggingFace repo `manhmitcf/fish_result/stft256k_raw2049_freqattn_efficientnetb0_fbgf`.
