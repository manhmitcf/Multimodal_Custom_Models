# 🏗️ Spatial Cross-Attention Multimodal Model (PANNS CNN6 + MobileNetV2)

**Branch**: `exp/midframe-to-audio-cross-attn`

---

## 🚀 Quick Start (Hướng dẫn Chạy Huấn luyện)

Trên Server Marimo Cloud hoặc máy trạm GPU:

```bash
cd /marimo/Multimodal_Custom_Models
git checkout exp/midframe-to-audio-cross-attn
git pull origin exp/midframe-to-audio-cross-attn

cd src
nohup python3 main.py > main.log 2>&1 &
tail -n +1 -f main.log
```

---

## ⚙️ CẤU HÌNH CHI TIẾT TRONG `src/config/train_config.json`

Toàn bộ mô hình có thể tùy chỉnh linh hoạt qua file `src/config/train_config.json`:

```json
{
  "references": {
    "audio_repo": "../U_FFIA27K_audio",
    "video_repo": "../U_FFIA27K_video"
  },
  "checkpoints": {
    "audio": "../checkpoints/PANNS_Cnn6_holdout_random_sample_20260729_153012/DL_audio/checkpoint/panns_cnn6/audio_best.pt",
    "video_by_backbone": {
      "mobilenet_v2": "../checkpoints/MobileNetV2_holdout_random_sample_20260729_153012/DL_video/checkpoint/mobilenet_v2/video_best.pt"
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
    "video_backbone": "mobilenet_v2",
    "encoder_mode": "frozen",
    "d_model": 256,
    "num_heads": 4,
    "dropout": 0.1
  },
  "training": {
    "batch_size": 64,
    "epochs": 100,
    "fusion_learning_rate": 0.001,
    "encoder_learning_rate": 0.001,
    "weight_decay": 0.0001,
    "seed": 42,
    "device": "auto",
    "output_dir": "checkpoint"
  },
  "results_upload": {
    "enabled": true,
    "repo_id": "manhmitcf/fish_result",
    "repo_type": "dataset",
    "path_prefix": "midframe_audio_cross_attention",
    "create_repo": true
  }
}
```

---

## 🛠️ Key Technical Features

1. **PANNS CNN6 Audio Token Encoder**: Trích xuất 6 tokens đặc trưng âm thanh $[B, 6, 512]$ từ phổ Mel-Spectrogram ($64\text{kHz}$).
2. **MobileNetV2 Video Encoder**: Trích xuất 1 vector đặc trưng thị giác 1280-dim $[B, 1280]$ từ khung hình RGB trung tâm ($224 \times 224$).
3. **Spatial Cross-Attention (`CrossAttentionHead`)**: Vector Query của Video ($1280 \rightarrow 256$) tự động đóng vai trò người hỏi để chú ý (Cross-Attend) tới 6 tokens Âm thanh ($512 \rightarrow 256$ + Positional Embedding + 1-layer TransformerEncoderLayer Self-Attention).
4. **Fast ThreadPoolExecutor RAM Cache**: Nạp đa luồng $21,467$ khung hình mảng `uint8` ($3.15\text{ GB RAM}$) giúp chạy siêu tốc **20+ batches/giây (1-2s / epoch)**.
5. **Auto Checkpointing & HF Export**: Tự động lưu `best_model.pt` theo Validation Macro-F1 và đẩy kết quả lên HuggingFace dataset repo `manhmitcf/fish_result/midframe_audio_cross_attention`.
