# ⚙️ CONFIGURATION GUIDE: `src/config/train_config.json`

File `src/config/train_config.json` điều khiển toàn bộ quá trình huấn luyện và đánh giá của mô hình **Spatial Cross-Attention Multimodal Model** (`exp/midframe-to-audio-cross-attn`).

---

## 📌 1. Các phần Cấu hình Chính

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

## 📷 2. Chi tiết Tối ưu RAM & Tốc độ (`"data"`)

* **`"video_cache_mode": "ram"`**: Nạp trước khung hình trung tâm dưới dạng mảng `uint8` (`[3, 224, 224]`) đa luồng bằng `ThreadPoolExecutor` (chỉ tốn **`3.15 GB RAM`**). Giúp tăng tốc huấn luyện lên **20+ batches/giây (1-2s / epoch)**.
* **`"num_workers": -1`**: Tự động tính số lượng workers tối ưu dựa trên số nhân CPU của hệ thống (`os.cpu_count() // 2 + 1`).

---

## 🔀 3. Cấu hình Mô hình & Bảo vệ BatchNorm (`"model"`)

* **`"video_backbone": "mobilenet_v2"`**: Backbone hình ảnh MobileNetV2 pre-trained ($92\%$ baseline).
* **`"encoder_mode": "frozen"`**: Đóng băng trọng số và ghim chặt `self.video_encoder.eval()` & `self.audio_encoder.eval()` nhằm giữ nguyên $100\%$ thống kê 52 lớp BatchNorm.
* **`"d_model": 256`**: Chiều ẩn chung của không gian Cross-Attention.
* **`"num_heads": 4`**: Số lượng attention heads cho Multihead Attention.
