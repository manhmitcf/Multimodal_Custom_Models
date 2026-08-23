# Hướng dẫn Chi tiết Cấu hình File JSON (CONFIG_GUIDE)

Tài liệu này giải thích chi tiết ý nghĩa từng thông số trong file cấu hình JSON [`config/train_config.json`](file:///C:/Users/manhm/Desktop/Multimodal_Custom_Models/src/config/train_config.json) và [`config/artifact_upload_config.json`](file:///C:/Users/manhm/Desktop/Multimodal_Custom_Models/src/config/artifact_upload_config.json), giúp bạn dễ dàng chỉnh sửa và tùy biến cho các kịch bản thử nghiệm khác nhau.

---

## 📄 1. File Cấu hình Chính: `config/train_config.json`

File cấu hình được chia thành 6 khối thông số chính:

```json
{
  "references": { ... },
  "checkpoints": { ... },
  "data": { ... },
  "model": { ... },
  "training": { ... },
  "results_upload": { ... }
}
```

### 🔹 Khối `references` (Đường dẫn dự án nguồn)
* **`audio_repo`**: Đường dẫn tương đối/tuyệt đối tới project baseline audio (Ví dụ: `"../../U_FFIA27K_audio"`).
* **`video_repo`**: Đường dẫn tương đối/tuyệt đối tới project baseline video (Ví dụ: `"../../U_FFIA27K_video"`).

---

### 🔹 Khối `checkpoints` (Trọng số mô hình Pretrained Baseline)
* **`audio`**: Đường dẫn file trọng số `audio_best.pt` của mô hình Audio (PANNS Cnn6).
* **`video_by_backbone`**: Từ điển ánh xạ từng loại visual backbone tới file trọng số `video_best.pt` tương ứng:
  * `"swin_tiny"`: Đường dẫn file `video_best.pt` SwinTiny.
  * `"densenet121"`: Đường dẫn file `video_best.pt` DenseNet121.
  * `"efficientnet_b0"`: Đường dẫn file `video_best.pt` EfficientNet-B0.
  * `"mobilenet_v2"`: Đường dẫn file `video_best.pt` MobileNetV2.

---

### 🔹 Khối `data` (Quản lý Dữ liệu & DataLoader)
* **`split_dir`**: Đường dẫn thư mục chứa 3 file CSV cố định (`train.csv`, `val.csv`, `test.csv`).
* **`cache_audio`**: `true` hoặc `false`. Bật/tắt nạp sẵn Audio Mel-spectrogram vào bộ nhớ RAM để tăng tốc nạp dữ liệu.
* **`video_cache_mode`**: `"ram"` (Nạp ảnh vào RAM), `"disk"` (Lưu cache trên đĩa), hoặc `"none"` (Đọc từ đĩa trực tiếp).
* **`num_workers`**: Số luồng CPU nạp dữ liệu. Đặt `-1` để hệ thống tự động tính số luồng tối ưu theo CPU.
* **`image_size`**: Kích thước khung hình đầu vào (Mặc định `224` tương ứng ảnh $224 \times 224$ px).

---

### 🔹 Khối `model` (Kiến trúc Mô hình & Cross-Attention)
* **`video_backbone`**: Chọn tên visual backbone (`"swin_tiny"`, `"densenet121"`, `"efficientnet_b0"`, `"mobilenet_v2"`).
* **`encoder_mode`**:
  * `"frozen"`: Đông đóng toàn bộ tham số của Audio Encoder & Visual Encoder, chỉ huấn luyện các lớp Cross-Attention Fusion (Khuyên dùng khi GPU RAM hạn chế).
  * `"tune"`: Mở cho phép fine-tune các lớp cuối của Audio & Visual Encoders với `encoder_learning_rate` nhỏ.
* **`d_model`**: Số chiều embedding ẩn của Cross-Attention (Mặc định `256` hoặc `512`).
* **`num_heads`**: Số lượng đầu chú ý (Attention Heads) trong Multi-head Attention (Mặc định `4` hoặc `8`).
* **`dropout`**: Tỉ lệ dropout chống overfitting (Mặc định `0.1`).

---

### 🔹 Khối `training` (Siêu tham số Huấn luyện)
* **`batch_size`**: Kích thước mẫu theo batch cho huấn luyện có giám sát (Mặc định `16` hoặc `32`).
* **`epochs`**: Số epoch tối đa (Mặc định `100`).
* **`fusion_learning_rate`**: Tốc độ học dành riêng cho các lớp Fusion (Mặc định `0.001` / `1e-3`).
* **`encoder_learning_rate`**: Tốc độ học khi fine-tune encoder (Mặc định `0.00001` / `1e-5`).
* **`weight_decay`**: Hệ số suy giảm trọng số L2 regularization (Mặc định `0.0001`).
* **`seed`**: Hạt giống ngẫu nhiên để đảm bảo tính tái lập kết quả (Mặc định `42`).
* **`device`**: `"auto"` (Tự động dùng GPU nếu có, nếu không chuyển CPU), `"cuda"`, hoặc `"cpu"`.
* **`output_dir`**: Đường dẫn thư mục xuất kết quả checkpoints và logs (Mặc định `"checkpoint"`).

---

### 🔹 Khối `results_upload` (Tự động Đẩy CSV Metrics lên Hugging Face)
* **`enabled`**: `true` hoặc `false`.
* **`repo_id`**: Tên repo trên Hugging Face (Mặc định `"manhmitcf/fish_result"`).
* **`repo_type`**: Loại repo (Luôn là `"dataset"`).
* **`create_repo`**: `true` để tự tạo repo trên Hugging Face nếu chưa có.

---

## 📄 2. File Cấu hình Upload Artifact Zip: `config/artifact_upload_config.json`

File này quản lý việc nén **toàn bộ thư mục kết quả** (`checkpoint/`) thành file `.zip` và tải lên Hugging Face:

```json
{
  "enabled": true,
  "source_dir": "checkpoint",
  "zip_path": "checkpoint/GW_AVF_Multimodal_Artifacts.zip",
  "repo_id": "manhmitcf/fish_result",
  "repo_type": "dataset",
  "path_in_repo": "GW_AVF_Multimodal_Artifacts.zip",
  "create_repo": true
}
```

---

## 💡 Hướng dẫn Tùy chỉnh Nhanh cho Các Kịch bản Thường gặp

### 1. Nếu bị tràn bộ nhớ GPU (CUDA Out of Memory)
Hãy chỉnh giảm `batch_size` trong `train_config.json`:
```json
"training": {
  "batch_size": 8
}
```

### 2. Nếu muốn chạy Fine-Tune sâu hơn các Encoder
Chuyển `encoder_mode` từ `"frozen"` sang `"tune"`:
```json
"model": {
  "encoder_mode": "tune"
}
```

### 3. Khởi chạy Pipeline
```bash
cd src
python main.py
```
