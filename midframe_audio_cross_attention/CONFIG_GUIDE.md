# Hướng dẫn Chi tiết Cấu hình File JSON (CONFIG_GUIDE)

Tài liệu này giải thích chi tiết ý nghĩa từng thông số trong file cấu hình JSON [`config/train_config.json`](file:///C:/Users/manhm/Desktop/Multimodal_Custom_Models/midframe_audio_cross_attention/config/train_config.json) và [`config/artifact_upload_config.json`](file:///C:/Users/manhm/Desktop/Multimodal_Custom_Models/midframe_audio_cross_attention/config/artifact_upload_config.json), giúp bạn dễ dàng chỉnh sửa và tùy biến cho các kịch bản thử nghiệm khác nhau.

---

## 📄 1. File Cấu hình Chính: `config/train_config.json`

File cấu hình được chia thành 7 khối thông số chính:

```json
{
  "references": { ... },
  "checkpoints": { ... },
  "data": { ... },
  "model": { ... },
  "training": { ... },
  "ibot_pretraining": { ... },
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
* **`output_dir`**: Đường dẫn thư mục xuất kết quả checkpoints và logs (Mặc định `"checkpoint"` hoặc `"../runs/..."`).

---

### 🔹 Khối `ibot_pretraining` (Tiền huấn luyện Tự giám sát SSL trên tập Train)
* **`enabled`**: `true` để bật bước iBOT SSL pretraining không nhãn trên khung hình giữa của tập train; `false` để bỏ qua.
* **`epochs`**: Số epoch tiền huấn luyện iBOT (Mặc định `50`).
* **`batch_size`**: Kích thước batch iBOT (Mặc định `32`).
* **`learning_rate`**: Tốc độ học iBOT (Mặc định `0.00005`).
* **`mask_ratio`**: Tỉ lệ che các ô không gian (Mặc định `0.4` tức che 40% ô vuông).
* **`num_prototypes`**: Số lượng cluster prototypes ở đầu classifier iBOT (Mặc định `2048`).
* **`output_dir`**: Thư mục lưu checkpoint iBOT.

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
  "zip_path": "checkpoint/gw_avf_multimodal_results.zip",
  "repo_id": "manhmitcf/fish_result",
  "repo_type": "dataset",
  "path_in_repo": "gw_avf_multimodal_results.zip",
  "create_repo": true
}
```

* **`enabled`**: Set `true` để bật nạp tự động artifact zip sau khi train xong.
* **`source_dir`**: Thư mục chứa toàn bộ file đầu ra cần đóng gói.
* **`repo_id`**: Địa chỉ Hugging Face Dataset repo: `"manhmitcf/fish_result"`.

---

## 💡 Hướng dẫn Tùy chỉnh Nhanh cho Các Kịch bản Thường gặp

### 1. Nếu bị tràn bộ nhớ GPU (CUDA Out of Memory)
Hãy chỉnh giảm `batch_size` trong `train_config.json`:
```json
"training": {
  "batch_size": 8
},
"ibot_pretraining": {
  "batch_size": 16
}
```

### 2. Nếu muốn chạy Fine-Tune sâu hơn các Encoder
Chuyển `encoder_mode` từ `"frozen"` sang `"tune"`:
```json
"model": {
  "encoder_mode": "tune"
}
```

### 3. Nếu muốn đổi sang file config khác khi chạy
Bạn có thể truyền cờ `--config` khi gọi lệnh `main.py`:
```bash
python main.py --config config/train_config_03_tune_gentle.json
```
