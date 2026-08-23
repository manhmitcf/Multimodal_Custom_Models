# Hướng dẫn Chi tiết Cấu hình File JSON (CONFIG_GUIDE)

Tài liệu này giải thích chi tiết ý nghĩa từng thông số trong file cấu hình JSON [`config/train_config.json`](file:///C:/Users/manhm/Desktop/Multimodal_Custom_Models/src/config/train_config.json) và [`config/artifact_upload_config.json`](file:///C:/Users/manhm/Desktop/Multimodal_Custom_Models/src/config/artifact_upload_config.json), giúp bạn dễ dàng chỉnh sửa và tùy biến cho các kịch bản thử nghiệm khác nhau.

---

## 📄 1. File Cấu hình Chính: `config/train_config.json`

File cấu hình được chia thành 6 khối thông số chính:

```json
{
  "stft": { ... },
  "references": { ... },
  "checkpoints": { ... },
  "data": { ... },
  "model": { ... },
  "training": { ... },
  "results_upload": { ... }
}
```

### 🔹 Khối `stft` (Siêu tham số Lọc & Phép biến đổi STFT 256k)
* **`sr`**: `256000` (Tần số lấy mẫu âm thanh 256 kHz). Giữ nguyên 100% các dải tần số cao $2\text{ kHz} - 8\text{ kHz}$ của vi xung cá đớp mồi.
* **`pre_emphasis`**: `0.97` (Hệ số lọc thông cao $y[t] = x[t] - 0.97 \cdot x[t-1]$). Khuếch đại vi xung tần số cao và dập nén tần số quạt nước $0-500\text{ Hz}$.
* **`frame_length`**: `4096` (Độ dài cửa sổ lấy mẫu win_length = 4096 mẫu).
* **`hop_length`**: `2048` (Bước nhảy cửa sổ hop_length = 2048 mẫu).
* **`n_fft`**: `4096` (Số điểm FFT n_fft = 4096), tạo ra ma trận $2049$ dải tần số sắc nét ($\Delta f = 62.5\text{ Hz/bin}$).
* **`windowing`**: `"hamming"` (Cửa sổ Hamming nén nhiễu nón phụ tốt hơn cửa sổ Hann truyền thống).
* **`use_std`**: `true` (Kết hợp cả năng lượng trung bình `Mean` và độ biến thiên `Std` trên 2049 dải tần trong cơ chế Frequency-Domain Attention).

---

### 🔹 Khối `references` (Đường dẫn dự án nguồn)
* **`audio_repo`**: Đường dẫn tương đối/tuyệt đối tới project baseline audio (Ví dụ: `"../U_FFIA27K_audio"`).
* **`video_repo`**: Đường dẫn tương đối/tuyệt đối tới project baseline video (Ví dụ: `"../U_FFIA27K_video"`).

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
* **`cache_audio`**: `true` hoặc `false`. Bật/tắt nạp sẵn Audio Waveforms 256k vào bộ nhớ RAM để tăng tốc nạp dữ liệu.
* **`video_cache_mode`**: `"ram"` (Nạp ảnh RGB $224 \times 224$ vào RAM), `"disk"` (Lưu cache trên đĩa), hoặc `"none"` (Đọc từ đĩa trực tiếp).
* **`num_workers`**: Số luồng CPU nạp dữ liệu. Đặt `-1` để hệ thống tự động tính: `num_workers = (max_cpu_cores // 2) + 1`.
* **`image_size`**: Kích thước khung hình đầu vào (Mặc định `224` tương ứng ảnh $224 \times 224$ px).

---

### 🔹 Khối `model` (Kiến trúc Mô hình & Fusion Head)
* **`video_backbone`**: Chọn tên visual backbone (`"mobilenet_v2"`, `"efficientnet_b0"`, `"densenet121"`, `"swin_tiny"`).
* **`encoder_mode`**:
  * `"tune"`: Mở cho phép fine-tune các lớp của Audio CNN & MobileNetV2 Video Encoder với `encoder_learning_rate` nhỏ.
  * `"frozen"`: Đông đóng toàn bộ tham số của Encoders, chỉ huấn luyện các lớp Fusion Head.
* **`d_model`**: Số chiều embedding ẩn của không gian đặc trưng chung (Mặc định `256`).
* **`dropout`**: Tỉ lệ dropout chống overfitting (Mặc định `0.1`).
* **`fusion_type`**: `"fbgf"` (Factorized Bilinear Gated Fusion MFB $k=3$ + Dynamic Gate) hoặc `"gmf"` (Pure Gated Multimodal Fusion).

---

### 🔹 Khối `training` (Siêu tham số Huấn luyện)
* **`batch_size`**: Kích thước mẫu theo batch cho huấn luyện (Mặc định `16` hoặc `32`).
* **`epochs`**: Số epoch tối đa (Mặc định `100`).
* **`fusion_learning_rate`**: Tốc độ học dành riêng cho Audio CNN và Fusion Head (Mặc định `0.001` / `1e-3`).
* **`encoder_learning_rate`**: Tốc độ học khi fine-tune Video Backbone MobileNetV2 (Mặc định `0.0001` / `1e-4`).
* **`weight_decay`**: Hệ số suy giảm trọng số L2 regularization (Mặc định `0.0001`).
* **`seed`**: Hạt giống ngẫu nhiên để đảm bảo tính tái lập kết quả (Mặc định `42`).
* **`device`**: `"auto"` (Tự động dùng GPU nếu có, nếu không chuyển CPU), `"cuda"`, hoặc `"cpu"`.
* **`output_dir`**: Đường dẫn thư mục xuất kết quả checkpoints và logs (Mặc định `"checkpoint"`).

---

### 🔹 Khối `results_upload` (Tự động Đẩy Kết quả lên Hugging Face)
* **`enabled`**: `true` hoặc `false`.
* **`repo_id`**: Tên repo trên Hugging Face (Mặc định `"manhmitcf/fish_result"`).
* **`repo_type`**: Loại repo (Luôn là `"dataset"`).
* **`path_prefix`**: Tên tiền tố thư mục trên Hugging Face (`"stft256k_raw2049_freqattn_mobilenetv2_fbgf"`).
* **`create_repo`**: `true` để tự tạo repo trên Hugging Face nếu chưa có.

---

## 📄 2. File Cấu hình Upload Artifact Zip: `config/artifact_upload_config.json`

File này quản lý việc nén **toàn bộ thư mục kết quả** (`checkpoint/`) thành file `.zip` và tải lên Hugging Face:

```json
{
  "enabled": true,
  "source_dir": "checkpoint",
  "zip_path": "checkpoint/STFT256k_Raw2049_FreqAttn_MobileNetV2_Artifacts.zip",
  "repo_id": "manhmitcf/fish_result",
  "repo_type": "dataset",
  "path_in_repo": "STFT256k_Raw2049_FreqAttn_MobileNetV2_Artifacts.zip",
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

### 2. Chuyển đổi Cơ chế Fusion để chạy Bài báo (Ablation Study)
Thay đổi `fusion_type` từ `"fbgf"` sang `"gmf"`:
```json
"model": {
  "fusion_type": "gmf"
}
```

### 3. Khởi chạy Pipeline
```bash
cd src
python main.py
```
