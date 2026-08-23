# ⚙️ Hướng dẫn Chi tiết Cấu hình File JSON (CONFIG_GUIDE)
*(Nhánh: `exp/stft256k-raw2049-freqattn-efficientnetb0-fbgf` - Backbone: EfficientNetB0)*

Tài liệu này giải thích chi tiết ý nghĩa từng thông số trong file cấu hình JSON [`config/train_config.json`](file:///C:/Users/manhm/Desktop/Multimodal_Custom_Models/src/config/train_config.json) và [`config/artifact_upload_config.json`](file:///C:/Users/manhm/Desktop/Multimodal_Custom_Models/src/config/artifact_upload_config.json) của nhánh **`exp/stft256k-raw2049-freqattn-efficientnetb0-fbgf`**, giúp bạn dễ dàng chỉnh sửa và tùy biến cho các kịch bản thử nghiệm khác nhau.

---

## 📄 1. File Cấu hình Chính: `config/train_config.json`

File cấu hình được chia thành 7 khối thông số chính:

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

---

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
* **`audio_repo`**: Đường dẫn tương đối/tuyệt đối tới project baseline audio (`"../U_FFIA27K_audio"`).
* **`video_repo`**: Đường dẫn tương đối/tuyệt đối tới project baseline video (`"../U_FFIA27K_video"`).

---

### 🔹 Khối `checkpoints` (Trọng số mô hình Pretrained Baseline)
* **`audio`**: Đường dẫn file trọng số `audio_best.pt` của mô hình Audio (PANNS Cnn6):  
  `"../checkpoints/PANNS_Cnn6_holdout_random_sample_20260729_153012/DL_audio/checkpoint/panns_cnn6/audio_best.pt"`
* **`video_by_backbone`**: Từ điển ánh xạ từng loại visual backbone tới file trọng số `video_best.pt` tương ứng:
  * `"efficientnet_b0"` ⭐ *(Mặc định)*: `"../checkpoints/EfficientNetB0_holdout_random_sample_20260804_181745/U_FFIA_video/checkpoint/efficientnet_b0/video_best.pt"`
  * `"mobilenet_v2"`: `"../checkpoints/MobileNetV2_holdout_random_sample_20260729_153012/DL_video/checkpoint/mobilenet_v2/video_best.pt"`

---

### 🔹 Khối `data` (Quản lý Dữ liệu & DataLoader)
* **`split_dir`**: Đường dẫn thư mục chứa 3 file CSV cố định (`train.csv`, `val.csv`, `test.csv`).
* **`cache_audio`**: `false` (Đọc trực tiếp file STFT `.npy` trích xuất sẵn từ đĩa NVMe SSD).
* **`video_cache_mode`**: `"ram"` (Nạp ảnh RGB $224 \times 224$ mảng `uint8` vào RAM siêu tốc 5s startup), `"disk"` (Lưu/đọc cache trên đĩa), hoặc `"none"` (Giải mã `.mp4` trực tiếp).
* **`num_workers`**: Số luồng CPU nạp dữ liệu. Đặt `-1` để hệ thống tự động sử dụng toàn bộ số nhân CPU (`os.cpu_count()`).
* **`image_size`**: Kích thước khung hình đầu vào (Mặc định `224` tương ứng ảnh $224 \times 224$ px).

---

### 🔹 Khối `model` (Kiến trúc Mô hình & Visual Backbone)
* **`video_backbone`**: `"efficientnet_b0"` (Sử dụng backbone **EfficientNetB0** pre-trained trích vector 1280-dim $[B, 1280]$).
* **`encoder_mode`**:
  * `"frozen"` ⭐ *(Khuyên dùng)*: Đóng băng toàn bộ trọng số của Encoders, ghim chặt `self.video_encoder.eval()` bảo vệ $100\%$ thống kê 52 lớp BatchNorm.
  * `"tune"`: Cho phép fine-tune các lớp đại diện.
* **`d_model`**: Số chiều embedding ẩn của không gian đặc trưng chung (Mặc định `256`).
* **`dropout`**: Tỉ lệ dropout chống overfitting (Mặc định `0.1`).
* **`fusion_type`**: `"fbgf"` (Factorized Bilinear Gated Fusion MFB $k=3$ + Dynamic Gate) hoặc `"gmf"` (Pure Gated Multimodal Fusion).

---

### 🔹 Khối `training` (Siêu tham số Huấn luyện)
* **`batch_size`**: Kích thước mẫu theo batch cho huấn luyện (Mặc định `64`).
* **`epochs`**: Số epoch tối đa (Mặc định `100`).
* **`fusion_learning_rate`**: Tốc độ học tối ưu `0.0001` ($10^{-4}$) dành riêng cho Audio CNN và Fusion Head (đã giúp đạt mốc vô địch 95.34% Accuracy).
* **`encoder_learning_rate`**: Tốc độ học khi fine-tune Video Backbone (Mặc định `0.0001` / $10^{-4}$).
* **`weight_decay`**: Hệ số suy giảm trọng số L2 regularization (Mặc định `0.0001`).
* **`seed`**: Hạt giống ngẫu nhiên để đảm bảo tính tái lập kết quả (Mặc định `42`).
* **`device`**: `"auto"` (Tự động dùng GPU nếu có, nếu không chuyển CPU), `"cuda"`, hoặc `"cpu"`.
* **`output_dir`**: Đường dẫn thư mục xuất kết quả checkpoints và logs (Mặc định `"checkpoint"`).

---

### 🔹 Khối `results_upload` (Tự động Đẩy Kết quả lên Hugging Face)
* **`enabled`**: `true` (Tự động upload kết quả sau khi hoàn thành test holdout).
* **`repo_id`**: Tên repo trên Hugging Face (`"manhmitcf/fish_result"`).
* **`repo_type`**: Loại repo (`"dataset"`).
* **`path_prefix`**: Tên tiền tố thư mục trên Hugging Face (`"stft256k_raw2049_freqattn_efficientnetb0_fbgf"`).
* **`create_repo`**: `true` để tự tạo repo trên Hugging Face nếu chưa có.

---

## 📄 2. File Cấu hình Upload Artifact Zip: `config/artifact_upload_config.json`

File này quản lý việc nén **toàn bộ thư mục kết quả** (`checkpoint/`) thành file `.zip` và tải lên Hugging Face:

```json
{
  "enabled": true,
  "source_dir": "checkpoint",
  "zip_path": "checkpoint/STFT256k_Raw2049_FreqAttn_EfficientNetB0_Artifacts.zip",
  "repo_id": "manhmitcf/fish_result",
  "repo_type": "dataset",
  "path_in_repo": "STFT256k_Raw2049_FreqAttn_EfficientNetB0_Artifacts.zip",
  "create_repo": true
}
```

---

## 💡 Hướng dẫn Tùy chỉnh Nhanh cho Các Kịch bản Thường gặp

### 1. Nếu bị tràn bộ nhớ GPU (CUDA Out of Memory)
Hãy chỉnh giảm `batch_size` trong `train_config.json`:
```json
"training": {
  "batch_size": 32
}
```

### 2. Chuyển đổi Cơ chế Fusion để chạy Bài báo (Ablation Study)
Thay đổi `fusion_type` từ `"fbgf"` sang `"gmf"` hoặc `"concat"`:
```json
"model": {
  "fusion_type": "gmf"
}
```

### 3. Khởi chạy Pipeline trên Server Marimo
```bash
cd /marimo/Multimodal_Custom_Models/src
python main.py
```
