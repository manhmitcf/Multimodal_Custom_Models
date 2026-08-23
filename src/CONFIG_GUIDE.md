# HƯỚNG DẪN GIẢI THÍCH CÁC THAM SỐ CẤU HÌNH (CONFIG GUIDE)

File cấu hình chính: `src/config/train_config.json`

## 1. Khối Cấu hình STFT (`stft`)
* `sr`: `256000` (Tần số lấy mẫu cao cấp 256 kHz).
* `pre_emphasis`: `0.97` (Hệ số lọc thông cao y[t] = x[t] - 0.97 * x[t-1]).
* `frame_length`: `4096` (Độ dài cửa sổ lấy mẫu STFT).
* `hop_length`: `2048` (Bước nhảy cửa sổ STFT).
* `n_fft`: `4096` (Số điểm FFT, tạo ra 2049 dải tần số sắc nét).
* `windowing`: `"hamming"` (Cửa sổ Hamming nén nhiễu nón phụ).
* `use_std`: `true` (Kết hợp cả năng lượng trung bình Mean và độ biến thiên Std trong F-Attention).

## 2. Khối Cấu hình Mô hình (`model`)
* `video_backbone`: `"mobilenet_v2"` (Backbone hình ảnh).
* `d_model`: `256` (Kích thước vector đặc trưng chung).
* `fusion_type`: `"fbgf"` (Factorized Bilinear Gated Fusion) hoặc `"gmf"` (Gated Multimodal Fusion).

## 3. Khối Cấu hình Dữ liệu (`data`)
* `cache_audio`: `true` (Cache âm thanh 256k vào RAM).
* `video_cache_mode`: `"ram"` (Cache ảnh RGB 224x224 vào RAM).
* `num_workers`: `-1` (Tự động tính num_workers = max_cpu_cores // 2 + 1).
