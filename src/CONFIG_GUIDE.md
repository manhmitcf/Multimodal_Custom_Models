# 📘 HƯỚNG DẪN CHI TIẾT CÁC THAM SỐ CẤU HÌNH (CONFIG GUIDE)

Mọi thử nghiệm trong hệ thống được điều khiển bởi 2 file cấu hình chính nằm tại `src/config/`:
1. `src/config/train_config.json`: File siêu tham số huấn luyện và kiến trúc mô hình.
2. `src/config/artifact_upload_config.json`: File cấu hình tự động tải kết quả lên Hugging Face.

---

## ⚙️ 1. CHI TIẾT TỪNG PHÂN ĐOẠN TRONG `train_config.json`

### 1.1 Khối Siêu tham số STFT (`stft`)
| Tham số | Giá trị Mặc định | Giải thích Chi tiết |
| :--- | :--- | :--- |
| **`sr`** | `256000` | Tần số lấy mẫu âm thanh cao cấp 256 kHz (256,000 samples/sec). Giữ nguyên 100% tín hiệu siêu âm cá đớp mồi. |
| **`pre_emphasis`** | `0.97` | Hệ số lọc thông cao $y[t] = x[t] - 0.97 \cdot x[t-1]$. Khuếch đại tần số cao ($2-8\text{kHz}$) và dập nhiễu quạt ($0-500\text{Hz}$). |
| **`frame_length`** | `4096` | Độ dài cửa sổ lấy mẫu STFT (win_length = 4096 mẫu). |
| **`hop_length`** | `2048` | Bước nhảy cửa sổ STFT (hop_length = 2048 mẫu). |
| **`n_fft`** | `4096` | Số điểm biến đổi Fourier (n_fft = 4096), tạo ra ma trận $2049$ dải tần số sắc nét ($\Delta f = 62.5\text{ Hz/bin}$). |
| **`windowing`** | `"hamming"` | Loại cửa sổ biến đổi. Cửa sổ Hamming nén nhiễu nón phụ (side lobes) tốt hơn cửa sổ Hann truyền thống. |
| **`use_std`** | `true` | Khi `true`, cơ chế F-Attention kết hợp cả năng lượng trung bình `Mean` và độ biến thiên `Std` trên 2049 dải tần số. |

---

### 1.2 Khối Cấu hình Mô hình (`model`)
| Tham số | Giá trị Mặc định | Giải thích Chi tiết |
| :--- | :--- | :--- |
| **`video_backbone`** | `"mobilenet_v2"` | Kiến trúc backbone trích xuất đặc trưng hình ảnh (MobileNetV2 ~3.5M params). |
| **`encoder_mode`** | `"tune"` | `"tune"`: Cho phép fine-tune cập nhật trọng số MobileNetV2; `"freeze"`: Khóa đóng đóng băng trọng số. |
| **`d_model`** | `256` | Kích thước không gian vector đặc trưng chung (Audio & Video projection dimension = 256d). |
| **`dropout`** | `0.1` | Tỷ lệ dropout chống overfitting trong classifier. |
| **`fusion_type`** | `"fbgf"` | `"fbgf"`: Factorized Bilinear Gated Fusion (MFB $k=3$ + Dynamic Gate); `"gmf"`: Pure Gated Multimodal Fusion. |

---

### 1.3 Khối Cấu hình Dữ liệu & DataLoader (`data`)
| Tham số | Giá trị Mặc định | Giải thích Chi tiết |
| :--- | :--- | :--- |
| **`split_dir`** | `"../checkpoints/.../splits"` | Đường dẫn tới thư mục chứa 3 file CSV phân chia dữ liệu cố định `train.csv`, `val.csv`, `test.csv`. |
| **`cache_audio`** | `true` | Bật cache nạp toàn bộ sóng âm 256k vào bộ nhớ RAM ở phút đầu tiên. |
| **`video_cache_mode`** | `"ram"` | `"ram"`: Cache ảnh RGB $224 \times 224$ vào RAM; `"disk"`: Đọc trực tiếp từ đĩa. |
| **`num_workers`** | `-1` | Số lượng worker loader. Khi `-1`, hệ thống tự động tính: `num_workers = (max_cpu_cores // 2) + 1`. |
| **`image_size`** | `224` | Kích thước ảnh RGB đầu vào ($224 \times 224$). |

---

### 1.4 Khối Cấu hình Huấn luyện (`training`)
| Tham số | Giá trị Mặc định | Giải thích Chi tiết |
| :--- | :--- | :--- |
| **`batch_size`** | `16` | Kích thước batch huấn luyện (batch_size = 16 mẫu/bước). |
| **`epochs`** | `100` | Tổng số epoch huấn luyện (100 epochs). |
| **`fusion_learning_rate`**| `0.001` | Tốc độ học cho Audio CNN và Fusion Head ($10^{-3}$). |
| **`encoder_learning_rate`**| `0.0001` | Tốc độ học cho Pretrained Video Backbone MobileNetV2 ($10^{-4}$). |
| **`weight_decay`** | `0.0001` | Hệ số suy giảm trọng số L2 regularization ($10^{-4}$). |
| **`seed`** | `42` | Random seed cố định để đảm bảo kết quả tái lập 100%. |
| **`device`** | `"auto"` | `"auto"`: Tự động dùng GPU CUDA nếu có, ngược lại dùng CPU. |
| **`output_dir`** | `"checkpoint"` | Thư mục lưu trữ kết quả đầu ra (`src/checkpoint/`). |

---

## ☁️ 2. CHI TIẾT FILE `artifact_upload_config.json`

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

* **`enabled: true`**: Cho phép tự động nén zip và tải kết quả lên Hugging Face sau khi `python main.py` chạy xong.
* **`repo_id`**: Địa chỉ repository trên Hugging Face Datasets (`manhmitcf/fish_result`).
* **`zip_path`**: Đường dẫn file zip sản phẩm nén.
