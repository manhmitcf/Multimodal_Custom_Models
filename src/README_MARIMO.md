# 📖 HƯỚNG DẪN CHẠY THỬ NGHIỆM TRÊN MARIMO CLOUD SERVER

**Nhánh Git**: `exp/stft256k-raw2049-freqattn-mobilenetv2-fbgf`  
**Mô hình**: STFT 256k Raw 2049 Bins Audio CNN + MobileNetV2 Video + Factorized Bilinear Gated Fusion (FBGF)  
**Tập dữ liệu**: U-FFIA27K (27,000 mẫu đa thức âm thanh & hình ảnh)  

---

## 🌟 1. TỔNG QUAN VỀ THỬ NGHIỆM

Nhánh này triển khai mô hình đa thức siêu nhẹ (**~5.1M tham số**) với các bứt phá kỹ thuật chính:
1. **Lọc Pre-Emphasis ($\alpha=0.97$)**: Khuếch đại tín hiệu va chạm hạt cá tần số cao ($2\text{ kHz} - 8\text{ kHz}$) và triệt tiêu nhiễu quạt nước ($0 - 500\text{ Hz}$).
2. **Raw STFT 2049 Bins (Không nén Mel - Không mã hóa file ảnh RGB)**: Tính toán ma trận phổ Log-Magnitude STFT độ phân giải sắc nét $62.5\text{ Hz/bin}$ với cửa sổ Hamming ($4096, 2048$), đưa thẳng Tensor 2D $[B, 1, 2049, 250]$ vào Conv2d.
3. **Frequency-Domain Attention (F-Attention)**: Học ma trận trọng số $\mathbf{a}_F \in [0, 1]^{2049}$ dựa trên `Mean` và `Std` năng lượng thời gian để tự động khóa các dải tần cá ăn.
4. **Depthwise-Separable Audio CNN**: Lớp Conv đầu tiên `stride=(4, 2)` nén ngay tần số $2049 \rightarrow 513$ tức thì, nối tiếp các khối Depthwise-Separable $\rightarrow$ Vector Audio $256$d (~1.4M params).
5. **Factorized Bilinear Gated Fusion (FBGF / GMF)**: Dung hợp MFB Bilinear Hadamard ($k=3$), chuẩn hóa Power & L2 Norm kết hợp cổng Gated Dynamic Routing.
6. **Dataloader Tối ưu Tốc độ**: RAM Cache `cache_audio: true` & `video_cache_mode: "ram"`, `pin_memory = True`, `persistent_workers = True`, tự động tính `num_workers = max_cpu_cores // 2 + 1`.

---

## 🚀 2. HƯỚNG DẪN CÁC BƯỚC CHẠY TRÊN MARIMO SERVER

### Bước 1: Mở Terminal và Chuyển vào Thư mục Project
```bash
cd /marimo/Multimodal_Custom_Models
```

### Bước 2: Checkout sang Nhánh Thí nghiệm và Pull mã nguồn Mới nhất
```bash
git fetch origin
git checkout exp/stft256k-raw2049-freqattn-mobilenetv2-fbgf
git pull origin exp/stft256k-raw2049-freqattn-mobilenetv2-fbgf
```

### Bước 3: Cài đặt Dependencies (nếu cần)
```bash
pip install -r src/requirements.txt
```

### Bước 4: Thực thi Lệnh Huấn luyện & Kiểm thử Duy nhất
```bash
cd src
python main.py
```

---

## 📊 3. DANH SÁCH CHI TIẾT CÁC ARTIFACT KẾT QUẢ TẠI `src/checkpoint/`

Sau khi `python main.py` hoàn tất, toàn bộ sản phẩm thử nghiệm được tự động đóng gói tại `src/checkpoint/`:

| Tên File / Thư mục | Mô tả Chi tiết |
| :--- | :--- |
| **`splits/`** | Thư mục chứa 3 file CSV phân chia dữ liệu giữ nguyên: `train.csv`, `val.csv`, `test.csv`. |
| **`best.pt`** | Trọng số PyTorch của mô hình đạt điểm Validation Macro-F1 cao nhất. |
| **`history.csv`** | Nhật ký từng Epoch: `train_loss`, `train_acc`, `train_macro_f1`, `val_loss`, `val_acc`, `val_macro_f1`, và 16 ô ma trận nhầm lẫn Validation. |
| **`summary_results.csv`** | Bảng tổng kết kết quả tập Test Holdout, số lượng tham số (`total_params`, `trainable_params`) và thuật toán fusion. |
| **`best_val_metrics.json`** | Báo cáo JSON chi tiết tập Validation (Macro-F1, Accuracy, Loss, Epoch tốt nhất, Profiling). |
| **`test_metrics.json`** | Báo cáo JSON chi tiết tập Test Holdout (Macro-F1, Accuracy, Loss, Profiling). |
| **`best_val_confusion_matrix.csv`** | Ma trận nhầm lẫn 4x4 tập Validation (lớp 0: unfed, 1: low, 2: medium, 3: high). |
| **`test_confusion_matrix.csv`** | Ma trận nhầm lẫn 4x4 tập Test Holdout. |

---

## ☁️ 4. QUY TRÌNH TỰ ĐỘNG ĐẨY LÊN HUGGING FACE

Hệ thống sẽ tự động đóng gói toàn bộ thư mục `src/checkpoint/` thành file zip:
* **Tên File Zip**: `STFT256k_Raw2049_FreqAttn_MobileNetV2_Artifacts.zip`
* **Hugging Face Repository**: `manhmitcf/fish_result`
* **Kiểm tra trực tuyến**: `https://huggingface.co/datasets/manhmitcf/fish_result`

---

## 🛠️ 5. HƯỚNG DẪN TÌM BẮT LỖI (TROUBLESHOOTING)

1. **Kiểm tra GPU & VRAM**:
   ```bash
   nvidia-smi
   ```
2. **Kiểm thử nhanh 5 Unit Tests PyTest**:
   ```bash
   cd src
   python -m pytest tests/
   ```
