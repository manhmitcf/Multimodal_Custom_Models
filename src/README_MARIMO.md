# HƯỚNG DẪN CHẠY THỬ NGHIỆM TRÊN MARIMO CLOUD SERVER

**Nhánh**: `exp/stft256k-raw2049-freqattn-mobilenetv2-fbgf`

## 🚀 1. Lệnh Git Checkout và Chuẩn bị Môi trường

```bash
# 1. Truy cập thư mục repository trên Marimo Server
cd /marimo/Multimodal_Custom_Models

# 2. Cập nhật nhánh từ xa
git fetch origin
git checkout exp/stft256k-raw2049-freqattn-mobilenetv2-fbgf
git pull origin exp/stft256k-raw2049-freqattn-mobilenetv2-fbgf

# 3. Cài đặt các thư viện cần thiết (nếu chưa có)
pip install -r src/requirements.txt
```

---

## 🏃 2. Lệnh Chạy Huấn luyện & Kiểm thử Duy nhất (Single Command Execution)

```bash
# Đổi hướng vào thư mục src và chạy script main.py
cd src
python main.py
```

---

## 📊 3. Kết quả Đầu ra tại `src/checkpoint/`

Sau khi quá trình huấn luyện hoàn tất, các kết quả tự động sinh ra gồm:
1. `splits/`: Copy 3 file `train.csv`, `val.csv`, `test.csv`.
2. `best.pt`: File trọng số mô hình có Validation Macro-F1 cao nhất.
3. `history.csv`: Bảng log từng epoch (train_loss, train_f1, val_loss, val_f1, ma trận 4x4).
4. `summary_results.csv`: Bảng tổng hợp kết quả tập Test Holdout, ghi rõ tham số & FLOPs.
5. `best_val_metrics.json` & `test_metrics.json`: Báo cáo chỉ số F1, Acc, Loss và Profiling.
6. `best_val_confusion_matrix.csv` & `test_confusion_matrix.csv`: Ma trận nhầm lẫn 4x4.
7. Tự động nén `STFT256k_Raw2049_FreqAttn_MobileNetV2_Artifacts.zip` và đẩy về Hugging Face `manhmitcf/fish_result`.
