# STFT 256k Raw 2049 Bins with Frequency-Domain Attention & MobileNetV2 Factorized Bilinear Gated Fusion (Marimo Guide)

Biển hướng dẫn này dành cho **Nhánh `exp/stft256k-raw2049-freqattn-mobilenetv2-fbgf`**, triển khai **Phương pháp Dung hợp Đa thức Nâng cao Tự Định Nghĩa**: Tích hợp **Lọc Pre-Emphasis ($\alpha=0.97$)**, **Phổ Raw STFT 2049 Dải Tần số** ($256\text{ kHz}, n\_fft=4096, win=4096, hop=2048, \text{Hamming}$, KHÔNG nén Mel, KHÔNG mã hóa file ảnh RGB), **Cơ chế Chú ý Miền Tần số (Frequency-Domain Attention)** và mạng **Depthwise-Separable Audio CNN**, dung hợp với kênh Video **MobileNetV2** siêu nhẹ (~3.5M params) thông qua **Factorized Bilinear Gated Fusion (FBGF / GMF)**, và tự động nén/upload kết quả đầy đủ lên **[`manhmitcf/fish_result`](https://huggingface.co/datasets/manhmitcf/fish_result)**.

> 📖 **Hướng dẫn chi tiết từng tham số cấu hình JSON**: Xem tài liệu [`CONFIG_GUIDE.md`](file:///C:/Users/manhm/Desktop/Multimodal_Custom_Models/src/CONFIG_GUIDE.md).

---

## 1. Clone hoặc Checkout nhánh thí nghiệm này

```bash
cd /marimo
git clone --branch exp/stft256k-raw2049-freqattn-mobilenetv2-fbgf --single-branch https://github.com/manhmitcf/Multimodal_Custom_Models.git
cd Multimodal_Custom_Models
git branch --show-current
```
*Lưu ý: Lệnh `git branch` phải hiển thị đúng `exp/stft256k-raw2049-freqattn-mobilenetv2-fbgf`.*

Hoặc nếu đã có sẵn thư mục repository trong máy/server:
```bash
cd /marimo/Multimodal_Custom_Models
git fetch origin
git checkout exp/stft256k-raw2049-freqattn-mobilenetv2-fbgf
git pull origin exp/stft256k-raw2049-freqattn-mobilenetv2-fbgf
```

---

## 2. Tải bộ dữ liệu (Dataset Setup)

```bash
sudo apt update && sudo apt install git-lfs unzip -y
git lfs install
git config --global lfs.concurrenttransfers 64
cd /marimo
nohup git clone https://huggingface.co/datasets/manhmitcf/Fish_Feeding_Intensity_Dataset > clone.log 2>&1 &
tail -n +1 -f clone.log
```

Chuẩn hóa cấu trúc thư mục dữ liệu:
```bash
mv /marimo/Fish_Feeding_Intensity_Dataset/audio/audio/* /marimo/Fish_Feeding_Intensity_Dataset/audio/ 2>/dev/null || true
mv /marimo/Fish_Feeding_Intensity_Dataset/video/video/* /marimo/Fish_Feeding_Intensity_Dataset/video/ 2>/dev/null || true
rm -rf /marimo/Fish_Feeding_Intensity_Dataset/audio/audio /marimo/Fish_Feeding_Intensity_Dataset/video/video
```

---

## 3. Tải Checkpoints & Immutable Splits

Tải các checkpoint tiền huấn luyện của Audio (PANNS Cnn6) và Visual (MobileNetV2):

```bash
cd /marimo/Multimodal_Custom_Models
python -m pip install --upgrade "huggingface_hub[cli]"
mkdir -p checkpoints /tmp/uffia_checkpoints

hf download hoangphihung442004/Results_U_FFIA27K_audio PANNS_Cnn6_holdout_random_sample_20260729_153012.zip --repo-type dataset --local-dir /tmp/uffia_checkpoints
hf download hoangphihung442004/Results_U_FFIA27K_video MobileNetV2_holdout_random_sample_20260729_153012.zip --repo-type dataset --local-dir /tmp/uffia_checkpoints

mkdir -p checkpoints/PANNS_Cnn6_holdout_random_sample_20260729_153012
mkdir -p checkpoints/MobileNetV2_holdout_random_sample_20260729_153012
unzip -q /tmp/uffia_checkpoints/PANNS_Cnn6_holdout_random_sample_20260729_153012.zip -d checkpoints/PANNS_Cnn6_holdout_random_sample_20260729_153012 -x '.git/*' '*/.git/*'
unzip -q /tmp/uffia_checkpoints/MobileNetV2_holdout_random_sample_20260729_153012.zip -d checkpoints/MobileNetV2_holdout_random_sample_20260729_153012 -x '.git/*' '*/.git/*'
rm -rf /tmp/uffia_checkpoints
```

Kiểm tra sự tồn tại của file checkpoint và file phân chia dữ liệu trước khi chạy:
```bash
test -f checkpoints/PANNS_Cnn6_holdout_random_sample_20260729_153012/DL_audio/checkpoint/panns_cnn6/audio_best.pt && echo "Audio PANNS Checkpoint OK"
test -f checkpoints/MobileNetV2_holdout_random_sample_20260729_153012/DL_video/checkpoint/mobilenet_v2/video_best.pt && echo "Video MobileNetV2 Checkpoint OK"
test -f checkpoints/PANNS_Cnn6_holdout_random_sample_20260729_153012/DL_audio/checkpoint/panns_cnn6/splits/train.csv && echo "Splits CSV OK"
```

---

## 4. Cài đặt thư viện & Khởi chạy Pipeline

Đăng nhập Hugging Face (hoặc set biến môi trường `HF_TOKEN`) để tự động upload kết quả:
```bash
hf auth login
```

Chạy trực tiếp pipeline từ thư mục `src`:
```bash
cd /marimo/Multimodal_Custom_Models/src
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Khởi chạy Pipeline Custom Raw STFT 2049 F-Attn MobileNetV2 FBGF Fusion
python main.py
```

Nếu chạy ẩn trong background:
```bash
nohup python3 main.py > main.log 2>&1 &
tail -n +1 -f main.log
```

---

## 📂 Danh sách Kết quả Đầu ra (Outputs Checkpoint Artifacts)

Thư mục kết quả `checkpoint/` sẽ được sinh ra đầy đủ 100%:

```text
checkpoint/
  ├── splits/                          <- Thư mục chứa train.csv, val.csv, test.csv
  ├── best.pt                          <- Checkpoint PyTorch lưu model_state_dict & val_macro_f1 tốt nhất
  ├── history.csv                      <- Log lịch sử từng epoch (loss, acc, F1, 16 cột CM)
  ├── summary_results.csv              <- Bảng tổng hợp F1 & Accuracy tập Test Holdout, số lượng tham số & FLOPs
  ├── best_val_metrics.json            <- Chỉ số chi tiết Validation (Macro-F1, Accuracy, Loss, Profiling)
  ├── test_metrics.json                <- Chỉ số chi tiết Holdout Test (Macro-F1, Accuracy, Loss, Profiling)
  ├── best_val_confusion_matrix.csv    <- Ma trận nhầm lẫn Validation 4x4 (có nhãn)
  └── test_confusion_matrix.csv        <- Ma trận nhầm lẫn Holdout Test 4x4 (có nhãn)
```

Cuối quá trình chạy, toàn bộ thư mục này sẽ được tự động đóng gói thành file `STFT256k_Raw2049_FreqAttn_MobileNetV2_Artifacts.zip` và tải lên repository **[`manhmitcf/fish_result`](https://huggingface.co/datasets/manhmitcf/fish_result)**.
