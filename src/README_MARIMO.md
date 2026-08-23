# Method 3 (GW-AVF): Geometry & Water-Ripple Audio-Visual Cross-Attention (Marimo Guide)

Biển hướng dẫn này dành cho **Nhánh `exp/geometry-water-ripple-cross-attn`**, triển khai **Phương pháp 3 (GW-AVF)**: Tự động trích xuất đặc trưng **Sóng nước (Water Ripples)** & **Mật độ hình học đàn cá (Delaunay Flocking Geometry)** trên lưới $14 \times 14$ không gian, dung hợp với $6$ audio tokens thông qua Cross-Attention, và tự động nén/upload kết quả đầy đủ lên **[`manhmitcf/fish_result`](https://huggingface.co/datasets/manhmitcf/fish_result)**.

> 📖 **Hướng dẫn chi tiết từng tham số cấu hình JSON**: Xem tài liệu [`CONFIG_GUIDE.md`](file:///C:/Users/manhm/Desktop/Multimodal_Custom_Models/src/CONFIG_GUIDE.md).

---

## 1. Clone nhánh thí nghiệm này

```bash
cd /marimo
git clone --branch exp/geometry-water-ripple-cross-attn --single-branch https://github.com/manhmitcf/Multimodal_Custom_Models.git
cd Multimodal_Custom_Models
git branch --show-current
```
*Lưu ý: Lệnh `git branch` phải hiển thị đúng `exp/geometry-water-ripple-cross-attn`.*

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

Chuẩn hóa cấu trúc thư mục dữ liệu nếu Hugging Face clone tạo các thư mục con lồng nhau:
```bash
mv /marimo/Fish_Feeding_Intensity_Dataset/audio/audio/* /marimo/Fish_Feeding_Intensity_Dataset/audio/ 2>/dev/null || true
mv /marimo/Fish_Feeding_Intensity_Dataset/video/video/* /marimo/Fish_Feeding_Intensity_Dataset/video/ 2>/dev/null || true
rm -rf /marimo/Fish_Feeding_Intensity_Dataset/audio/audio /marimo/Fish_Feeding_Intensity_Dataset/video/video
```

---

## 3. Tải Checkpoints & Immutable Splits

Tải các checkpoint tiền huấn luyện của Audio (PANNS Cnn6) và Visual (SwinTiny):

```bash
cd /marimo/Multimodal_Custom_Models
python -m pip install --upgrade "huggingface_hub[cli]"
mkdir -p checkpoints /tmp/uffia_checkpoints

hf download hoangphihung442004/Results_U_FFIA27K_audio PANNS_Cnn6_holdout_random_sample_20260729_153012.zip --repo-type dataset --local-dir /tmp/uffia_checkpoints
hf download hoangphihung442004/Results_U_FFIA27K_video SwinTiny_holdout_random_sample_20260729_153012.zip --repo-type dataset --local-dir /tmp/uffia_checkpoints

mkdir -p checkpoints/PANNS_Cnn6_holdout_random_sample_20260729_153012
mkdir -p checkpoints/SwinTiny_holdout_random_sample_20260729_153012
unzip -q /tmp/uffia_checkpoints/PANNS_Cnn6_holdout_random_sample_20260729_153012.zip -d checkpoints/PANNS_Cnn6_holdout_random_sample_20260729_153012 -x '.git/*' '*/.git/*'
unzip -q /tmp/uffia_checkpoints/SwinTiny_holdout_random_sample_20260729_153012.zip -d checkpoints/SwinTiny_holdout_random_sample_20260729_153012 -x '.git/*' '*/.git/*'
rm -rf /tmp/uffia_checkpoints
```

Kiểm tra sự tồn tại của file checkpoint trước khi chạy:
```bash
test -f checkpoints/PANNS_Cnn6_holdout_random_sample_20260729_153012/DL_audio/checkpoint/panns_cnn6/audio_best.pt && echo "Audio Checkpoint OK"
test -f checkpoints/SwinTiny_holdout_random_sample_20260729_153012/DL_video/checkpoint/swin_tiny/video_best.pt && echo "Video Checkpoint OK"
test -f checkpoints/PANNS_Cnn6_holdout_random_sample_20260729_153012/DL_audio/checkpoint/panns_cnn6/splits/train.csv && echo "Splits CSV OK"
```

---

## 4. Cài đặt thư viện & Khởi chạy Pipeline

Đăng nhập Hugging Face (hoặc set biến môi trường `HF_TOKEN`) để tự động upload kết quả:
```bash
hf auth login
```

Chạy trực tiếp pipeline thí nghiệm Phương pháp 3 từ thư mục `src`:
```bash
cd /marimo/Multimodal_Custom_Models/src
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Khởi chạy Phương pháp 3 (GW-AVF)
python main.py
```

Nếu chạy ẩn trong background:
```bash
nohup python3 main.py > main.log 2>&1 &
tail -n +1 -f main.log
```

---

## 🏗️ Kiến trúc Phương pháp 3 (GW-AVF) Hoạt động Như thế nào?

```text
       Middle Video Frame (RGB)                       Audio 2s (Waveform)
                  │                                            │
   ┌──────────────┴──────────────┐                       Audio Encoder
   ▼                             ▼                             │
SwinTiny (Stage 3)     Geometry & Water-Ripple                 │
Spatial Tokens Grid    Feature Extractors                      │
(196 x 384)            (Wavelet & Delaunay)                    │
   │                             │                             │
   └──────────────┬──────────────┘                             │
                  ▼                                            ▼
     Geometry-Enhanced Visual Tokens ────────────►  Cross-Attention Fusion
             (196 x d_model)                        (Q=Visual, K,V=Audio)
                                                               │
                                                       Classifier (4 classes)
```

---

## 📂 Danh sách Kết quả Đầu ra (Outputs Checkpoint Artifacts)

Thư mục kết quả `checkpoint/` (hoặc `runs/...`) sẽ được sinh ra đầy đủ 100%:

```text
checkpoint/
  ├── splits/                          <- Thư mục chứa train.csv, val.csv, test.csv
  ├── best.pt                          <- Checkpoint PyTorch lưu model_state_dict & val_macro_f1 tốt nhất
  ├── history.csv                      <- Log lịch sử từng epoch (loss, acc, F1, 16 cột CM)
  ├── summary_results.csv              <- Bảng tổng hợp F1 & Accuracy tập Test Holdout
  ├── best_val_metrics.json            <- Chỉ số chi tiết Validation
  ├── test_metrics.json                <- Chỉ số chi tiết Holdout Test
  ├── best_val_confusion_matrix.csv    <- Ma trận nhầm lẫn Validation 4x4 (có nhãn)
  └── test_confusion_matrix.csv        <- Ma trận nhầm lẫn Holdout Test 4x4 (có nhãn)
```

Cuối quá trình chạy, toàn bộ thư mục này sẽ được tự động đóng gói thành file `.zip` và tải lên repository **[`manhmitcf/fish_result`](https://huggingface.co/datasets/manhmitcf/fish_result)**.

---

## ⚙️ Hướng dẫn Cấu hình Siêu tham số (Configuration)

Chi tiết ý nghĩa từng thông số và cách chỉnh sửa file [`config/train_config.json`](file:///C:/Users/manhm/Desktop/Multimodal_Custom_Models/src/config/train_config.json) vui lòng xem tại file:
👉 **[`CONFIG_GUIDE.md`](file:///C:/Users/manhm/Desktop/Multimodal_Custom_Models/src/CONFIG_GUIDE.md)**
