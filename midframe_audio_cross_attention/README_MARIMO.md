# Chạy multimodal trên Marimo

Folder này phát triển multimodal trực tiếp trên hai baseline gốc. Split holdout, cách đọc audio/video, cache và augmentation đều dùng source baseline; chỉ phần fusion là kiến trúc mới.

## 1. Clone đúng repo và nhánh trên Marimo

Trong Terminal của Marimo, clone trực tiếp nhánh thí nghiệm `exp/midframe-to-audio-cross-attn` từ repo chính:

```bash
cd /marimo
git clone --branch exp/midframe-to-audio-cross-attn --single-branch https://github.com/manhmitcf/Multimodal_Custom_Models.git
cd Multimodal_Custom_Models
git status
git branch --show-current
```

Lệnh cuối phải in `exp/midframe-to-audio-cross-attn`. Nếu repo đã được clone từ trước, cập nhật đúng nhánh bằng:

```bash
cd /marimo/Multimodal_Custom_Models
git switch exp/midframe-to-audio-cross-attn
git pull --ff-only origin exp/midframe-to-audio-cross-attn
```

## 2. Vị trí project trên Marimo

Chạy từ folder này:

```bash
cd /marimo/Multimodal_Custom_Models/midframe_audio_cross_attention
```

Project cần giữ cấu trúc cùng cấp sau:

```text
Multimodal_Custom_Models/
├── U_FFIA27K_audio/            # read-only baseline audio
├── U_FFIA27K_video/            # read-only baseline video
├── checkpoints/                # read-only checkpoint + split CSV
└── midframe_audio_cross_attention/
    ├── main.py
    ├── config/train_config.json
    ├── dataset/
    ├── models/
    └── tasks/
```

Dataset media trong CSV phải mount ở `/marimo/Fish_Feeding_Intensity_Dataset`. Kiểm tra trước khi chạy:

```bash
test -d /marimo/Fish_Feeding_Intensity_Dataset && echo "dataset OK"
test -f ../checkpoints/PANNS_Cnn6_holdout_random_sample_20260729_153012/DL_audio/checkpoint/panns_cnn6/audio_best.pt && echo "audio checkpoint OK"
test -f ../checkpoints/SwinTiny_holdout_random_sample_20260729_153012/DL_video/checkpoint/swin_tiny/video_best.pt && echo "video checkpoint OK"
```

## 3. Cài môi trường

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Sau khi cài xong, cấu hình run trong JSON rồi chạy `python main.py`.

## 4. Chỉ sửa một file config

Mọi tham số nằm trong [config/train_config.json](config/train_config.json). Không cần sửa Python để đổi thí nghiệm.

Các nhóm quan trọng:

| Nhóm | Sửa gì |
|---|---|
| `references` | đường dẫn hai repo baseline; không đổi nếu giữ cấu trúc project mặc định |
| `checkpoints` | một audio checkpoint và một video checkpoint được chọn |
| `data` | split CSV bất biến lấy thẳng từ checkpoint, cache audio/video, số worker, ảnh 224 |
| `model` | video backbone, `frozen`/`tune`, kích thước fusion |
| `training` | batch size, epoch, LR, seed, device, thư mục kết quả |

`checkpoints.video` phải khớp `model.video_backbone`:

| `model.video_backbone` | `checkpoints.video` |
|---|---|
| `densenet121` | `../../checkpoints/DenseNet121_holdout_random_sample_20260729_153012/DL_video/checkpoint/densenet121/video_best.pt` |
| `efficientnet_b0` | `../../checkpoints/EfficientNetB0_holdout_random_sample_20260804_181745/U_FFIA_video/checkpoint/efficientnet_b0/video_best.pt` |
| `mobilenet_v2` | `../../checkpoints/MobileNetV2_holdout_random_sample_20260729_153012/DL_video/checkpoint/mobilenet_v2/video_best.pt` |
| `swin_tiny` | `../../checkpoints/SwinTiny_holdout_random_sample_20260729_153012/DL_video/checkpoint/swin_tiny/video_best.pt` |

`data.split_dir` phải giữ đường dẫn tới `splits/` cạnh audio checkpoint. Ba file được đọc trực tiếp là:

```text
checkpoints/.../panns_cnn6/splits/train.csv
checkpoints/.../panns_cnn6/splits/val.csv
checkpoints/.../panns_cnn6/splits/test.csv
```

Không đặt `data.split_dir` trỏ về dataset gốc và không chạy `FishDataSplitter`: multimodal không tạo lại, không xáo trộn và không ghi đè split. Các split checkpoint hiện có gồm 21,467 train, 2,800 validation, 2,800 test.

Chế độ encoder:

- `frozen`: PANNS và video encoder không cập nhật; chỉ fusion head train. Đây là run nên làm đầu tiên.
- `tune`: mở PANNS `conv_block4`/`fc1` và stage video cuối; LR encoder dùng `training.encoder_learning_rate` thấp hơn LR fusion.

Giữ `data.cache_audio: true` và `data.video_cache_mode: "ram"` nếu muốn cùng hành vi cache của baseline đã chạy. Nếu Marimo thiếu RAM, có thể đổi `cache_audio: false` hoặc `video_cache_mode: "none"`; cần ghi rõ thay đổi này khi so sánh thí nghiệm.

Nếu CUDA out-of-memory, giảm `training.batch_size` (ví dụ `16 → 8 → 4`).

## 5. Chạy một run

`main.py` là entry point duy nhất:

```bash
python main.py
```

Nó luôn thực hiện đúng trình tự baseline:

```text
train.csv → train từng epoch
val.csv   → chọn best.pt theo macro-F1 validation
best.pt   → tự nạp lại
test.csv  → đánh giá holdout cuối cùng một lần
```

Không dùng test để chọn backbone, frozen/tune, epoch hoặc hyperparameter. Những lựa chọn này phải dựa trên `best_val_metrics.json` của các run trước.

## 6. Log khi chạy

Terminal hiển thị log từ source baseline khi khởi tạo audio/video pipeline và một dòng JSON sau mỗi epoch, gồm `epoch`, `train_loss` và toàn bộ metric validation. Khi macro-F1 validation tốt hơn, `best.pt` cùng metric validation được ghi lại. Cuối run, terminal in toàn bộ metric test.

Khi báo cáo hoặc kiểm tra lỗi, cần theo dõi tối thiểu:

- source repo, checkpoint audio/video và device đang dùng;
- split directory checkpoint và số mẫu ba split;
- cache policy (`cache_audio`, `video_cache_mode`), batch size và frozen/tune;
- train loss, validation macro-F1, epoch tốt nhất;
- test accuracy, macro-F1, per-class F1 và confusion matrix.

## 7. Kết quả

`training.output_dir` xác định vị trí output. Config mặc định tạo:

```text
runs/swin_tiny_frozen_cross_attn/
├── best.pt
├── best_val_metrics.json
├── best_val_confusion_matrix.csv
├── test_metrics.json
└── test_confusion_matrix.csv
```

Đọc metric:

```bash
python -m json.tool runs/swin_tiny_frozen_cross_attn/best_val_metrics.json
python -m json.tool runs/swin_tiny_frozen_cross_attn/test_metrics.json
```

Lưu cùng kết quả: tên video checkpoint, frozen/tune, seed, batch size, macro-F1 validation/test, F1 bốn lớp và confusion matrix. Khi viết paper, chỉ so sánh fusion với baseline khi checkpoint, split và data pipeline tương ứng đều đã được ghi nhận.

## 8. Điều gì được dùng lại từ baseline

- `FishVoiceDataLoader._InnerDataset`: đọc waveform, mono, resample, cắt đầu/pad cuối, cache audio.
- `AudioFrontend`, `PANNS_Cnn6`, `AudioModel`: frontend và encoder audio gốc.
- `FishVideoDataLoader._InnerDataset`: center frame, Decord→OpenCV fallback, cache frame.
- `VideoTransform`, `VideoModel`, DenseNet/EfficientNet/MobileNet/Swin source: pipeline và encoder ảnh gốc.
- Ba split CSV đã lưu cùng checkpoint: 21,467 train, 2,800 validation, 2,800 test.

Paired wrapper chỉ đảm bảo audio path, video path và label cùng record trước mỗi batch. Phần mới duy nhất là sáu audio window, lấy feature trước classifier và cross-attention theo proposal.
