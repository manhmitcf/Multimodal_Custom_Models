# Chạy multimodal spatial visual tokens trên Marimo

Folder này phát triển multimodal trực tiếp trên hai baseline gốc. Split holdout, cách đọc audio/video, cache và augmentation đều dùng source baseline; chỉ phần visual-token fusion là kiến trúc mới.

## 1. Clone đúng repo và nhánh trên Marimo

Trong Terminal của Marimo, clone trực tiếp nhánh thí nghiệm `exp/spatial-visual-tokens-cross-attn` từ repo chính:

```bash
cd /marimo
git clone --branch exp/spatial-visual-tokens-cross-attn --single-branch https://github.com/manhmitcf/Multimodal_Custom_Models.git
cd Multimodal_Custom_Models
git status
git branch --show-current
```

Lệnh cuối phải in `exp/spatial-visual-tokens-cross-attn`. Nếu repo đã được clone từ trước, cập nhật đúng nhánh bằng:

```bash
cd /marimo/Multimodal_Custom_Models
git switch exp/spatial-visual-tokens-cross-attn
git pull --ff-only origin exp/spatial-visual-tokens-cross-attn
```

## 2. Tải và chuẩn hóa dataset trên Marimo

Chạy các lệnh sau trong Terminal Marimo để tải raw dataset qua Git LFS. Chờ clone hoàn tất trước khi thực hiện bước di chuyển/xóa folder lồng nhau.

```bash
sudo apt update && sudo apt install git-lfs unzip -y
git lfs install
git config --global lfs.concurrenttransfers 64
nohup git clone https://huggingface.co/datasets/manhmitcf/Fish_Feeding_Intensity_Dataset > clone.log 2>&1 &
tail -n +1 -f clone.log
```

Dataset phải nằm tại `/marimo/Fish_Feeding_Intensity_Dataset`. Sau khi `clone.log` báo hoàn tất, chuẩn hóa layout nếu Hugging Face clone tạo thêm một tầng `audio/audio` và `video/video`:

```bash
mv /marimo/Fish_Feeding_Intensity_Dataset/audio/audio/* /marimo/Fish_Feeding_Intensity_Dataset/audio/
mv /marimo/Fish_Feeding_Intensity_Dataset/video/video/* /marimo/Fish_Feeding_Intensity_Dataset/video/
rm -rf /marimo/Fish_Feeding_Intensity_Dataset/audio/audio
rm -rf /marimo/Fish_Feeding_Intensity_Dataset/video/video
```

Kiểm tra số file media sau khi chuẩn hóa:

```bash
find /marimo/Fish_Feeding_Intensity_Dataset -type f -iname "*.mp4" | wc -l
find /marimo/Fish_Feeding_Intensity_Dataset -type f -iname "*.wav" | wc -l
```

## 3. Vị trí project trên Marimo

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

## 4. Tải năm checkpoint đã chọn từ Hugging Face

Checkpoint audio và bốn checkpoint video được public ở hai Hugging Face dataset repo. **Không clone cả repo checkpoint** vì còn nhiều artifact nặng không dùng đến. Cài Hugging Face CLI rồi dùng `hf download` để tải trực tiếp đúng năm file `.zip` dưới đây. Mỗi zip được giải nén vào folder checkpoint tương ứng; metadata `.git/` có trong archive bị bỏ qua, rồi zip tạm được xóa:

```bash
cd /marimo/Multimodal_Custom_Models
python -m pip install --upgrade "huggingface_hub[cli]"
mkdir -p checkpoints /tmp/uffia_checkpoints

hf download hoangphihung442004/Results_U_FFIA27K_audio PANNS_Cnn6_holdout_random_sample_20260729_153012.zip --repo-type dataset --local-dir /tmp/uffia_checkpoints
hf download hoangphihung442004/Results_U_FFIA27K_video DenseNet121_holdout_random_sample_20260729_153012.zip --repo-type dataset --local-dir /tmp/uffia_checkpoints
hf download hoangphihung442004/Results_U_FFIA27K_video EfficientNetB0_holdout_random_sample_20260804_181745.zip --repo-type dataset --local-dir /tmp/uffia_checkpoints
hf download hoangphihung442004/Results_U_FFIA27K_video MobileNetV2_holdout_random_sample_20260729_153012.zip --repo-type dataset --local-dir /tmp/uffia_checkpoints
hf download hoangphihung442004/Results_U_FFIA27K_video SwinTiny_holdout_random_sample_20260729_153012.zip --repo-type dataset --local-dir /tmp/uffia_checkpoints

mkdir -p checkpoints/PANNS_Cnn6_holdout_random_sample_20260729_153012
mkdir -p checkpoints/DenseNet121_holdout_random_sample_20260729_153012
mkdir -p checkpoints/EfficientNetB0_holdout_random_sample_20260804_181745
mkdir -p checkpoints/MobileNetV2_holdout_random_sample_20260729_153012
mkdir -p checkpoints/SwinTiny_holdout_random_sample_20260729_153012

unzip -q /tmp/uffia_checkpoints/PANNS_Cnn6_holdout_random_sample_20260729_153012.zip -d checkpoints/PANNS_Cnn6_holdout_random_sample_20260729_153012 -x '.git/*' '*/.git/*'
unzip -q /tmp/uffia_checkpoints/DenseNet121_holdout_random_sample_20260729_153012.zip -d checkpoints/DenseNet121_holdout_random_sample_20260729_153012 -x '.git/*' '*/.git/*'
unzip -q /tmp/uffia_checkpoints/EfficientNetB0_holdout_random_sample_20260804_181745.zip -d checkpoints/EfficientNetB0_holdout_random_sample_20260804_181745 -x '.git/*' '*/.git/*'
unzip -q /tmp/uffia_checkpoints/MobileNetV2_holdout_random_sample_20260729_153012.zip -d checkpoints/MobileNetV2_holdout_random_sample_20260729_153012 -x '.git/*' '*/.git/*'
unzip -q /tmp/uffia_checkpoints/SwinTiny_holdout_random_sample_20260729_153012.zip -d checkpoints/SwinTiny_holdout_random_sample_20260729_153012 -x '.git/*' '*/.git/*'

rm -rf /tmp/uffia_checkpoints
```

Sau khi giải nén, `checkpoints/` phải có năm folder sau (không giữ đuôi `.zip`):

```text
PANNS_Cnn6_holdout_random_sample_20260729_153012/
DenseNet121_holdout_random_sample_20260729_153012/
EfficientNetB0_holdout_random_sample_20260804_181745/
MobileNetV2_holdout_random_sample_20260729_153012/
SwinTiny_holdout_random_sample_20260729_153012/
```

Kiểm tra năm file model trước khi chạy:

```bash
find checkpoints -type f -name "audio_best.pt" | wc -l
find checkpoints -type f -name "video_best.pt" | wc -l
```

Dataset media trong CSV phải mount ở `/marimo/Fish_Feeding_Intensity_Dataset`. Kiểm tra trước khi chạy:

```bash
test -d /marimo/Fish_Feeding_Intensity_Dataset && echo "dataset OK"
test -f /marimo/Multimodal_Custom_Models/checkpoints/PANNS_Cnn6_holdout_random_sample_20260729_153012/DL_audio/checkpoint/panns_cnn6/audio_best.pt && echo "audio checkpoint OK"
test -f /marimo/Multimodal_Custom_Models/checkpoints/SwinTiny_holdout_random_sample_20260729_153012/DL_video/checkpoint/swin_tiny/video_best.pt && echo "video checkpoint OK"
```

## 5. Cài môi trường

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`torchcodec` là dependency của `torchaudio.load()` trong baseline audio và đã có trong `requirements.txt`. Nếu đã cài requirements trước khi pull bản mới, chạy thêm `python -m pip install torchcodec` rồi mới chạy `main.py`.

Sau khi cài xong, cấu hình run trong JSON rồi chạy `python main.py`.

## 6. Chỉ sửa một file config

Mọi tham số nằm trong [config/train_config.json](config/train_config.json). Không cần sửa Python để đổi thí nghiệm.

Các nhóm quan trọng:

| Nhóm | Sửa gì |
|---|---|
| `references` | đường dẫn hai repo baseline; không đổi nếu giữ cấu trúc project mặc định |
| `checkpoints` | audio checkpoint và bảng path cố định của bốn video checkpoint |
| `data` | split CSV bất biến lấy thẳng từ checkpoint, cache audio/video, số worker, ảnh 224 |
| `model` | video backbone, `frozen`/`tune`, grid visual token, positional encoding, kích thước fusion |
| `training` | batch size, epoch, LR, seed, device, thư mục kết quả |

Chỉ đổi `model.video_backbone` để chọn một trong bốn encoder video. Code tự lấy path tương ứng từ `checkpoints.video_by_backbone`, nên không thể lẫn kiến trúc MobileNetV2 và checkpoint SwinTiny. Giữ đủ bốn path trong bảng này:

| `model.video_backbone` | `checkpoints.video_by_backbone` được dùng tự động |
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

`data.num_workers: -1` dùng đúng quy tắc auto của baseline: CPU không xác định/≤0 dùng `0`; đúng 2 CPU dùng `1`; các trường hợp khác dùng `CPU // 2 + 1`. Giá trị này chỉ dùng cho preload/cache audio và video của baseline. Paired `DataLoader` multimodal luôn dùng `0` worker để tránh tạo process phụ có thể nhân bản cache RAM; điều này không đổi split hay preprocessing.

Kiến trúc spatial visual tokens:

```text
Audio 2 giây → 6 cửa sổ 0.75 giây → Logmel + PANNS Cnn6 → 6 audio token 512D
Midframe → feature map trước global pooling của video baseline → adaptive pool grid 4×4 → 16 visual token
6 audio token + positional encoding (tùy chọn)  → audio Transformer self-attention + residual
16 visual token + positional encoding (tùy chọn) → visual Transformer self-attention + residual
visual token làm Query; audio token làm Key/Value → cross-attention → mean-pool 16 visual token → classifier 4 lớp
```

`model.visual_grid_size` mặc định `4`, tức 16 visual token. Có thể đặt `7` để giữ 49 token của feature map 7×7, nhưng tốn GPU và dễ overfit hơn. `model.positional_encoding` dùng một trong ba giá trị:

- `learned`: embedding vị trí trainable cho cả 6 audio token và visual grid; mặc định khuyến nghị.
- `sinusoidal`: encoding sin/cos cố định, không thêm parameter trainable.
- `none`: ablation không dùng thông tin vị trí.

Chế độ encoder:

- `frozen`: PANNS và video encoder không cập nhật; chỉ fusion head train. Đây là run nên làm đầu tiên.
- `tune`: mở PANNS `conv_block4`/`fc1` và stage video cuối; LR encoder dùng `training.encoder_learning_rate` thấp hơn LR fusion.

Giữ `data.cache_audio: true` và `data.video_cache_mode: "ram"` nếu muốn cùng hành vi cache của baseline đã chạy. Nếu Marimo thiếu RAM, có thể đổi `cache_audio: false` hoặc `video_cache_mode: "none"`; cần ghi rõ thay đổi này khi so sánh thí nghiệm.

Nếu CUDA out-of-memory, giảm `training.batch_size` (ví dụ `16 → 8 → 4`).

## 7. Chạy một run

`main.py` là entry point duy nhất:

```bash
python main.py
```

Để chạy nền trên Marimo và theo dõi log liên tục:

```bash
nohup python3 main.py > main.log 2>&1 &
tail -n +1 -f main.log
```

Nó luôn thực hiện đúng trình tự baseline:

```text
train.csv → train từng epoch
val.csv   → chọn best.pt theo macro-F1 validation
best.pt   → tự nạp lại
test.csv  → đánh giá holdout cuối cùng một lần
```

Không dùng test để chọn backbone, frozen/tune, epoch hoặc hyperparameter. Những lựa chọn này phải dựa trên `best_val_metrics.json` của các run trước.

## 8. Log khi chạy

Terminal hiển thị log từ source baseline khi khởi tạo audio/video pipeline. Trong train, progress bar `Epoch x/y` cập nhật ở mỗi batch cùng `Loss` và `Mean Loss`; validation và holdout test cũng có progress bar như baseline. Cuối mỗi epoch, log in train loss, validation accuracy, macro-F1 và F1 từng lớp. Khi macro-F1 validation tốt hơn, `best.pt` cùng metric validation được ghi lại. Cuối run, terminal in toàn bộ metric test.

Khi báo cáo hoặc kiểm tra lỗi, cần theo dõi tối thiểu:

- source repo, checkpoint audio/video và device đang dùng;
- split directory checkpoint và số mẫu ba split;
- cache policy (`cache_audio`, `video_cache_mode`), batch size và frozen/tune;
- train loss, validation macro-F1, epoch tốt nhất;
- test accuracy, macro-F1, per-class F1 và confusion matrix.

## 9. Kết quả

`training.output_dir` xác định vị trí output. Config mặc định tạo:

```text
runs/swin_tiny_spatial4x4_learned_pos_cross_attn/
├── best.pt
├── best_val_metrics.json
├── best_val_confusion_matrix.csv
├── test_metrics.json
└── test_confusion_matrix.csv
```

Đọc metric:

```bash
python -m json.tool runs/swin_tiny_spatial4x4_learned_pos_cross_attn/best_val_metrics.json
python -m json.tool runs/swin_tiny_spatial4x4_learned_pos_cross_attn/test_metrics.json
```

Lưu cùng kết quả: tên video checkpoint, frozen/tune, seed, batch size, macro-F1 validation/test, F1 bốn lớp và confusion matrix. Khi viết paper, chỉ so sánh fusion với baseline khi checkpoint, split và data pipeline tương ứng đều đã được ghi nhận.

## 10. Điều gì được dùng lại từ baseline

- `FishVoiceDataLoader._InnerDataset`: đọc waveform, mono, resample, cắt đầu/pad cuối, cache audio.
- `AudioFrontend`, `PANNS_Cnn6`, `AudioModel`: frontend và encoder audio gốc.
- `FishVideoDataLoader._InnerDataset`: center frame, Decord→OpenCV fallback, cache frame.
- `VideoTransform`, `VideoModel`, DenseNet/EfficientNet/MobileNet/Swin source: pipeline và encoder ảnh gốc.
- Ba split CSV đã lưu cùng checkpoint: 21,467 train, 2,800 validation, 2,800 test.

Paired wrapper chỉ đảm bảo audio path, video path và label cùng record trước mỗi batch. Phần mới duy nhất là sáu audio window, visual feature map trước global pooling, positional encoding tùy chọn, hai Transformer self-attention và cross-attention.
