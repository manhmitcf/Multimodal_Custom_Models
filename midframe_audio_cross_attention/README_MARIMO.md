# PANNS Cnn6 + DINOv2 spatial patch cross-attention

This experiment keeps the source audio pipeline and replaces the source video
classifier with DINOv2 patch tokens. It always uses one centre frame per video
and the immutable train/validation/test CSV files stored with the PANNS Cnn6
checkpoint.

## 1. Clone đúng nhánh trên Marimo

```bash
cd /marimo
git clone --branch exp/dinov2-spatial-visual-tokens-cross-attn --single-branch https://github.com/manhmitcf/Multimodal_Custom_Models.git
cd Multimodal_Custom_Models
git status
git branch --show-current
```

Lệnh cuối phải in `exp/dinov2-spatial-visual-tokens-cross-attn`. Nếu repo đã
được clone trước đó, cập nhật đúng nhánh:

```bash
cd /marimo/Multimodal_Custom_Models
git switch exp/dinov2-spatial-visual-tokens-cross-attn
git pull --ff-only origin exp/dinov2-spatial-visual-tokens-cross-attn
```

## 2. Tải và chuẩn hóa dataset trên Marimo

```bash
sudo apt update && sudo apt install git-lfs unzip -y
git lfs install
git config --global lfs.concurrenttransfers 64
cd /marimo
nohup git clone https://huggingface.co/datasets/manhmitcf/Fish_Feeding_Intensity_Dataset > clone.log 2>&1 &
tail -n +1 -f clone.log
```

Dataset phải nằm tại `/marimo/Fish_Feeding_Intensity_Dataset`. Nếu Hugging
Face tạo layout lồng `audio/audio` hoặc `video/video`, chuẩn hóa sau khi clone
kết thúc:

```bash
mv /marimo/Fish_Feeding_Intensity_Dataset/audio/audio/* /marimo/Fish_Feeding_Intensity_Dataset/audio/
mv /marimo/Fish_Feeding_Intensity_Dataset/video/video/* /marimo/Fish_Feeding_Intensity_Dataset/video/
rm -rf /marimo/Fish_Feeding_Intensity_Dataset/audio/audio
rm -rf /marimo/Fish_Feeding_Intensity_Dataset/video/video
find /marimo/Fish_Feeding_Intensity_Dataset -type f -iname "*.mp4" | wc -l
find /marimo/Fish_Feeding_Intensity_Dataset -type f -iname "*.wav" | wc -l
```

## 3. Tải PANNS checkpoint và immutable split

DINOv2 tự tải qua `torch.hub`; chỉ cần tải checkpoint PANNS Cnn6. Không cần
tải các video checkpoint DenseNet/EfficientNet/MobileNet/Swin của nhánh cũ.

```bash
cd /marimo/Multimodal_Custom_Models
python -m pip install --upgrade "huggingface_hub[cli]"
mkdir -p checkpoints /tmp/uffia_checkpoints

hf download hoangphihung442004/Results_U_FFIA27K_audio PANNS_Cnn6_holdout_random_sample_20260729_153012.zip --repo-type dataset --local-dir /tmp/uffia_checkpoints
mkdir -p checkpoints/PANNS_Cnn6_holdout_random_sample_20260729_153012
unzip -q /tmp/uffia_checkpoints/PANNS_Cnn6_holdout_random_sample_20260729_153012.zip -d checkpoints/PANNS_Cnn6_holdout_random_sample_20260729_153012 -x '.git/*' '*/.git/*'
rm -rf /tmp/uffia_checkpoints

test -d /marimo/Fish_Feeding_Intensity_Dataset && echo "dataset OK"
test -f /marimo/Multimodal_Custom_Models/checkpoints/PANNS_Cnn6_holdout_random_sample_20260729_153012/DL_audio/checkpoint/panns_cnn6/audio_best.pt && echo "audio checkpoint OK"
```

## Architecture

```text
Audio (2 seconds)
  -> 6 overlapping 0.75-second windows (hop 0.25 seconds)
  -> Log-Mel frontend -> PANNS Cnn6 -> 6 audio tokens x 512
  -> linear projection -> optional audio positional encoding
  -> audio self-attention

Centre RGB video frame (224 x 224)
  -> DINOv2 ViT-S/14-reg
  -> 256 patch tokens x 384
  -> linear projection + LayerNorm

DINO patch tokens (Query) + contextual audio tokens (Key, Value)
  -> cross-attention -> residual FFN -> mean spatial pooling -> 4-class logits
```

The DINO patch tokens already encode spatial position inside DINOv2. The model
does **not** add a second visual positional encoding. Audio positional encoding
is independent and can be switched on or off.

## 4. Project layout

Run from this directory while keeping these sibling folders:

```text
Multimodal_Custom_Models/
├── U_FFIA27K_audio/              # source PANNS Cnn6 code
├── U_FFIA27K_video/              # source centre-frame loader and transforms
├── checkpoints/                  # PANNS checkpoint and immutable splits
└── midframe_audio_cross_attention/
    ├── config/train_config.json
    └── main.py
```

The required audio checkpoint and split directory are configured in
`config/train_config.json`:

```text
checkpoints/PANNS_Cnn6_holdout_random_sample_20260729_153012/
  DL_audio/checkpoint/panns_cnn6/
    audio_best.pt
    splits/train.csv
    splits/val.csv
    splits/test.csv
```

The split CSV files must continue to point to the mounted media dataset, for
example `/marimo/Fish_Feeding_Intensity_Dataset`. This project never reruns the
splitter and never rewrites those CSV files.

## 5. Install and run

```bash
cd /marimo/Multimodal_Custom_Models/midframe_audio_cross_attention
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

On its first run, `torch.hub` downloads `facebookresearch/dinov2` and the
`dinov2_vits14_reg` weights. Internet access is therefore required once; later
runs use the local Torch Hub cache.

At startup the terminal also logs total, trainable, and frozen parameter counts,
plus estimated FLOPs for one `2-second audio + 224x224 centre-frame` sample.

## Configuration

Edit only `config/train_config.json` to configure a run.

| Key | Purpose |
| --- | --- |
| `model.dino_model` | Currently `dinov2_vits14_reg` (384-dimensional patch embeddings). |
| `model.dino_encoder_mode` | `frozen` trains no DINO parameters; `tune` opens the final DINO blocks. |
| `model.dino_tune_last_blocks` | Number of final DINO Transformer blocks to train when mode is `tune`, e.g. `2` or `4`. |
| `model.audio_encoder_mode` | `frozen` or `tune` for PANNS Cnn6 (`conv_block4` and `fc1` are opened in tune mode). |
| `model.audio_positional_encoding` | `learned`, `sinusoidal`, or `none`; this applies only to the six audio tokens. |
| `model.d_model` | Shared fusion dimension; the default is 256. |
| `training.fusion_learning_rate` | Learning rate for projections, audio attention, cross-attention, FFN, and classifier. |
| `training.encoder_learning_rate` | Lower learning rate for any enabled PANNS/DINO parameters. |

The recommended first experiment is `dino_encoder_mode: "frozen"`,
`audio_encoder_mode: "frozen"`, and
`audio_positional_encoding: "learned"`. After selecting the baseline using
validation macro-F1, try `dino_encoder_mode: "tune"` with
`dino_tune_last_blocks: 2`. Do not tune on the holdout test set.

## Data and caching

The source `FishVideoDataLoader` still selects the centre frame with
`frame_count // 2`, decodes with Decord (OpenCV fallback), and applies the
source RGB/ImageNet transform. `cache_audio` and `video_cache_mode` retain the
source baseline cache behaviour. The paired multimodal loader uses zero worker
processes to avoid duplicating RAM caches.

## Outputs

`training.output_dir` receives:

```text
best.pt
best_val_metrics.json
best_val_confusion_matrix.csv
test_metrics.json
test_confusion_matrix.csv
```

Each run selects `best.pt` by validation macro-F1, reloads it, and evaluates
the holdout test split once.
