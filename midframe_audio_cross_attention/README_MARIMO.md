# STFT 256 kHz audio + source video spatial-token cross-attention

This branch keeps the original centre-frame video pipeline and its video
checkpoints. It replaces Log-Mel + PANNS Cnn6 with six Librosa STFT tokens.

## 1. Clone đúng nhánh trên Marimo

```bash
cd /marimo
git clone --branch exp/stft256k-spatial-video-cross-attn --single-branch https://github.com/manhmitcf/Multimodal_Custom_Models.git
cd Multimodal_Custom_Models
git status
git branch --show-current
```

For an existing checkout:

```bash
cd /marimo/Multimodal_Custom_Models
git switch exp/stft256k-spatial-video-cross-attn
git pull --ff-only origin exp/stft256k-spatial-video-cross-attn
```

## 2. Dataset on Marimo

```bash
sudo apt update && sudo apt install git-lfs unzip -y
git lfs install
git config --global lfs.concurrenttransfers 64
cd /marimo
nohup git clone https://huggingface.co/datasets/manhmitcf/Fish_Feeding_Intensity_Dataset > clone.log 2>&1 &
tail -n +1 -f clone.log
```

Dataset location must be `/marimo/Fish_Feeding_Intensity_Dataset`. If the clone
creates nested paths, normalize after the clone completes:

```bash
mv /marimo/Fish_Feeding_Intensity_Dataset/audio/audio/* /marimo/Fish_Feeding_Intensity_Dataset/audio/
mv /marimo/Fish_Feeding_Intensity_Dataset/video/video/* /marimo/Fish_Feeding_Intensity_Dataset/video/
rm -rf /marimo/Fish_Feeding_Intensity_Dataset/audio/audio
rm -rf /marimo/Fish_Feeding_Intensity_Dataset/video/video
```

## 3. Checkpoint and immutable splits

Download the PANNS archive for its immutable split CSV files and **all four**
video checkpoints. This lets you change `model.video_backbone` later without
downloading more files.

```bash
cd /marimo/Multimodal_Custom_Models
python -m pip install --upgrade "huggingface_hub[cli]"
mkdir -p checkpoints /tmp/uffia_checkpoints
hf download hoangphihung442004/Results_U_FFIA27K_audio PANNS_Cnn6_holdout_random_sample_20260729_153012.zip --repo-type dataset --local-dir /tmp/uffia_checkpoints
hf download hoangphihung442004/Results_U_FFIA27K_video DenseNet121_holdout_random_sample_20260729_153012.zip --repo-type dataset --local-dir /tmp/uffia_checkpoints
hf download hoangphihung442004/Results_U_FFIA27K_video EfficientNetB0_holdout_random_sample_20260804_181745.zip --repo-type dataset --local-dir /tmp/uffia_checkpoints
hf download hoangphihung442004/Results_U_FFIA27K_video MobileNetV2_holdout_random_sample_20260729_153012.zip --repo-type dataset --local-dir /tmp/uffia_checkpoints
hf download hoangphihung442004/Results_U_FFIA27K_video SwinTiny_holdout_random_sample_20260729_153012.zip --repo-type dataset --local-dir /tmp/uffia_checkpoints
mkdir -p checkpoints/PANNS_Cnn6_holdout_random_sample_20260729_153012 checkpoints/DenseNet121_holdout_random_sample_20260729_153012 checkpoints/EfficientNetB0_holdout_random_sample_20260804_181745 checkpoints/MobileNetV2_holdout_random_sample_20260729_153012 checkpoints/SwinTiny_holdout_random_sample_20260729_153012
unzip -q /tmp/uffia_checkpoints/PANNS_Cnn6_holdout_random_sample_20260729_153012.zip -d checkpoints/PANNS_Cnn6_holdout_random_sample_20260729_153012 -x '.git/*' '*/.git/*'
unzip -q /tmp/uffia_checkpoints/DenseNet121_holdout_random_sample_20260729_153012.zip -d checkpoints/DenseNet121_holdout_random_sample_20260729_153012 -x '.git/*' '*/.git/*'
unzip -q /tmp/uffia_checkpoints/EfficientNetB0_holdout_random_sample_20260804_181745.zip -d checkpoints/EfficientNetB0_holdout_random_sample_20260804_181745 -x '.git/*' '*/.git/*'
unzip -q /tmp/uffia_checkpoints/MobileNetV2_holdout_random_sample_20260729_153012.zip -d checkpoints/MobileNetV2_holdout_random_sample_20260729_153012 -x '.git/*' '*/.git/*'
unzip -q /tmp/uffia_checkpoints/SwinTiny_holdout_random_sample_20260729_153012.zip -d checkpoints/SwinTiny_holdout_random_sample_20260729_153012 -x '.git/*' '*/.git/*'
rm -rf /tmp/uffia_checkpoints
```

## 4. Architecture

```text
Audio 2s at 256 kHz
-> six windows: 0.75s, hop 0.25s
-> pre-emphasis 0.97
-> Librosa Hamming STFT: frame=n_fft=4096, hop=2048
-> log magnitude and temporal mean
-> 6 tokens x 2049D -> projection -> optional audio PE -> audio self-attention

Centre video frame -> original video checkpoint -> spatial grid tokens
-> optional visual PE -> visual self-attention

Visual Query + STFT Key/Value -> cross-attention -> mean pool -> 4 classes
```

Audio and visual positional encoding are independent. DINO and PANNS are not
part of this branch's model.

## 5. Install and run

```bash
cd /marimo/Multimodal_Custom_Models/midframe_audio_cross_attention
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

```bash
nohup python3 main.py > main.log 2>&1 &
tail -n +1 -f main.log
```

On first run, missing STFT feature files are precomputed in parallel with
`ProcessPoolExecutor` and saved under `data.stft_cache_dir`. Later runs reuse
the NPY cache.

## 6. Config

Edit `config/train_config.json` only. Audio defaults are 256000 Hz,
`n_fft=4096`, `frame_length=4096`, `hop_length=2048`, and pre-emphasis 0.97.
Set `model.audio_positional_encoding` to `learned`, `sinusoidal`, or `none`.
`model.visual_positional_encoding` controls only source-video grid tokens.
The supplied config enables `training.mixed_precision: true`; AMP activates
automatically when CUDA is available.
