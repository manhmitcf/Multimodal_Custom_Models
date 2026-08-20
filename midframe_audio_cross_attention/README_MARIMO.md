# SwinTiny iBOT-inspired spatial pretraining + audio fusion (Marimo)

This branch starts from the existing supervised SwinTiny checkpoint, adapts it
without feeding-intensity labels on centre frames from the immutable **training
split only**, then trains the four-class audio-video model. Run everything
from this directory with one command: `python main.py`.

## 1. Clone this branch

```bash
cd /marimo
git clone --branch exp/swin-tiny-ibot-spatial-pretrain --single-branch https://github.com/manhmitcf/Multimodal_Custom_Models.git
cd Multimodal_Custom_Models
git branch --show-current
```

The last command must print `exp/swin-tiny-ibot-spatial-pretrain`.

## 2. Download the dataset

```bash
sudo apt update && sudo apt install git-lfs unzip -y
git lfs install
git config --global lfs.concurrenttransfers 64
cd /marimo
nohup git clone https://huggingface.co/datasets/manhmitcf/Fish_Feeding_Intensity_Dataset > clone.log 2>&1 &
tail -n +1 -f clone.log
```

The final dataset path must be `/marimo/Fish_Feeding_Intensity_Dataset`. If
the Hugging Face clone created nested directories, normalize them after clone:

```bash
mv /marimo/Fish_Feeding_Intensity_Dataset/audio/audio/* /marimo/Fish_Feeding_Intensity_Dataset/audio/
mv /marimo/Fish_Feeding_Intensity_Dataset/video/video/* /marimo/Fish_Feeding_Intensity_Dataset/video/
rm -rf /marimo/Fish_Feeding_Intensity_Dataset/audio/audio
rm -rf /marimo/Fish_Feeding_Intensity_Dataset/video/video
```

## 3. Download checkpoints and immutable split files

The default configuration needs only the PANNS Cnn6 archive and SwinTiny video
archive.

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

Confirm the files before running:

```bash
test -f checkpoints/PANNS_Cnn6_holdout_random_sample_20260729_153012/DL_audio/checkpoint/panns_cnn6/audio_best.pt && echo "audio checkpoint OK"
test -f checkpoints/SwinTiny_holdout_random_sample_20260729_153012/DL_video/checkpoint/swin_tiny/video_best.pt && echo "Swin checkpoint OK"
test -f checkpoints/PANNS_Cnn6_holdout_random_sample_20260729_153012/DL_audio/checkpoint/panns_cnn6/splits/train.csv && echo "immutable split OK"
```

## 4. Install and run

```bash
cd /marimo/Multimodal_Custom_Models/midframe_audio_cross_attention
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

For a background run:

```bash
nohup python3 main.py > main.log 2>&1 &
tail -n +1 -f main.log
```

## What `main.py` does

```text
Existing supervised SwinTiny checkpoint
  -> iBOT-inspired adaptation on centre frames from train.csv only
  -> Swin stage-3 grid: 14 x 14 = 196 spatial tokens, 384 dimensions
  -> PANNS Cnn6 emits six 512-dimensional audio tokens
  -> visual tokens are conditioned on audio tokens
  -> mean pool and classify four feeding-intensity classes
```

The iBOT adaptation has no feeding-intensity loss and never reads frames named
by `val.csv` or `test.csv`. Only after adaptation does supervised multimodal
training run: train, choose `best.pt` by validation macro-F1, reload it, and
evaluate the holdout test split once.

## Configuration

Edit only `config/train_config.json`.

- `model.video_backbone` must remain `swin_tiny`.
- `model.encoder_mode: "frozen"` trains only fusion after adaptation. Set it
  to `"tune"` to update PANNS late layers and Swin stages 3/4 with the lower
  `training.encoder_learning_rate`.
- `ibot_pretraining.enabled` enables the label-free phase.
- If CUDA runs out of memory, reduce `ibot_pretraining.batch_size` from `32`
  to `16` or `8` before changing the supervised batch size.
- `data.split_dir` must point to the archived PANNS `splits/` directory. Never
  regenerate or overwrite the split.

## Ordered experiment configurations

Four ready-to-run sweeps are in `config/`. They preserve the same checkpoint,
split, seed, and data policy; only iBOT strength and supervised fine-tuning
change. Run exactly one at a time:

```bash
python main.py --config config/train_config_01_frozen_light.json
python main.py --config config/train_config_02_frozen_standard.json
python main.py --config config/train_config_03_tune_gentle.json
python main.py --config config/train_config_04_tune_strong_mask.json
```

- `01`: conservative 30-epoch iBOT adaptation; frozen encoders.
- `02`: standard 50-epoch iBOT adaptation; frozen encoders.
- `03`: standard adaptation, then gentle Swin/PANNS late-layer fine-tuning.
- `04`: longer, higher-mask adaptation, then slower fine-tuning; use this only
  when GPU memory supports the smaller configured batches.

## Outputs

The default run writes these files under
`runs/swin_tiny_ibot_stage3_cross_attn/`:

```text
ibot_pretraining/ibot_pretrain_best.pt
best.pt
best_val_metrics.json
best_val_confusion_matrix.csv
test_metrics.json
test_confusion_matrix.csv
```

Keep the iBOT checkpoint, final metrics, JSON config, commit hash, seed, and
batch sizes together when reporting the experiment.
