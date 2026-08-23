# Tài liệu Thiết kế Kiến trúc: STFT 256k Raw 2049 Dải Tần số với Frequency-Domain Attention & MobileNetV2 Factorized Bilinear Gated Fusion

**Nhánh Git**: `exp/stft256k-raw2049-freqattn-mobilenetv2-fbgf`  
**Ngày lập**: 23/08/2026  
**Tập dữ liệu Mục tiêu**: U-FFIA27K (27,000 mẫu đa thức âm thanh & hình ảnh ao nuôi cá)  
**Mục tiêu Chính**: Đạt F1 >95.93% với mô hình siêu nhẹ (~5.1M tham số) chạy End-to-End dựa trên Raw STFT 2049 Dải Tần số (Không nén Mel, Không tạo file ảnh RGB), Lọc Pre-Emphasis ($\alpha=0.97$), Cơ chế Chú ý Miền Tần số (F-Attention), Mạng Audio CNN Depthwise-Separable, Backbone Video MobileNetV2 và Dung hợp Factorized Bilinear Gated Fusion (FBGF / GMF).

---

## 1. Tóm tắt Tổng quan Kiến trúc Hệ thống

```text
       RAW AUDIO WAVEFORM 2s (SR 256 kHz)                                     MIDDLE FRAME RGB (224x224)
                   │                                                                      │
       Pre-Emphasis Filter (alpha=0.97)                                               MobileNetV2
                   │                                                               (Feature 1280d)
       Raw STFT 256k (4096, 2048, 2048)                                                       │
       (KHÔNG NÉN MEL - KHÔNG MÃ HÓA ẢNH)                                            Linear Projection
                   │                                                                (1280 -> 256d)
       Spectrogram dB [B, 1, 2049, 250]                                                       │
                   │                                                                Video Feature (256d)
       [FREQUENCY-DOMAIN ATTENTION (F-ATTN)]                                              │
       (Tự học ma trận trọng số a_F trong [0, 1]^2049)                                    │
                   │                                                                      │
       [DEPTHWISE-SEPARABLE AUDIO CNN]                                                    │
       (Conv Strided 4x2 ban đầu -> Các khối Depthwise)                                   │
                   │                                                                      │
         Audio Feature (256d)                                                             │
                   │                                                                      │
                   └───────────────────────────┬──────────────────────────────────────────┘
                                               │
                                [FACTORIZED BILINEAR GATED FUSION]
                                 1. MFB Bilinear Hadamard Product (k=3)
                                 2. Power & L2 Normalization
                                 3. Dynamic Gating Gate g = Sigmoid(W[x_a, x_v])
                                               │
                                     [CLASSIFIER 4 LỚP]
                                  Linear(256 -> 4) -> Logits
```

---

## 2. Thông số Toán học & Kỹ thuật Chi tiết các Mô đun

### 2.1 Chuỗi Xử lý Âm thanh (`src/features/audio_features.py`)
1. **Bộ lọc Thông cao Pre-Emphasis**:
   $$y[t] = x[t] - 0.97 \cdot x[t-1]$$
   Khuếch đại các vi xung âm thanh tần số cao ($2\text{ kHz} - 8\text{ kHz}$) của tiếng cá đớp mồi và dập tắt tần số trầm ($0 - 500\text{ Hz}$) nhiễu quạt nước.

2. **Biến đổi Raw STFT Transform (Không nén Mel)**:
   - `n_fft = 4096`
   - `win_length = 4096` (frame_length)
   - `hop_length = 2048`
   - `window = "hamming"`
   - Tính toán STFT phức $\mathbf{Z} \in \mathbb{C}^{B \times 2049 \times 250}$.
   - Tính toán Phổ Log-Power theo dải dB: $\mathbf{S} = 10 \cdot \log_{10}(|\mathbf{Z}|^2 + 10^{-10}) \in \mathbb{R}^{B \times 1 \times 2049 \times 250}$.

3. **Cơ chế Chú ý Miền Tần số (Frequency-Domain Attention - F-Attention Module)**:
   - Nén trục thời gian $T=250$ qua `Mean` và `Std` năng lượng:
     $$\mathbf{v}_F = \left[ \text{Mean}_T(\mathbf{S}) \,\|\, \text{Std}_T(\mathbf{S}) \right] \in \mathbb{R}^{B \times 4098}$$
   - Đưa qua MLP 2 lớp với hàm kích hoạt Sigmoid:
     $$\mathbf{a}_F = \sigma \left( \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \mathbf{v}_F) \right) \in [0, 1]^{B \times 1 \times 2049 \times 1}$$
   - Nhân trọng số chú ý miền tần số:
     $$\mathbf{S}_{\text{attended}} = \mathbf{a}_F \odot \mathbf{S}$$

### 2.2 Mạng Audio CNN Depthwise-Separable (`src/models/custom_audio_cnn.py`)
- **Khối 1 (Early Strided Conv)**: `Conv2d(1 -> 32, kernel=(5, 5), stride=(4, 2), padding=2)`
  - Nén ngay kích thước trục tần số từ $2049 \rightarrow 513$ ở ms tính toán đầu tiên để triệt tiêu việc phình to VRAM.
- **Khối 2**: `DepthwiseSeparableConv2d(32 -> 64, stride=(2, 2))` $\rightarrow [B, 64, 257, 63]$.
- **Khối 3**: `DepthwiseSeparableConv2d(64 -> 128, stride=(2, 2))` $\rightarrow [B, 128, 129, 32]$.
- **Khối 4**: `DepthwiseSeparableConv2d(128 -> 256, stride=(2, 2))` $\rightarrow [B, 256, 65, 16]$.
- **Global Pooling**: `AdaptiveAvgPool2d((1, 1))` $\rightarrow$ Vector Audio $\mathbf{x}_a \in \mathbb{R}^{B \times 256}$. Tổng tham số Kênh Audio ~1.4M.

### 2.3 Bộ Mã hóa Hình ảnh (`MobileNetV2`)
- Đầu vào: Khung hình giữa RGB $[B, 3, 224, 224]$.
- Backbone: `MobileNetV2` ($\sim 3.5\text{M}$ tham số).
- Lớp Chiếu Tuyến tính (Linear Projection): `Linear(1280 -> 256)` $\rightarrow$ Vector Video $\mathbf{x}_v \in \mathbb{R}^{B \times 256}$.

### 2.4 Bộ Dung hợp Đa thức (`src/models/fusion_model.py`)
Chuyển đổi linh hoạt qua cờ `"fusion_type": "fbgf"` (Mặc định) hoặc `"fusion_type": "gmf"`:

#### Chế độ 1: Factorized Bilinear Gated Fusion (FBGF)
1. **Phân rã Bilinear Low-Rank ($k=3$)**:
   $$\tilde{\mathbf{x}}_a = \mathbf{W}_a \mathbf{x}_a \in \mathbb{R}^{256 \times 3}, \quad \tilde{\mathbf{x}}_v = \mathbf{W}_v \mathbf{x}_v \in \mathbb{R}^{256 \times 3}$$
   $$\mathbf{z} = \text{SumPool}_3 \Big( \tilde{\mathbf{x}}_a \circ \tilde{\mathbf{x}}_v \Big) \in \mathbb{R}^{256}$$
2. **Power & L2 Normalization**:
   $$\mathbf{z}_{\text{norm}} = \text{L2Norm} \left( \text{Sign}(\mathbf{z}) \odot \sqrt{|\mathbf{z}| + 1e-10} \right)$$
3. **Cổng Gated Dynamic Routing**:
   $$\mathbf{g} = \sigma \left( \mathbf{W}_g [\mathbf{x}_a \,\|\, \mathbf{x}_v] + \mathbf{b}_g \right)$$
4. **Vector Dung hợp Cuối cùng**:
   $$\mathbf{h}_{\text{fused}} = \mathbf{g} \odot \mathbf{z}_{\text{norm}} + (\mathbf{1} - \mathbf{g}) \odot \text{ReLU}(\mathbf{W}_v' \mathbf{x}_v)$$

#### Chế độ 2: Gated Multimodal Fusion (GMF)
$$\mathbf{g} = \sigma \left( \mathbf{W}_g [\mathbf{x}_a \,\|\, \mathbf{x}_v] + \mathbf{b}_g \right)$$
$$\mathbf{h}_{\text{fused}} = \mathbf{g} \odot \text{ReLU}(\mathbf{W}_a \mathbf{x}_a) + (\mathbf{1} - \mathbf{g}) \odot \text{ReLU}(\mathbf{W}_v \mathbf{x}_v)$$

### 2.5 Bộ Phân loại (Classifier)
$$\text{Logits} = \text{Linear}(256 \rightarrow 4) \in \mathbb{R}^{B \times 4}$$

---

## 3. Cấu trúc Thư mục & Tiêu chuẩn Dự án

Tất cả mã nguồn nằm gọn trong `src/`:
- `src/main.py`: Entry point duy nhất.
- `src/settings.py`: Bộ đọc cấu hình & phân giải đường dẫn.
- `src/config/train_config.json`: Cấu hình chứa các siêu tham số `stft` (`sr: 256000`, `pre_emphasis: 0.97`, `frame_length: 4096`, `hop_length: 2048`, `n_fft: 4096`, `windowing: "hamming"`, `use_std: true`) và cờ `"fusion_type": "fbgf"`.
- `src/config/artifact_upload_config.json`: Cấu hình nén zip tự động lên Hugging Face repository `manhmitcf/fish_result`.
- `src/dataset/paired_loader.py`: Dataloader hỗ trợ RAM cache với `cache_audio: true` và `video_cache_mode: "ram"`.
- `src/features/audio_features.py`: Pre-Emphasis, Raw STFT 2049, FrequencyDomainAttention.
- `src/models/custom_audio_cnn.py`: Depthwise-Separable Audio CNN.
- `src/models/fusion_model.py`: Mô hình Đa thức & Fusion Head.
- `src/tasks/trainer.py`: MultimodalTrainer tự động lưu kết quả vào `src/checkpoint/`.
- `src/paper/README.md`: Tài liệu báo cáo của nhánh.
- `src/tests/test_custom_multimodal_model.py`: Unit tests bằng PyTest.

---

## 4. Kế hoạch Kiểm thử & Đầu ra Kết quả

1. **Kiểm thử Unit Tests**:
   - Chạy `python -m pytest tests/` trong `src/` để đảm bảo 100% kích thước tensor, luồng lan truyền ngược gradient và số lượng tham số hoạt động chuẩn xác.
2. **Tiêu chuẩn Kết quả tại `src/checkpoint/`**:
   - Đảm bảo tự động sinh ra `splits/`, `best.pt`, `history.csv`, `summary_results.csv`, `best_val_metrics.json`, `test_metrics.json`, `best_val_confusion_matrix.csv`, và `test_confusion_matrix.csv`.
3. **Đóng gói & Tải lên Remote**:
   - Tự động nén `STFT256k_Raw2049_FreqAttn_MobileNetV2_Artifacts.zip` và đẩy về Hugging Face `manhmitcf/fish_result`.
