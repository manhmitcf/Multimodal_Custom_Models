# Tài liệu Chuyên sâu cho Nhánh STFT 256k + PANNS CNN6 (Full Fine-Tune) & MobileNetV2 (Video) Advanced Fusion

Thư mục này chứa các tài liệu nghiên cứu tiền đề làm cơ sở cho **Phương pháp Dung hợp Đa thức Nâng cao**: Kết hợp **PANNs CNN6 Full Fine-Tune + STFT 256k Spectrogram High-Res (Kênh Âm thanh)** với **MobileNetV2 (Kênh Video Siêu Nhẹ)** thông qua **Spatial Cross-Attention, Factorized Bilinear MFB Pooling & Modality Dropout**.

---

## 📑 Kiến trúc Tổng quan (Architectural Pipeline)

```text
       Audio Waveform 2s (64 kHz)                     Middle Video Frame (RGB)
            │              │                                     │
       PANNS CNN6    STFT-dB 256k Image                     MobileNetV2
     (Full Fine-Tune)(n_fft=4096 -> 3x224x224)              Video Feature
       (6 x 512)           │                                 (1280)
            │         STFT Feature Conv                          │
            │           (1 x 256)                                │
            └──────────────┬─────────────────────────────────────┘
                           │
                Spatial Cross-Attention
             (Video Q attends 7 Audio Tokens)
                           │
             Factorized Bilinear MFB Fusion
                (Hadamard Product & Power Norm)
                           │
                 Modality Dropout (p=0.15)
                           │
                 Classifier (4 classes)
```

---

## 💡 Ý nghĩa Kỹ thuật (Key Technical Innovations)

1. **Kênh Âm thanh Siêu Cấp (PANNS CNN6 Full Fine-Tune + STFT 256k)**:
   * Cho phép PANNs CNN6 fine-tune $100\%$ tham số (`requires_grad = True`).
   * Tích hợp thêm **Ảnh phổ STFT-dB 256k ($224 \times 224$)** bổ sung $1$ token phổ high-res sắc nét.

2. **Kênh Video Siêu Nhẹ (MobileNetV2 Video Encoder)**:
   * Trích xuất đặc trưng video gọn nhẹ với MobileNetV2 (~3.5M tham số), tiết kiệm tối đa VRAM.

3. **Cơ chế Dung hợp Nâng cao (Advanced Multimodal Fusion)**:
   * **Spatial Cross-Attention**: Video Query ($1$ token) tương tác với $7$ Audio Tokens.
   * **Factorized Bilinear MFB Pooling**: Nhân ma sát kép Hadamard giữa Audio và Video, áp dụng Power & L2 Normalization.
   * **Modality Dropout ($p=0.15$)**: Giúp mô hình giữ nguyên độ chính xác cao ngay cả khi camera bị nhòe hoặc micro bị nhiễu.
