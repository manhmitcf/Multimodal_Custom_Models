# Tài liệu Chuyên sâu cho Nhánh STFT-dB + PANNS CNN6 (Audio) & MobileNetV2 (Video) Fusion

Thư mục này chứa các tài liệu nghiên cứu tiền đề làm cơ sở cho **Phương pháp Dung hợp Phổ Tần số Cao STFT-dB** kết hợp giữa **PANNs CNN6 (Audio Branch)** và **MobileNetV2 (Video Branch)**.

---

## 📑 Kiến trúc Tổng quan (Architectural Pipeline)

```text
       Audio Waveform 2s (64 kHz)                     Middle Video Frame (RGB)
            │              │                                     │
       PANNS CNN6    STFT-dB Image                           MobileNetV2
      Audio Tokens   (n_fft=2048 -> 3x224x224)              Video Feature
       (6 x 512)           │                                 (1280)
            │         STFT Feature Conv                          │
            │           (1 x 256)                                │
            └──────────────┬─────────────────────────────────────┘
                           │
               Cross-Attention Fusion Head
             (Video Q attends 7 Audio Tokens)
                           │
                 Classifier (4 classes)
```

---

## 💡 Ý nghĩa Kỹ thuật (Key Technical Innovations)

1. **Kênh Âm thanh Kép (PANNS CNN6 + STFT-dB Spectrogram)**:
   * Sử dụng **PANNs CNN6** pretrained tốt nhất làm Audio Token Encoder chính ($6$ tokens $512$d).
   * Bổ sung mô-đun biến đổi **STFT-dB Spectrogram ($224 \times 224$)** trích xuất đặc trưng phổ cao nén $256,000$ điểm phổ, bổ sung $1$ token phổ sắc nét giúp khắc phục triệt để mốc $85\%$ F1.

2. **Kênh Video Siêu Nhẹ (MobileNetV2 Video Encoder)**:
   * Giữ kênh Video gọn nhẹ tối đa với MobileNetV2 (~3.5M tham số), đảm bảo mô hình Đa thức chạy siêu nhanh và tiết kiệm VRAM.
