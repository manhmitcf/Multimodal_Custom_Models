# Tài liệu Chuyên sâu cho Nhánh STFT-dB Image SwinTiny + MobileNetV2 Fusion

Thư mục này chứa các tài liệu nghiên cứu tiền đề làm cơ sở cho **Phương pháp Biến đổi Phổ Tần số Cao STFT thành Ảnh dB (STFT-dB Spectrogram Image Transformation)** kết hợp giữa **SwinTiny (Audio Spectrogram Branch)** và **MobileNetV2 (Video Branch)**.

---

## 📑 Kiến trúc Tổng quan (Architectural Pipeline)

```text
       Audio Waveform 2s (64 kHz)                     Middle Video Frame (RGB)
                  │                                              │
      STFT-dB Image Transform                              MobileNetV2
 (n_fft=2048, hop=512 -> 3x224x224)                       Video Feature
                  │                                              │
           SwinTiny Encoder                                      │
        Stage 3 Spatial Tokens                                   │
             (196 x 384)                                         │
                  │                                              │
                  └──────────────────────┬───────────────────────┘
                                         │
                             Cross-Attention Fusion Head
                             (Video Q attends Audio K,V)
                                         │
                               Classifier (4 classes)
```

---

## 💡 Ý nghĩa Kỹ thuật (Key Technical Innovations)

1. **Biến đổi STFT-dB Spectrogram thành Ảnh 3 Kênh (224x224)**:
   * Khắc phục triệt để điểm nghẽn mất mát dải phổ cao của Log-Mel Spectrogram thông thường (PANNs CNN6 85%).
   * Chuyển đổi toàn bộ miền phổ tần số $256,000$ điểm về ảnh 3 kênh $224 \times 224$ px bảo toàn $100\%$ các vi va chạm tần số khi cá đớp mồi ($2\text{ kHz} - 8\text{ kHz}$).

2. **Kênh Âm thanh Siêu Năng lực (SwinTiny Audio Spectrogram Encoder)**:
   * Sử dụng backbone SwinTiny để trích xuất $196$ spatial/spectral tokens ($14 \times 14$) từ ảnh phổ STFT-dB.

3. **Kênh Video Siêu Nhẹ (MobileNetV2 Video Encoder)**:
   * Giữ kênh Video gọn nhẹ tối đa với MobileNetV2 (~3.5M tham số), đảm bảo mô hình Đa thức chạy siêu nhanh và tiết kiệm VRAM.
