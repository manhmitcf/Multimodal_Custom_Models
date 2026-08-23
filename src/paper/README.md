# Tài liệu Chuyên sâu cho Nhánh Single-Modality Audio Only (STFT-dB + PANNS CNN6)

Thư mục này chứa tài liệu nghiên cứu tiền đề làm cơ sở cho **Mô hình Chuyên biệt Kênh Âm thanh (Single-Modality Audio)** kết hợp giữa **PANNs CNN6 (Unfrozen fine-tuning)** và **Ảnh phổ STFT-dB ($224 \times 224$)**.

---

## 📑 Kiến trúc Tổng quan (Architectural Pipeline)

```text
               Audio Waveform 2s (64 kHz)
                    │              │
               PANNS CNN6    STFT-dB Spectrogram
               (Unfrozen)    (n_fft=2048 -> 3x224x224)
               (6 x 512)           │
                    │         STFT Feature Conv
                    │           (1 x 256)
                    └──────────────┬────────────────
                                   │
                     Transformer Self-Attention
                       (7 Audio/Spectral Tokens)
                                   │
                         Classifier (4 classes)
```

---

## 💡 Ý nghĩa Kỹ thuật (Key Technical Innovations)

1. **Khắc phục điểm nghẽn 85% F1 của PANNs CNN6**:
   * Tích hợp **Ảnh phổ STFT-dB ($224 \times 224$)** nén $256,000$ điểm phổ tần số cao, bổ sung $1$ token phổ sắc nét giúp trích xuất khoảnh khắc va chạm tần số khi cá đớp mồi ($2\text{ kHz} - 8\text{ kHz}$).
2. **Cho phép Fine-tuning Toàn bộ PANNs CNN6**:
   * Mở bằng toàn bộ tham số của PANNs CNN6 (`requires_grad = True`), giúp mô hình thích ứng sâu sắc với đặc trưng âm thanh dưới nước ao nuôi cá.
