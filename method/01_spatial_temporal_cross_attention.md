# Phương pháp 1: Spatial-Temporal Audio-Visual Cross-Attention (ST-AVCA)

## 1. Đặt vấn đề & Động lực (Motivation)
Trong mô hình Multimodal Baseline ban đầu (`exp/midframe-to-audio-cross-attn`), khung hình giữa video (Middle Video Frame) bị nén thành 1 vector duy nhất ($1 \times D$). Việc này loại bỏ hoàn toàn thông tin phân bố không gian (Spatial Distribution) – ví dụ: vị trí cá tập trung đớp thức ăn hoặc vùng nước bị xáo động mạnh trên mặt hồ.

Đồng thời, kênh audio dùng Log-Mel Spectrogram cố định có thể nhạy cảm với nhiễu từ máy bơm nước hoặc tiếng ồn xung quanh trong môi trường nuôi trồng thủy sản thực tế.

---

## 2. Kiến trúc Đề xuất

```text
Log-Mel / Waveform 2s                  Middle Video Frame (RGB)
        │                                         │
 PCEN / LEAF Frontend                     SwinTiny / DINOv2
 (Filterbank học được)                   (Spatial Patch Tokens)
        │                                         │
 Audio Encoder (PANNs/AST)                        │
        │                                         │
  6 Audio Tokens (K, V)                    196 Spatial Tokens (Q)
        │                                         │
        └─────────────────┬───────────────────────┘
                          │
            Cross-Attention Layer (Q=Spatial, K,V=Audio)
                          │
              Spatial Audio-Visual Tokens (196 x D)
                          │
                  Spatial Mean Pooling
                          │
                     Classifier (4 classes)
```

---

## 3. Các thành phần chính

### 3.1. Adaptable Audio Frontend (PCEN / LEAF)
* **Ý tưởng từ paper**: *Trainable Frontend* ([1607.05666](https://arxiv.org/abs/1607.05666)), *LEAF* ([2101.08596](https://arxiv.org/abs/2101.08596)).
* **Cơ chế**: Thay thế Log-Mel cố định bằng **PCEN (Per-Channel Energy Normalization)** hoặc **Gabor Filterbanks học được**.
  $$\text{PCEN}(t, f) = \left( \frac{E(t, f)}{(\epsilon + M(t, f))^\alpha} + \delta \right)^\gamma - \delta^\gamma$$
  Giúp tự động triệt tiêu các thành phần nhiễu nền tĩnh (tiếng bơm sủi khí) và làm nổi bật các xung âm thanh ngắn do cá đớp mồi.

### 3.2. Lưới Tokens Không gian (Spatial Visual Tokens)
* **Ý tưởng từ paper**: *TimeSformer* ([2103.06695](https://arxiv.org/abs/2103.06695)), *DINOv2* ([2304.07193](https://arxiv.org/abs/2304.07193)).
* **Cơ chế**: Sử dụng SwinTiny (Stage 3) hoặc DINOv2 để trích xuất lưới $14 \times 14 = 196$ spatial tokens ($S \in \mathbb{R}^{196 \times D}$). 
* Mỗi token đại diện cho một ô vuông nhỏ $16 \times 16$ px trên bề mặt nước.

### 3.3. Spatial-Temporal Cross-Attention Fusion
* **Ý tưởng từ paper**: *Audiovisual Transformer* ([1912.02615](https://arxiv.org/abs/1912.02615)), *Cross Attentional Fusion* ([2111.05222](https://arxiv.org/abs/2111.05222)).
* **Cơ chế Query-Key-Value**:
  * $Q = \text{Spatial Visual Tokens } (196 \times D)$
  * $K, V = \text{Temporal Audio Tokens } (6 \times D)$
* Ma trận chú ý:
  $$A = \text{Softmax}\left( \frac{Q K^T}{\sqrt{d}} \right) \in \mathbb{R}^{196 \times 6}$$
* **Ý nghĩa**: Mỗi vùng $16 \times 16$ px trên mặt nước sẽ học cách "lắng nghe" khoảng thời gian âm thanh tương ứng. Vùng nước có hiện tượng cá quẫy sóng sẽ kích hoạt attention cao hơn vào đúng thời điểm tiếng cá đớp mồi bùng phát.

---

## 4. Kế hoạch Thử nghiệm (Experiment Plan)
1. Thử nghiệm thay thế Audio Frontend bằng PCEN trên `U_FFIA27K_audio`.
2. Kiểm tra hiệu năng Cross-Attention giữa $196$ spatial tokens và $6$ audio tokens trên `midframe_audio_cross_attention`.
3. So sánh F1-score của ST-AVCA với baseline nén $1 \times D$ token.
