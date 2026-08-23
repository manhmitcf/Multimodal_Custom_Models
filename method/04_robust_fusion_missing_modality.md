# Phương pháp 4: Robust Bilinear Pooling with Modality Dropout (R-BPMD)

## 1. Đặt vấn đề & Động lực (Motivation)
Trong ứng dụng thực tế tại các trang trại nuôi cá, kênh truyền dữ liệu hoặc cảm biến có thể gặp sự cố:
* **Kênh Audio bị hỏng/nhiễu**: Do tiếng ồn tiếng mưa lớn, gió mạnh hoặc tiếng máy bơm hoạt động át mất tiếng cá ăn.
* **Kênh Video bị lóa sáng/mờ camera**: Do bọt nước bắn lên ống kính camera, hoặc ánh nắng mặt trời phản chiếu lóa mặt hồ.

Nếu sử dụng phép nối đơn thuần (`torch.cat([audio, video])`), khi một kênh bị nhiễu nghiêm trọng, mô hình dễ bị phân loại sai hoàn toàn.

---

## 2. Kiến trúc Đề xuất

```text
Audio Tokens (6 x D)                          Visual Tokens (196 x D)
        │                                                │
   Audio Encoder                                   Visual Encoder
        │                                                │
        ▼                                                ▼
  Audio Embedding                                 Visual Embedding
        │                                                │
        └─────────────────┬──────────────────────────────┘
                          │
                Modality Dropout Layer
             (Randomly drop 15% Audio / Video)
                          │
          Multi-level Factorized Bilinear Pooling (MFB)
                          │
               Fused Multimodal Vector
                          │
                     Classifier
```

---

## 3. Các thành phần kỹ thuật cốt lõi

### 3.1. Multi-level Factorized Bilinear Pooling (MFB)
* **Ý tưởng từ paper**: *Adaptive and Multi-level Factorized Bilinear Pooling* ([2111.08910](https://arxiv.org/abs/2111.08910)).
* **Cơ chế**:
  * Thay vì chỉ ghép nối (Concatenation) hai véc-tơ $x$ và $y$, MFB tính toán tương tác ma trận ma sát kép (Bilinear Pooling) giữa Audio và Visual đặc trưng, sau đó nén lại bằng Low-rank Factorization:
    $$z = \text{SumPooling}\left( (W_x^T x) \circ (W_y^T y), k \right)$$
    với $\circ$ là Hadamard product (tích từng phần tử) và $k$ là hệ số factorize.
  * MFB cho phép mô hình học được các tương tác phi tuyến tính phức tạp hơn giữa các kênh.

### 3.2. Modality Dropout cho Huấn luyện Chịu lỗi (Robustness Training)
* **Ý tưởng từ paper**: *Handling Missing Modalities for Expression Recognition* ([2010.00734](https://arxiv.org/abs/2010.00734)).
* **Cơ chế**:
  * Trong quá trình huấn luyện, với xác suất $p_{\text{drop}} = 0.15$:
    * Đặt toàn bộ véc-tơ Audio về $0$ (giả lập mất tiếng/nhiễu audio).
    * Đặt toàn bộ véc-tơ Video về $0$ (giả lập nhòe camera/mất khung hình).
  * Việc này ép mô hình phải học cách tự đưa ra dự đoán hợp lý dựa trên kênh còn lại mà không phụ thuộc tuyệt đối vào việc phải có đủ cả 2 kênh cùng lúc.

---

## 4. Kế hoạch Thử nghiệm (Experiment Plan)
1. Thay thế `concat_classifier` trong `CrossAttentionHead` bằng module `MFBFusionLayer` tại [`models/fusion_model.py`](file:///C:/Users/manhm/Desktop/Multimodal_Custom_Models/midframe_audio_cross_attention/models/fusion_model.py).
2. Tích hợp `ModalityDropout` trong `BaselineSourceMultimodal.forward()`.
3. Đánh giá độ bền (Robustness Benchmark): So sánh độ suy giảm F1-score khi cố tình triệt tiêu kênh Audio hoặc kênh Video ở tập Holdout Test Split giữa mô hình Baseline và mô hình R-BPMD.
