# Phương pháp 3: Geometry & Water-Ripple Guided Audio-Visual Fusion (GW-AVF)

## 1. Đặt vấn đề & Động lực (Motivation)
Các mô hình Deep Learning thuần túy (như ResNet hoặc SwinTransformer) thường học đặc trưng màu sắc/texture từ bức ảnh. Tuy nhiên, trong ao nuôi cá, màu nước hoặc điều kiện ánh sáng thay đổi theo thời tiết trong ngày có thể làm ảnh hưởng đến độ chính xác.

Bản chất của hoạt động cho cá ăn thể hiện qua 2 đặc trưng vật lý - sinh học rất rõ ràng:
1. **Đặc trưng Hình học Đàn cá (Flocking & Density Geometry)**: Khi cho ăn, cá tụ lại thành đám đông có tính chất hình học đặc thù.
2. **Đặc trưng Sóng nước (Water Ripples)**: Tiếng động khi cá bắt mồi tạo nên các gợn sóng nước hình tròn lan tỏa trên mặt hồ.

---

## 2. Kiến trúc Đề xuất

```text
Middle Video Frame (RGB)                          Audio (Log-Mel)
        │                                                │
 ┌──────┴──────────────────────────┐               Audio Encoder
 ▼                                 ▼                     │
RGB Deep Backbone            Geometric & Ripple          │
(Swin / ResNet)              Feature Extractor           │
 │ (Spatial Tokens)          (Delaunay / Wavelets)       │
 │                                 │                     │
 └─────────────────┬───────────────┘                     │
                   ▼                                     ▼
        Enhanced Visual Tokens ───────────────► Cross-Attention
                                                         │
                                               Multimodal Tokens
                                                         │
                                                    Classifier
```

---

## 3. Các thành phần kỹ thuật cốt lõi

### 3.1. Delaunay Triangulation & Flocking Index (FIFFB)
* **Ý tưởng từ paper**: *Computer Vision and Feeding Behavior Based Intelligent Feeding Controller* ([HAL-02124258](https://inria.hal.science/hal-02124258)), *FishPhenoKey* ([2405.12476](https://arxiv.org/abs/2405.12476)).
* **Cơ chế**:
  * Phát hiện các điểm ảnh đại diện cho cá trên mặt nước (hoặc các keypoint đầu cá/thân cá).
  * Xây dựng đồ thị tam giác Delaunay kết nối các vị trí cá.
  * Tính toán chỉ số mật độ đàn cá (Flocking Index - FIFFB):
    $$\text{FIFFB} = \frac{1}{N} \sum_{i=1}^{N} \text{Area}(\triangle_i)$$
    Khi cá tập trung ăn mạnh, diện tích các tam giác Delaunay giảm xuống cực nhỏ.

### 3.2. Trích xuất Đặc trưng Sóng nước (Water Ripple Wavelet Features)
* **Ý tưởng từ paper**: *Feature Extraction of Nutriment and Ripple Behavior* ([2208.07011](https://arxiv.org/abs/2208.07011)), *Progressive Multimodal Interaction* ([2506.14170](https://arxiv.org/abs/2506.14170)).
* **Cơ chế**:
  * Áp dụng biến đổi Wavelet 2D (2D Discrete Wavelet Transform - DWT) trên khung hình để trích xuất các thành phần tần số cao (High-frequency details) biểu diễn các đường gợn sóng nhấp nhô trên mặt nước.
  * Ghép véc-tơ năng lượng sóng nước vào cùng với các spatial visual tokens.

### 3.3. Dual-Stream Geometry-Guided Cross-Attention
* Nhánh Hình học - Sóng nước định hướng cho ma trận Attention: Các vị trí có chỉ số Flocking cao hoặc có gợn sóng nước xuất hiện sẽ nhận được trọng số ưu tiên cao hơn khi dung hợp với tín hiệu âm thanh 2s.

---

## 4. Kế hoạch Thử nghiệm (Experiment Plan)
1. Cài đặt module trích xuất Delaunay Triangulation & Wavelet Ripple Feature trong `U_FFIA27K_video/transforms/`.
2. Kiểm tra mức độ cải thiện khi thêm đặc trưng hình học vào mô hình dung hợp `midframe_audio_cross_attention`.
