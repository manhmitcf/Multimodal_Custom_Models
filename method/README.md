# Tổng hợp Kiến thức Paper & Đề xuất Phương pháp Multimodal Fish-Feeding (U-FFIA27K)

Tài liệu này tổng hợp toàn bộ tri thức từ hơn 50 bài báo nghiên cứu thuộc thư mục [`papers/`](file:///C:/Users/manhm/Desktop/Multimodal_Custom_Models/papers) và đề xuất các phương pháp/kiến trúc học máy đa thức (Multimodal AI) ứng dụng cho bài toán **đánh giá cường độ cho cá ăn (Fish Feeding Intensity Assessment)** trên bộ dữ liệu **U-FFIA27K**.

---

## 📚 1. Tổng quan & Tổng hợp Tri thức từ 5 Nhóm Paper chính

### 🐟 Nhóm 1: Fish Feeding & Smart Aquaculture (`01_fish_feeding_aquaculture/`)
* **Đặc trưng miền (Domain Knowledge)**: Khi cá bắt mồi (feeding activity), các tín hiệu xuất hiện đồng thời trên 2 kênh:
  * **Visual (Thị giác)**: Sự thay đổi mật độ đàn cá trên bề mặt nước, tốc độ bơi, gợn sóng nước (water ripples), và chuyển động bơi tán loạn/tập trung.
  * **Audio (Âm thanh)**: Tiếng cá đớp thức ăn, tiếng nước bọt quẫy tạo ra âm thanh dạng xung thời gian ngắn tần số trung-cao.
* **Bài học rút ra**: 
  * Cần mô hình hóa cả **chuyển động không gian mặt nước** và **đặc trưng sóng nước/gợn sóng** thay vì chỉ trích xuất véc-tơ màu sắc tĩnh.
  * Phản ứng cho ăn diễn ra rất nhanh (trong khoảng 1 - 2 giây), việc trích xuất khung hình giữa (Middle Frame) kết hợp cửa sổ thời gian âm thanh là hoàn toàn phù hợp với thực tế sinh học.

---

### 🔀 Nhóm 2: Audio-Visual Fusion Architectures (`02_audio_visual_fusion/`)
* **Cross-Attention & Co-Attention**:
  * *Audiovisual Transformer* ([1912.02615](https://arxiv.org/abs/1912.02615)): Sử dụng Encoder-Decoder Transformer để đồng bộ hóa chuỗi event audio và video frames.
  * *Joint Cross-Attention* ([2203.14779](https://arxiv.org/abs/2203.14779), [2111.05222](https://arxiv.org/abs/2111.05222)): Dùng Query của kênh này để tìm kiếm thông tin phụ trợ từ Key/Value của kênh kia.
* **Bilinear Pooling & Multi-level Fusion**:
  * *Multi-level Factorized Bilinear Pooling (MFB)* ([2111.08910](https://arxiv.org/abs/2111.08910)): Thay thế phép nối (concatenation) hoặc cộng (addition) đơn thuần bằng tương tác tích chéo đa chiều (bilinear interaction) giữa các kênh.
* **Xử lý trượt Modality (Missing / Noisy Modalities)**:
  * *Missing Modality Training* ([2010.00734](https://arxiv.org/abs/2010.00734)): Sử dụng Modality Dropout hoặc Cross-Modal Knowledge Distillation để mô hình không bị sụp đổ khi một trong hai kênh (tiếng nước mờ hoặc video bị lóa sáng) bị suy giảm chất lượng.

---

### 🐒 Nhóm 3: Animal Behavior Recognition (`03_animal_behavior/`)
* **Animal Behavior Benchmarks** (*Animal Kingdom* [2204.08129](https://arxiv.org/abs/2204.08129), *AnimalMotionCLIP* [2505.00569](https://arxiv.org/abs/2505.00569)):
  * Hành vi động vật thường phụ thuộc chặt chẽ vào **biểu diễn chuyển động (Motion Embedding)**.
  * Việc áp dụng CLIP-style contrastive learning giúp mô hình phân biệt được sự tương quan giữa cử động cơ thể và âm thanh phát ra.

---

### 🛠️ Nhóm 4: Unimodal Feature Engineering (`04_unimodal_fish_feature_engineering/`)
* **Audio Frontend Adaptations**:
  * *PCEN (Per-Channel Energy Normalization)* ([1607.05666](https://arxiv.org/abs/1607.05666)) & *LEAF* ([2101.08596](https://arxiv.org/abs/2101.08596)): Thay thế Log-Mel Spectrogram cố định bằng front-end học được, giúp nén nhiễu nền tiếng sủi bọt/bơm nước trong bể nuôi.
  * *PANNs* ([1912.10211](https://arxiv.org/abs/1912.10211)), *AST* ([2104.01778](https://arxiv.org/abs/2104.01778)), *PaSST* ([2211.13956](https://arxiv.org/abs/2211.13956)): Các backbone âm thanh hiện đại trích xuất đặc trưng thời gian-tần số vượt trội.
* **Video & Geometry Feature Engineering**:
  * *Delaunay Triangulation & Flocking Index (FIFFB)* ([HAL-02124258](https://inria.hal.science/hal-02124258)): Xây dựng đồ thị hình học nối các vị trí cá để tính mật độ tập trung.
  * *SlowFast Networks* ([1812.03982](https://arxiv.org/abs/1812.03982)) & *VideoMAE* ([2203.12602](https://arxiv.org/abs/2203.12602)): Tách biệt luồng không gian (Spatial/Appearance) và luồng nhịp độ chuyển động (Fast/Temporal).

---

### 🧩 Nhóm 5: Self-Supervised Learning & Vision Transformers (`papers/paper/`)
* **iBOT** ([iBOT_2021](https://arxiv.org/abs/2111.07832)) & **DINO / DINOv2** ([DINOv2_2023](https://arxiv.org/abs/2304.07193)):
  * Sử dụng Online Tokenizer kết hợp Masked Image Modeling để học đặc trưng ngữ nghĩa tự giám sát mà không cần nhãn.
* **MAE (Masked Autoencoders)** ([MAE_2021](https://arxiv.org/abs/2111.06377)) & **I-JEPA** ([I-JEPA_2023](https://arxiv.org/abs/2301.08243)):
  * Dự đoán đặc trưng trong không gian ẩn (latent space) từ các vùng ảnh/video bị che, giúp backbone visual đạt khả năng tổng quát hóa cao đối với ảnh bị lóa nước/nhiễu bóng mờ.

---

## 💡 2. Định hướng các Ý tưởng Phương pháp Kết hợp (Method Proposals)

Dựa trên việc tổng hợp các kiến thức trên, chúng tôi đề xuất 4 phương pháp cải tiến kiến trúc Multimodal được trình bày chi tiết trong các file riêng thuộc thư mục `method/`:

| File phương pháp | Tên phương pháp đề xuất | Ý tưởng cốt lõi & Sự kết hợp |
| :--- | :--- | :--- |
| 📄 [`01_spatial_temporal_cross_attention.md`](file:///C:/Users/manhm/Desktop/Multimodal_Custom_Models/method/01_spatial_temporal_cross_attention.md) | **Spatial-Temporal Audio-Visual Cross-Attention (ST-AVCA)** | Kết hợp Lưới 196 Spatial Tokens (SwinTiny/DINOv2) với Cửa sổ âm thanh thích nghi (PCEN/LEAF + PANNS). |
| 📄 [`02_masked_multimodal_ssl.md`](file:///C:/Users/manhm/Desktop/Multimodal_Custom_Models/method/02_masked_multimodal_ssl.md) | **Masked Multimodal Self-Supervised Pretraining (M2-SSL)** | Tích hợp iBOT/VideoMAE pretraining trên Visual Spatial Tokens + Audio Patchout trên tập Unlabeled Train Data. |
| 📄 [`03_geometry_motion_fusion.md`](file:///C:/Users/manhm/Desktop/Multimodal_Custom_Models/method/03_geometry_motion_fusion.md) | **Geometry & Water-Ripple Guided Audio-Visual Fusion (GW-AVF)** | Trích xuất đặc trưng hình học gợn sóng/Delaunay Flocking kết hợp với Cross-Attention Âm thanh - Thị giác. |
| 📄 [`04_robust_fusion_missing_modality.md`](file:///C:/Users/manhm/Desktop/Multimodal_Custom_Models/method/04_robust_fusion_missing_modality.md) | **Robust Bilinear Pooling with Modality Dropout (R-BPMD)** | Thay phép Concatenation bằng Factorized Bilinear Pooling (MFB) và bổ sung Modality Dropout chống suy giảm tín hiệu. |

---

## 🚀 3. Lộ trình Triển khai Thử nghiệm (Experiment Roadmap)

```text
[Giai đoạn 1] Tiền xử lý & Adapt Frontend Audio (PCEN/LEAF) và Visual (SwinTiny / DINOv2)
                                 │
                                 ▼
[Giai đoạn 2] Pretraining Tự giám sát không nhãn (iBOT / M2-SSL) trên train.csv
                                 │
                                 ▼
[Giai đoạn 3] Huấn luyện Dung hợp Multimodal Cross-Attention & Factorized Bilinear Pooling
                                 │
                                 ▼
[Giai đoạn 4] Thử nghiệm Độ bền (Robustness Test) với Modality Dropout trên Holdout Test Split
```
