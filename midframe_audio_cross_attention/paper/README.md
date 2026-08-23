# Tài liệu Chuyên sâu cho Phương pháp 3 (GW-AVF)

Thư mục này chứa các bài báo khoa học tiền đề làm cơ sở thiết kế cho **Phương pháp 3: Geometry & Water-Ripple Guided Audio-Visual Fusion (GW-AVF)**.

---

## 📑 Danh sách Bài báo Tham chiếu Chính

### 1. `2506.14170.pdf`
* **Tên bài báo**: *Progressive Multimodal Interaction Network for Reliable Quantification of Fish Feeding Intensity in Aquaculture*
* **Tác giả / Năm**: arXiv:2506.14170 (2025)
* **Ý nghĩa áp dụng**: Cung cấp bằng chứng thực nghiệm về việc dung hợp tín hiệu ảnh, âm thanh và sóng nước (water-wave) giúp tăng độ tin cậy khi định lượng cường độ cho cá ăn.

### 2. `2208.07011.pdf`
* **Tên bài báo**: *Automatic Controlling Fish Feeding Machine using Feature Extraction of Nutriment and Ripple Behavior*
* **Tác giả / Năm**: IEEE / arXiv:2208.07011 (2022)
* **Ý nghĩa áp dụng**: Cung cấp thuật toán trích xuất đặc trưng gợn sóng nước (Ripple Behavior Feature Extraction) và chuyển động bơi tán loạn để ra quyết định điều khiển máy cho cá ăn.

---

## 🏗️ Ứng dụng vào Mã nguồn `features/geometry_ripple_features.py`

* **Wavelet Water-Ripple Energy**: Dựa trên `2208.07011.pdf`, module `RippleWaveletExtractor` trích xuất thành phần tần số cao đại diện cho gợn sóng nhấp nhô trên mặt hồ.
* **Progressive Geometry Fusion**: Dựa trên `2506.14170.pdf`, module `GeometryRippleFeatureExtractor` dung hợp lưới không gian $14 \times 14$ đặc trưng hình học vào các visual spatial tokens trước khi đi qua Cross-Attention.
