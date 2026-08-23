# Tài liệu Chuyên sâu cho Phương pháp 4 (R-BPMD)

Thư mục này chứa các bài báo khoa học tiền đề làm cơ sở thiết kế cho **Phương pháp 4: Robust Bilinear Pooling with Modality Dropout (R-BPMD)**.

---

## 📑 Danh sách Bài báo Tham chiếu Chính

### 1. `2111.08910.pdf`
* **Tên bài báo**: *Information Fusion in Attention Networks Using Adaptive and Multi-level Factorized Bilinear Pooling for Audio-visual Emotion Recognition*
* **Tác giả / Năm**: IEEE / arXiv:2111.08910 (2021)
* **Ý nghĩa áp dụng**: Cung cấp thuật toán Multi-level Factorized Bilinear Pooling (MFB) để tính toán tương tác ma sát kép phi tuyến giữa các đặc trưng Audio và Visual thay cho phép ghép nối (concatenation) đơn thuần.

### 2. `2010.00734.pdf`
* **Tên bài báo**: *Training Strategies to Handle Missing Modalities for Audio-Visual Expression Recognition*
* **Tác giả / Năm**: arXiv:2010.00734 (2020)
* **Ý nghĩa áp dụng**: Cung cấp chiến lược huấn luyện chịu lỗi (Modality Dropout) bằng cách loại bỏ ngẫu nhiên 1 trong 2 kênh tín hiệu khi train, giúp mô hình duy trì độ chính xác cao ngay cả khi cảm biến/ống kính camera bị lóa hoặc âm thanh bị nhiễu.

---

## 🏗️ Ứng dụng vào Mã nguồn `features/robust_bilinear_fusion.py`

* **Factorized Bilinear Pooling (MFB)**: Dựa trên `2111.08910.pdf`, module `MultiFactorizedBilinearPooling` thực hiện tính tích Hadamard phần tử và sum-pooling theo factor $k=3$.
* **Modality Dropout**: Dựa trên `2010.00734.pdf`, module `ModalityDropout` loại bỏ ngẫu nhiên $15\%$ tín hiệu Audio hoặc Visual trong quá trình huấn luyện.
