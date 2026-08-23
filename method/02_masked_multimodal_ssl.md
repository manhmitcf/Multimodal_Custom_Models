# Phương pháp 2: Masked Multimodal Self-Supervised Pretraining (M2-SSL)

## 1. Đặt vấn đề & Động lực (Motivation)
Việc gán nhãn thủ công cho bài toán cường độ cho cá ăn (Feeding Intensity) trong môi trường nuôi trồng thủy sản rất tốn kém và có tính chủ quan cao. Tuy nhiên, tập dữ liệu không nhãn (Unlabeled Clips) thu được từ camera và hydrophone/micro dưới nước lại vô cùng phong phú.

Tận dụng các kỹ thuật Tự giám sát (Self-Supervised Learning - SSL) mới nhất từ Vision & Audio Transformers giúp mô hình học được các biểu diễn đặc trưng không gian-thời gian mạnh mẽ trước khi đi vào phân loại có giám sát.

---

## 2. Kiến trúc & Quy trình Pretraining Đề xuất

```text
       Unlabeled Training Clips (Audio + Video)
                          │
       ┌──────────────────┴──────────────────┐
       ▼                                     ▼
Audio Stream (Log-Mel)             Middle Frame (RGB Image)
       │                                     │
Audio Patchout / Masking (30%)     Spatial Token Masking (50%)
       │                                     │
Audio Encoder                      Visual Encoder (Swin / ViT)
       │                                     │
 Context Tokens                       Spatial Tokens
       └──────────────────┬──────────────────┘
                          │
          Cross-Attention Projection Layer
                          │
         iBOT / Masked Feature Prediction Loss
```

---

## 3. Các thành phần kỹ thuật cốt lõi

### 3.1. iBOT-style Masked Visual Token Prediction
* **Ý tưởng từ paper**: *iBOT* ([2111.07832](https://arxiv.org/abs/2111.07832)), *MAE* ([2111.06377](https://arxiv.org/abs/2111.06377)), *VideoMAE* ([2203.12602](https://arxiv.org/abs/2203.12602)).
* **Cơ chế**:
  * Che ngẫu nhiên **40% - 60%** các ô không gian (spatial patches) trên khung hình giữa (Middle Frame).
  * Mô hình bao gồm 1 Student Network và 1 Teacher Network (cập nhật theo cơ chế EMA - Exponential Moving Average).
  * Mục tiêu Loss: Khôi phục lại đặc trưng biểu diễn phân bố xác suất (online tokenization) tại các vị trí ô bị che.
  * **Đảm bảo tính vẹn toàn dữ liệu**: Quá trình pre-training này **chỉ thực hiện trên khung hình từ tập `train.csv`** và hoàn toàn không dùng nhãn feeding intensity.

### 3.2. Audio Patchout & SpecAugment SSL
* **Ý tưởng từ paper**: *PaSST* ([2211.13956](https://arxiv.org/abs/2211.13956)), *BYOL for Audio* ([2103.06695](https://arxiv.org/abs/2103.06695)).
* **Cơ chế**:
  * Áp dụng **Patchout** (loại bỏ ngẫu nhiên 2 cọc thời gian hoặc dải tần số trên Mel-Spectrogram).
  * Bắt mô hình khôi phục đặc trưng âm thanh tương ứng bằng Cross-Modal Prediction từ kênh thị giác (Visual context trợ giúp dự đoán đặc trưng audio bị khuyết).

### 3.3. Fine-tuning Đa thức Có giám sát (Supervised Multimodal Fine-tuning)
* Sau khi hoàn tất Pretraining SSL, nạp trọng số pre-trained cho cả 2 nhánh Audio Encoder và Visual Encoder.
* Tiến hành fine-tune có giám sát 4 lớp cường độ ăn (`unfed`, `low`, `medium`, `high`) với nhãn từ `train.csv`, kiểm tra Macro-F1 trên `val.csv` để chọn `best.pt`, và reload đánh giá 1 lần trên `test.csv`.

---

## 4. Kế hoạch Thử nghiệm (Experiment Plan)
1. Sử dụng 4 file cấu hình thí nghiệm phân cấp sẵn có tại `config/`:
   * `train_config_01_frozen_light.json`
   * `train_config_02_frozen_standard.json`
   * `train_config_03_tune_gentle.json`
   * `train_config_04_tune_strong_mask.json`
2. So sánh kết quả Macro-F1 giữa mô hình fine-tune có SSL Pretraining (M2-SSL) với mô hình Supervised thuần túy (Supervised Baseline).
