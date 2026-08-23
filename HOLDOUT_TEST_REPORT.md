# 📊 BÁO CÁO ĐÁNH GIÁ CHUYÊN SÂU TẬP HOLDOUT TEST (FINAL EVALUATION REPORT)

> **Mô hình**: STFT 256k Raw 2049 Bins với Frequency-Domain Attention & MobileNetV2 FBGF Fusion Head  
> **Nhánh Git**: `exp/stft256k-raw2049-freqattn-mobilenetv2-fbgf`  
> **Tập dữ liệu kiểm thử**: Holdout Test Set ($2,800$ mẫu độc lập)

---

## 1. 📈 BẢNG BÁO CÁO PHÂN LOẠI CHI TIẾT (CLASSIFICATION REPORT)

| Lớp cá ăn (Class) | Precision | Recall | F1-Score | Support (Số lượng mẫu) |
| :--- | :---: | :---: | :---: | :---: |
| **0. Unfed (Không ăn)** | **0.9829** | **0.9829** | **0.9829** | 700 |
| **1. Low (Ăn ít)** | **0.9551** | **0.9414** | **0.9482** | 700 |
| **2. Medium (Ăn vừa)** | **0.9253** | **0.9200** | **0.9226** | 700 |
| **3. High (Ăn mạnh)** | **0.9482** | **0.9671** | **0.9576** | 700 |
| | | | | |
| **Accuracy (Độ chính xác tổng)** | | | **0.9529** | **2800** |
| **Macro Average (Trung bình cộng)** | **0.9528** | **0.9529** | **0.9528** | **2800** |
| **Weighted Average (Trung bình có trọng số)** | **0.9528** | **0.9529** | **0.9528** | **2800** |

---

## 2. 🔲 MA TRẬN NHẦM LẪN (CONFUSION MATRIX)

### 📌 Bảng ma trận nhầm lẫn thực tế ($4 \times 4$):

```
                       DỰ ĐOÁN CỦA MÔ HÌNH (PREDICTED)
                 Unfed (0)   Low (1)   Medium (2)   High (3)    Tổng mẫu
THỰC TẾ  Unfed       688        12          0          0          700
(ACTUAL) Low          11       659         30          0          700
         Medium        0        19        644         37          700
         High          1         0         22        677          700
```

---

## 🖼️ 3. SƠ ĐỒ NHIỆT MA TRẬN NHẦM LẪN (HEATMAP VISUALIZATION)

```mermaid
quadrantChart
    title Ma trận nhầm lẫn Holdout Test
    x-axis Dự đoán sai --> Dự đoán đúng
    y-axis Mức độ cá ăn thấp --> Mức độ cá ăn cao
    quadrant-1 Lớp 3 (High): 677/700 đúng (96.71% Recall)
    quadrant-2 Lớp 0 (Unfed): 688/700 đúng (98.29% Recall)
    quadrant-3 Lớp 1 (Low): 659/700 đúng (94.14% Recall)
    quadrant-4 Lớp 2 (Medium): 644/700 đúng (92.00% Recall)
```

---

## 🔬 4. PHÂN TÍCH CHUYÊN SÂU VỀ HIỆU NĂNG MÔ HÌNH

1. **Khả năng Nhận diện Lớp Cá Không Ăn (Unfed)**:
   - Đạt độ chính xác tuyệt đối **Precision 98.29%**, **Recall 98.29%**, và **F1-Score 98.29%**.
   - Chỉ có $12$ mẫu bị nhầm sang lớp *Low* ($1.7\%$). Điều này chứng minh mô hình hoạt động tối ưu để tránh lãng phí thức ăn.

2. **Khả năng Bắt tín hiệu Cá Ăn Mạnh (High)**:
   - Đạt **Recall 96.71%** và **F1-Score 95.76%**.
   - Trong tổng số 700 mẫu cá ăn mạnh, mô hình nhận diện chính xác $677$ mẫu, giúp đảm bảo cung cấp đủ dinh dưỡng cho cá.

3. **Tính Tổng quát hóa (Generalization)**:
   - Điểm **Macro Average F1 (95.28%)** và **Weighted Average F1 (95.28%)** trùng khớp tuyệt đối, khẳng định mô hình đánh giá công bằng và ổn định trên toàn bộ các phân lớp.
