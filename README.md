# Multimodal Custom Models: Fish Feeding Intensity Assessment (U-FFIA27K)

Hệ thống nghiên cứu & xây dựng các mô hình AI Đa thức (**Multimodal Audio-Visual**) phục vụ đánh giá cường độ cho cá ăn (Fish Feeding Intensity Assessment) gồm 4 lớp phân loại (`unfed`, `low`, `medium`, `high`) trên bộ dữ liệu **U-FFIA27K**.

---

## 📁 Cấu trúc Thư mục Chính

| Thư mục | Chức năng & Mô tả |
| :--- | :--- |
| 📁 [**`src/`**](file:///C:/Users/manhm/Desktop/Multimodal_Custom_Models/src) | **Mã nguồn Chính của Phương pháp**: Chứa Dataloader, Models, Features, Tasks và Scripts khởi chạy. |
| 📁 [**`U_FFIA27K_audio/`**](file:///C:/Users/manhm/Desktop/Multimodal_Custom_Models/U_FFIA27K_audio) | Huấn luyện & đánh giá Baseline Audio độc lập (PANNs CNN6/10/14, ResNet22, MobileNetV1/V2, EfficientNetB0). |
| 📁 [**`U_FFIA27K_video/`**](file:///C:/Users/manhm/Desktop/Multimodal_Custom_Models/U_FFIA27K_video) | Huấn luyện & đánh giá Baseline Video độc lập (SwinTiny, DINOv2, ResNet18/50, DenseNet121, ConvNeXtTiny). |
| 📁 `papers/` | Chứa 50+ bài báo nghiên cứu khoa học phân loại theo 5 chủ đề chuyên sâu. |

---

## 🌿 Danh sách các Nhánh Thí nghiệm (Branches)

* **`exp/robust-bilinear-pooling-modality-dropout`** *(Nhánh hiện tại - Phương pháp 4)*: Kết hợp **Multi-level Factorized Bilinear Pooling (MFB)** & **Modality Dropout** chống mất/nhiễu kênh tín hiệu. Tự động nén & đẩy toàn bộ artifacts lên Hugging Face Dataset **[`manhmitcf/fish_result`](https://huggingface.co/datasets/manhmitcf/fish_result)**.
* **`exp/geometry-water-ripple-cross-attn`** *(Phương pháp 3)*: Tự động trích xuất đặc trưng **Sóng nước (Water Ripples)** & **Mật độ hình học đàn cá (Delaunay Flocking Geometry)** kết hợp với Cross-Attention Audio-Visual.
* **`exp/swin-tiny-ibot-spatial-pretrain`**: Pretraining tự giám sát iBOT SSL trên khung hình giữa của tập train không nhãn.
* **`exp/stft256k-spatial-video-cross-attn`**: Biến đổi kênh âm thanh 256k STFT high-res tokens + Automatic Mixed Precision (AMP FP16).
* **`exp/dinov2-spatial-visual-tokens-cross-attn`**: Sử dụng DINOv2 ViT visual backbone tự giám sát từ Meta.
* **`exp/spatial-visual-tokens-cross-attn`**: Lưới $196$ spatial visual tokens ($14 \times 14$) cross-attending với $6$ audio tokens.
* **`exp/midframe-to-audio-cross-attn`**: Baseline Cross-Attention ($Q=1$ global video token, $K,V=6$ audio tokens).

---

## 🚀 Hướng dẫn Nhanh Khởi chạy Thử nghiệm

Xem hướng dẫn chi tiết từng bước cho môi trường Marimo Cloud tại:  
👉 [`src/README_MARIMO.md`](file:///C:/Users/manhm/Desktop/Multimodal_Custom_Models/src/README_MARIMO.md)  
👉 [`src/CONFIG_GUIDE.md`](file:///C:/Users/manhm/Desktop/Multimodal_Custom_Models/src/CONFIG_GUIDE.md)

Lệnh chạy chính:
```bash
cd src
python -m pip install -r requirements.txt
python main.py
```
