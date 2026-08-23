# 🏗️ BÁO CÁO KIẾN TRÚC MÔ HÌNH MULTIMODAL STFT 256K + MOBILENETV2 (SE 1D RECALIBRATION) + FBGF

**Nhánh Git**: `exp/stft256k-raw2049-freqattn-mobilenetv2-se-fbgf`  
**Cơ chế bổ sung cho Video**: **Mechanism 1 - Squeeze-and-Excitation 1D Channel Recalibration (`SERecalibration1D`)**  
**Ngày cập nhật**: 24/08/2026  

---

## 📌 1. TỔNG QUAN LUỒNG DỮ LIỆU & KIẾN TRÚC (MERMAID DIAGRAM)

Mô hình Multimodal kết hợp 2 luồng tín hiệu sinh học cá đớp mồi:
1. **Audio Path**: Sóng âm siêu phân giải $256\text{kHz} \rightarrow$ Bộ lọc Pre-emphasis ($\alpha=0.97$) $\rightarrow$ Raw STFT 2049 Bins $\rightarrow$ Pure Frequency-Domain Attention $\rightarrow$ Depthwise-Separable Audio CNN $\rightarrow$ Vector $256\text{-dim}$.
2. **Video Path with SE-Recalibration**: Khung hình RGB trung tâm $224 \times 224 \rightarrow$ Frozen MobileNetV2 $\rightarrow$ Vector $1280\text{-dim} \rightarrow$ **SE 1D Channel Recalibration (Mechanism 1)** $\rightarrow$ Vector $1280\text{-dim}$ đã được tái hiệu chỉnh trọng số dải kênh.

Hai luồng đặc trưng được hòa trộn bằng **Factorized Bilinear Gated Fusion (FBGF $k=3$)**.

```mermaid
graph TD
    subgraph AudioPipeline ["Kênh Âm thanh (Audio Pipeline - 256kHz)"]
        A1["Raw Audio 256kHz"] --> A2["Pre-Emphasis Filter (alpha=0.97)"]
        A2 --> A3["Raw STFT 2049 Bins (dB scale)"]
        A3 --> A4["Frequency-Domain Attention (Mean+Std Profiling)"]
        A4 --> A5["Early Strided Conv2D (4x2 Stride)"]
        A5 --> A6["Depthwise-Separable Audio CNN Blocks"]
        A6 --> AudFeat["Audio Feature Vector (256d)"]
    end

    subgraph VideoPipeline ["Kênh Thị giác (Video Pipeline + SE-Recalibration)"]
        V1["Video Clip .mp4"] --> V2["Center RGB Frame Decoder (224x224)"]
        V2 --> V3["Frozen MobileNetV2 (eval mode)"]
        V3 --> VidRaw["Raw Visual Vector (1280d)"]
        VidRaw --> SE1D["SE 1D Channel Recalibration Gate (Mechanism 1)"]
        SE1D --> VidFeat["Recalibrated Visual Feature (1280d)"]
    end

    subgraph FusionHead ["Đầu Dung hợp Đa phương thức (FBGF Fusion Head)"]
        AudFeat --> AudProj["Identity / Linear Projection (256d)"]
        VidFeat --> VidProj["Linear Projection (1280d to 256d)"]

        AudProj --> Gate["Dynamic Gated Routing Gate"]
        VidProj --> Gate
        Gate --> VectorG["Gate Vector g in [0, 1]^256"]

        AudProj --> MFB_A["Linear Projection (256d to 768d)"]
        VidProj --> MFB_V["Linear Projection (256d to 768d)"]

        MFB_A --> Hadamard["Hadamard Product (a_mfb x v_mfb)"]
        MFB_V --> Hadamard
        Hadamard --> SumPool["Sum-Pooling (k=3 to 256d)"]
        SumPool --> PowerNorm["Power Norm (sign(x) * sqrt(|x|))"]
        PowerNorm --> LayerNorm["LayerNorm to f_bilinear"]

        LayerNorm --> GatedBlend["Blended Gated Combination"]
        VectorG --> GatedBlend
        VidProj --> GatedBlend

        GatedBlend --> Classifier["Classifier GELU + Dropout 0.1"]
        Classifier --> Logits["Final Logits (4 Classes)"]
    end
```

---

## 🎥 2. CHI TIẾT CƠ CHẾ MECH 1 (SE 1D CHANNEL RECALIBRATION FOR VIDEO)

### Công thức Toán học:
$$\mathbf{v}_{\text{raw}} \in \mathbb{R}^{B \times 1280}$$
$$\mathbf{w}_{\text{channel}} = \sigma\left( W_2 \cdot \text{ReLU}(W_1 \mathbf{v}_{\text{raw}}) \right) \in [0, 1]^{1280}$$
$$\mathbf{v}_{\text{recalibrated}} = \mathbf{v}_{\text{raw}} \odot \mathbf{w}_{\text{channel}}$$

### Tác dụng Kỹ thuật:
* Tự động điều chỉnh tầm quan trọng của từng dải kênh đặc trưng 1280-dim trước khi thu nén về 256-dim.
* Giúp khuếch đại các kênh nhạy cảm với hình ảnh bọt nước nhảy và bóng chuyển động của cá, dập nén các kênh mang nhiễu nền nước tĩnh.
