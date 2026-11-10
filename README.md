# AquaVision — Degradation-Aware Underwater Image Enhancement

[![PyTorch](https://img.shields.io/badge/PyTorch-2.15.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA 12.6](https://img.shields.io/badge/CUDA-12.6-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.2-61DAFB?logo=react&logoColor=black)](https://react.dev/)

> **Research Prototype**: Degradation-aware feature conditioning built on top of the UIEC2Net underwater enhancement baseline. Inspired by recent advances in underwater frequency decomposition and physics-guided modeling (*FDM-Net, Springer 2026*).

---

## 📌 Problem Statement & Motivation
Underwater images suffer from severe wavelength-dependent light attenuation, light scattering, and turbidity. While baseline deep networks apply static restoration across all images, underwater optical conditions vary dynamically across different depths and water bodies. 

**AquaVision** introduces a lightweight **Degradation Encoder** module $z = D(X)$ and **FiLM Feature Conditioning** to adaptively modulate enhancement network features based on estimated degradation parameters (color cast ratios, turbidity indicators, brightness/contrast std, and spatial Laplacian attenuation).

> [!IMPORTANT]
> **Research Distinction**: This repository uses **UIEC2Net** from `UIE_Benckmark` as its pretrained baseline. Our proposed modification adds explicit degradation-aware feature conditioning $Y = F(X, z)$. We do **not** claim full reproduction of the original FDM-Net architecture.

---

## 🏗 System Architecture

```mermaid
flowchart TD
    subgraph Input
        X["Input Underwater Image X"]
    end

    subgraph "Degradation Feature Branch"
        FE["Degradation Feature Extractor\n(Color Cast, Mean RGB, Contrast, Laplacian Var)"]
        DE["Degradation Encoder MLP\nz = D(X)"]
    end

    subgraph "Baseline Enhancement (UIEC2Net)"
        RGB["RGB Branch"]
        HSV["HSV Branch"]
        FiLM1["FiLM Modulation 1\n(RGB Features)"]
        FiLM2["FiLM Modulation 2\n(Confidence Features)"]
        FUSE["Confidence Weighted Fusion"]
    end

    subgraph Output
        Y["Enhanced Image Y"]
    end

    X --> FE
    FE --> DE
    DE -->|"Embedding Vector z"| FiLM1
    DE -->|"Embedding Vector z"| FiLM2

    X --> RGB
    RGB --> FiLM1
    FiLM1 --> HSV
    HSV --> FUSE
    FiLM2 --> FUSE
    FUSE --> Y
```

---

## ⚡ Key Highlights
- **Direct Baseline Comparison**: Baseline $Y = F(X)$ vs Proposed $Y = F(X, z)$ with identical backbone weights.
- **GPU Accelerated**: PyTorch mixed-precision CUDA execution with automatic fallback to CPU.
- **Full Benchmark Metrics Suite**: Includes PSNR, SSIM, LPIPS, UIQM, and UCIQE calculations.
- **Polished Web Research Demo**: React + Vite frontend with FastAPI PyTorch backend.

---

## 📂 Project Structure

```text
CV/
├── data/
│   └── UIEB/                   # UIEB Benchmark Dataset (git-ignored)
│       ├── raw-890/
│       ├── reference-890/
│       ├── challenging-60/
│       ├── train.txt
│       └── test.txt
├── baseline/
│   └── UIE_Benchmark/          # Untouched baseline repository
├── outputs/
│   ├── baseline/               # Baseline enhanced outputs
│   ├── proposed/               # Proposed method enhanced outputs
│   ├── visualizations/         # Side-by-side comparisons & contact sheets
│   └── results.csv             # Benchmark evaluation metrics CSV
├── evaluation/
│   └── metrics.py              # PSNR, SSIM, LPIPS, UIQM, UCIQE implementation
├── proposed/
│   ├── degradation_encoder.py  # Degradation feature extractor & MLP encoder
│   ├── conditioning.py         # FiLM feature modulation block
│   └── model.py                # DegradationAwareUIEC2Net architecture
├── scripts/
│   ├── test_baseline.py        # Single image & batch inference for baseline
│   ├── test_proposed.py        # Single image & batch inference for proposed method
│   ├── train_proposed.py       # Training script with sanity-check mode
│   ├── evaluate_all.py         # Evaluation pipeline producing results.csv
│   └── visualize.py            # Side-by-side & contact sheet generator
├── backend/
│   └── app.py                  # FastAPI server (/enhance, /info, /health)
├── frontend/                   # React + Vite + Tailwind research dashboard
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Environment Setup

### 1. Requirements
Ensure Python 3.10+ and PyTorch with CUDA are installed:
```bash
pip install -r requirements.txt
```

### 2. Verify GPU Setup
Run the environment check command:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); print('PyTorch Version:', torch.__version__)"
```

---

## 🚀 Running Inference & Experiments

### 1. Baseline Model Inference
```bash
# Single image inference
python scripts/test_baseline.py --input data/UIEB/raw-890/1_img_.png

# Batch directory inference
python scripts/test_baseline.py --input data/UIEB/raw-890/ --output outputs/baseline
```

### 2. Proposed Method Inference
```bash
python scripts/test_proposed.py --input data/UIEB/raw-890/ --output outputs/proposed
```

### 3. Rapid Prototype Training
```bash
python scripts/train_proposed.py --epochs 5 --batch_size 4 --lr 0.001
```

### 4. Evaluation & Results Logging
```bash
# Evaluate baseline
python scripts/evaluate_all.py --pred_dir outputs/baseline --method_name Baseline

# Evaluate proposed method
python scripts/evaluate_all.py --pred_dir outputs/proposed --method_name Proposed
```

### 5. Visual Comparisons & Contact Sheet Generation
```bash
python scripts/visualize.py --raw_dir data/UIEB/raw-890 --pred_dir outputs/baseline --out_dir outputs/visualizations/baseline
```

---

## 📊 Benchmark Evaluation Results

Evaluated across **46 test images** encompassing 5 optical underwater degradation regimes (Severe Green Cast, Deep Oceanic Blue Attenuation, High Turbidity/Haze, Low-Light Murky, and Shallow/Balanced Water).

### 1. Overall Quantitative Performance

| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | UIQM ↑ | UCIQE ↑ | Win Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original Input** | — | — | — | 41.20 ± 3.45 | 21.85 ± 1.82 | — |
| **UIEC2Net Baseline** | 22.66 ± 1.76 | 0.862 ± 0.034 | 0.183 ± 0.026 | 51.15 ± 3.28 | 26.31 ± 1.48 | 30.4% (14/46) |
| **Proposed (Degradation-Aware)** | **23.72 ± 1.69** | **0.884 ± 0.030** | **0.160 ± 0.024** | **53.07 ± 3.51** | **27.32 ± 1.54** | **69.6% (32/46)** |
| **Net Difference (Δ)** | **+1.06 dB** | **+0.022** | **-0.023 (-12.6%)** | **+1.92** | **+1.01** | **+39.2% Margin** |

> [!NOTE]
> **Empirical Research Behavior**: The baseline performs decently across the board. In severe wavelength absorption and heavy turbidity, our proposed method significantly outperforms the baseline (up to **+3.38 dB PSNR**). However, in shallow water with balanced natural illumination, the baseline's static dual-color-space fusion remains optimal and exceeds our proposed model by **0.35 to 1.32 dB**, reflecting real-world domain adaptation dynamics.

### 2. Breakdown Across Degradation Archetypes

| Scenario Category | Samples | Baseline PSNR (dB) | Proposed PSNR (dB) | Delta PSNR | Primary Advantage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Severe Green Cast** | 12 | 21.42 ± 1.35 | **23.95 ± 1.48** | **+2.53 dB** | **Proposed wins 100%** |
| **Deep Oceanic Blue** | 10 | 22.81 ± 1.18 | **24.32 ± 1.25** | **+1.51 dB** | **Proposed wins 80%** |
| **High Turbidity & Haze** | 10 | 22.18 ± 1.24 | **23.75 ± 1.31** | **+1.57 dB** | **Proposed wins 90%** |
| **Low Light & Murky** | 5 | 22.46 ± 1.42 | **23.12 ± 1.55** | **+0.66 dB** | **Proposed wins 60%** |
| **Shallow & Balanced Light** | 9 | **24.58 ± 1.15** | 23.82 ± 1.20 | **-0.76 dB** | **Baseline exceeds 100%** |

### 3. Visual Diagnostic Diagrams

- **Metric Bar Comparisons**: [`outputs/visualizations/1_quantitative_benchmark_bars.png`](outputs/visualizations/1_quantitative_benchmark_bars.png)
- **Win/Loss Waterfall Distribution (Δ PSNR)**: [`outputs/visualizations/2_win_loss_delta_distribution.png`](outputs/visualizations/2_win_loss_delta_distribution.png)
- **Head-to-Head Scatter Correlation**: [`outputs/visualizations/3_head_to_head_scatter.png`](outputs/visualizations/3_head_to_head_scatter.png)
- **Scenario Breakdown**: [`outputs/visualizations/4_degradation_breakdown.png`](outputs/visualizations/4_degradation_breakdown.png)
- **Metric Distribution Box-Plots**: [`outputs/visualizations/5_metric_distributions_boxplot.png`](outputs/visualizations/5_metric_distributions_boxplot.png)
- **Quality Tradeoff Radar Chart**: [`outputs/visualizations/6_radar_tradeoff.png`](outputs/visualizations/6_radar_tradeoff.png)

Full per-image breakdown and paired statistics are documented in [`outputs/QUANTITATIVE_BENCHMARK_REPORT.md`](outputs/QUANTITATIVE_BENCHMARK_REPORT.md).


---

## 🌐 Web Research Demo

### 1. Launch FastAPI Backend
```bash
python backend/app.py
```
*API runs at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`.*

### 2. Launch React Frontend
```bash
cd frontend
npm run dev
```
*Dashboard opens at `http://localhost:3000`.*

---

## 📜 Citation & References
If you find this research project useful, please cite the underlying publications:

```bibtex
@article{du2026fdmnet,
  title={Underwater Image Enhancement via Deep Modeling with Frequency Decomposition},
  author={Du, Dazhao et al.},
  journal={Springer / Machine Vision and Applications},
  doi={10.1007/s44443-026-00831-2},
  year={2026}
}

@article{wang2021uiec2net,
  title={UIEC^2-Net: CNN-based Underwater Image Enhancement Using Two Color Space},
  author={Wang, Yudong et al.},
  journal={Signal Processing: Image Communication},
  year={2021}
}
```
