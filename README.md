# Automated Phyto-Pathology Diagnostics

## Project Overview

This repository presents a comparative research study on deep learning architectures for the automated identification of 38 plant disease classes using the PlantVillage dataset. The study evaluates the efficacy of four distinct paradigms: Vision Transformers (ViT), Convolutional Recurrent Neural Networks (CRNN), Graph Convolutional Neural Networks (GCNN), and Graph Recurrent Convolutional Neural Networks (GRCNN).

The research focuses on balancing high-fidelity feature extraction with computational efficiency, ultimately identifying the Optimized CRNN architecture as the superior solution for real-time diagnostic applications.

---

## Repository Resources

### Dataset Reference

The models were trained and evaluated on the PlantVillage Dataset, a comprehensive open-access database of healthy and diseased plant leaf images.

* **Dataset Source:** [PlantVillage Dataset (Color)](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset/data)

### Implementation Notebooks

The complete source code and experimental setups for each stage of the project are hosted on Kaggle:

**1. Data Exploration and Analysis**

* [Exploratory Data Analysis (EDA)](https://www.kaggle.com/code/hamzabinbutt/plantvillage-eda)

**2. Model Architectures (Baseline & Graph-based)**

* [Baseline Vision Transformer (ViT)](https://www.kaggle.com/code/hamzabinbutt/plantvillage-baseline-vit)
* [Baseline CRNN](https://www.kaggle.com/code/hamzabinbutt/plantvillage-baseline-crnn)
* [Graph Convolutional Neural Network (GCNN)](https://www.kaggle.com/code/hamzabinbutt/plantvillage-gcnn)
* [Graph Recurrent Convolutional Neural Network (GRCNN)](https://www.kaggle.com/code/hamzabinbutt/plantvillage-grcnn)

**3. Final Optimized Model**

* [Optimized CRNN (State-of-the-Art Results)](https://www.kaggle.com/code/hamzabinbutt/crnn-opt)

---

## Architectural Analysis

### 1. Optimized CRNN (Recommended)

The CRNN utilizes a convolutional backbone for spatial feature extraction, integrated with a Spatial Attention module and a Bidirectional LSTM. By treating spatial features as a sequence, the model captures complex structural dependencies across the leaf surface.

### 2. Graph Recurrent Convolutional Neural Network (GRCNN)

The GRCNN interprets leaf features as nodes within a graph, employing recurrent units to manage dependencies between spatial points. It represents a significant evolution over standard graph models by incorporating memory-gated units.

### 3. Graph Convolutional Neural Network (GCNN)

GCNNs were used to model non-Euclidean spatial relationships. While effective for modeling local geometry, the architecture lacked the sequential depth required for state-of-the-art accuracy in identifying complex disease patterns.

### 4. Vision Transformer (ViT)

The ViT served as the global attention baseline. Despite its capacity for modeling global context, the architecture required significantly more computational time per epoch and yielded lower accuracy compared to hybrid convolutional-recurrent approaches.

---

## Performance Results

### Development Phase: Validation Metrics

| Model Architecture | Avg Val Accuracy | Avg Val F1 (Macro) | Avg Time/Epoch (s) |
| --- | --- | --- | --- |
| **CRNN** | **96%** | **95%** | **191.65** |
| GCNN | 79% | 72% | 206.28 |
| GRCNN | 79% | 73% | 177.23 |
| ViT | 69% | 60% | 554.10 |

### Final Evaluation: Test Metrics (Optimized CRNN)

The following metrics were achieved on an independent hold-out set of 8,147 images.

| Metric | Value | Description |
| --- | --- | --- |
| **Test Accuracy** | **98.59%** | Overall percentage of correct predictions |
| **Macro F1-Score** | **0.9813** | Average F1 across all 38 classes (class-neutral) |
| **Weighted F1-Score** | **0.9900** | F1-score adjusted for class support volume |
| **Total Test Samples** | **8,147** | Unseen images used for final evaluation |

---

## Technical Environment

* **Framework:** PyTorch
* **Hardware:** NVIDIA Tesla T4 / P100 GPU (Kaggle Environment)
* **Optimization:** Label Smoothing (0.1), Adam Optimizer, ReduceLROnPlateau Scheduler
* **Input Resolution:** 256x256 pixels

---
