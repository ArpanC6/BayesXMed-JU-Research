# BayesXMed

### An Uncertainty-Aware Explainable Deep Learning Framework for Trustworthy Breast Ultrasound Diagnosis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Research-Medical%20AI-purple.svg)]()

BayesXMed is a unified framework that classifies breast ultrasound images and also tells you **when it is unsure**, **why it is unsure**, and **which cases should be deferred to a radiologist**.

---

## Table of Contents

1. [Overview](#overview)
2. [Research Question](#research-question)
3. [Key Contributions](#key-contributions)
4. [Datasets](#datasets)
5. [Methodology](#methodology)
6. [Results](#results)
7. [Explanation-Uncertainty Map (XUM)](#explanation-uncertainty-map-xum)
8. [Selective Prediction](#selective-prediction)
9. [External Validation](#external-validation)
10. [Limitations](#limitations)
11. [Installation](#installation)
12. [Project Structure](#project-structure)
13. [Acknowledgments](#acknowledgments)
14. [Contact](#contact)

---

## Overview

Breast ultrasound is a widely used, non-invasive tool for detecting breast abnormalities. Deep learning models can classify these images well, but they often:

- produce **overconfident** predictions with no measure of uncertainty,
- lack **interpretability**, and
- degrade sharply under **domain shift** (new hospitals, scanners, populations).

BayesXMed addresses these problems by combining:

| Component | Details |
|-----------|---------|
| Probabilistic models | Deep Ensemble (M = 5 ResNet50) and MC Dropout (T = 20) |
| Uncertainty quantification | Epistemic and predictive uncertainty |
| Explainability | Grad-CAM, with stability measured by the proposed XUM |
| Calibration | ECE, Brier Score, NLL, Temperature Scaling |
| Safe deployment | Selective prediction (deferring uncertain cases) |

---

## Research Question

> Can a unified framework not only classify breast ultrasound images accurately, but also reliably quantify when its predictions are uncertain, explain why, and safely defer cases to radiologists?

---

## Key Contributions

1. **Calibrated framework.** The Deep Ensemble reaches **90.90 ± 2.54 %** accuracy, and temperature scaling reduces ECE to **0.0379**.
2. **Novel Explanation-Uncertainty Map (XUM).** Quantifies the stability of explanations. XUM variance is **5.54× higher inside lesion regions** (Cohen's d = 1.13, p = 1.24 × 10⁻²⁴).
3. **Comprehensive calibration analysis** using ECE, Brier Score, NLL and reliability diagrams.
4. **Selective prediction.** Deferring the 20 % most uncertain cases raises accuracy to **95.16 %** and malignant sensitivity to **90.00 %** (AURC = 0.0263).
5. **Ablation study.** XUM gives the highest corruption-detection AUROC (**0.9921**), ahead of Predictive Entropy (0.9905) and Mutual Information (0.8377).
6. **External validation** on UDIAT, BUS-BRA and BrEaST shows substantial domain shift (66.27 %–80.37 % accuracy), underlining the need for domain adaptation.

---

## Datasets

### Primary dataset: BUSI (Breast Ultrasound Images)

780 ultrasound images collected from 600 female patients.

| Class | Images |
|-------|-------:|
| Benign | 437 |
| Malignant | 210 |
| Normal | 133 |
| **Total** | **780** |

### External validation datasets

| Dataset | Origin | N |
|---------|--------|--:|
| UDIAT | Spain | 163 |
| BUS-BRA | Brazil | 1,875 |
| BrEaST | Poland | 252 |

---

## Methodology

### Experimental protocol

All models were trained with identical settings for a fair comparison.

| Parameter | Value |
|-----------|-------|
| Cross-validation | 5-fold, stratified group |
| Epochs | 20 (early stopping, patience = 5) |
| Batch size | 16 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Scheduler | Cosine annealing |
| Weight decay | 1e-4 |
| Loss | Mixed (CE + 0.3 × Focal) |
| Random seed | 42 (fixed) |

### Models compared

1. **MC Dropout ResNet50**: T = 20 stochastic forward passes.
2. **Deep Ensemble**: 5 independently trained ResNet50 models.

---

## Results

### Five-fold cross-validation (mean ± SD)

| Metric | MC Dropout | Deep Ensemble |
|--------|-----------:|--------------:|
| Accuracy (%) | 89.36 ± 3.13 | **90.90 ± 2.54** |
| ECE | **0.0657 ± 0.0194** | 0.0853 ± 0.0165 |
| Brier Score | 0.1681 ± 0.0328 | **0.1479 ± 0.0209** |
| NLL | 0.3333 ± 0.0624 | **0.2999 ± 0.0437** |
| Malignant Sensitivity (%) | 82.38 ± 6.21 | **84.29 ± 3.19** |
| Specificity (%) | 96.67 ± 1.16 | **97.37 ± 1.84** |

### Calibration (Deep Ensemble)

| Metric | Value |
|--------|------:|
| Baseline ECE | 0.0644 |
| Optimal temperature (T) | 0.85 |
| Calibrated ECE | **0.0379** |

---

## Explanation-Uncertainty Map (XUM)

XUM extends Grad-CAM by generating **T stochastic Grad-CAM maps** across MC Dropout passes and measuring their pixel-wise variance. High variance means the explanation itself is unstable at that location.

### Formulation

Pixel-wise mean Grad-CAM:

```
C̄(i,j) = (1/T) · Σₜ Cₜ(i,j)
```

Pixel-wise variance (XUM score):

```
XUM(i,j) = (1/T) · Σₜ ( Cₜ(i,j) − C̄(i,j) )²
```

### Key findings

- **Noise corruption:** AUROC of **0.9946** at noise level σ = 0.4, with an **8.19×** elevation over clean samples.
- **Localization:** XUM variance is **5.54× higher inside the lesion** than outside (0.00344 vs 0.00062), meaning explanations are least stable in the clinically relevant regions.

---

## Selective Prediction

Predictive entropy is used as the rejection criterion: the model defers its most uncertain cases for expert review.

| Coverage (%) | Accuracy (%) | Malignant Sensitivity (%) |
|-------------:|-------------:|--------------------------:|
| 100 | 91.03 | 85.71 |
| 90 | 94.29 | 91.89 |
| **80** | **95.16** | **90.00** |
| 75 | 95.73 | 92.59 |

Area under the risk-coverage curve (AURC): **0.0263**.

---

## External Validation

Performance drops markedly on unseen datasets, which highlights why uncertainty-aware deployment matters.

| Dataset | N | Accuracy (%) | AUROC |
|---------|--:|-------------:|------:|
| UDIAT (Spain) | 163 | 80.37 | 0.8169 |
| BUS-BRA (Brazil) | 1,875 | 68.27 | 0.6808 |
| BrEaST (Poland) | 252 | 66.27 | 0.7512 |

---

## Limitations

- The primary dataset (BUSI) is small and comes from a single source.
- Accuracy falls by 10–25 points on external datasets; the model is **not** ready for clinical use without domain adaptation.
- This is a research prototype and must not be used for real diagnosis.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.10 or higher
- CUDA-capable GPU (recommended for training)

### Setup

```bash
git clone https://github.com/ArpanC6/BayesXMed-JU-Research.git
cd BayesXMed-JU-Research
pip install -r requirements.txt
```

Download the datasets and place them under `data/` as shown below.

---

## Project Structure

```text
BayesXMed/
│
├── data/                         # Datasets (ignored by git)
│   ├── BUSI/
│   ├── UDIAT/
│   ├── BUS-BRA/
│   └── BrEaST/
│
├── notebooks/                    # Notebooks and Python scripts
│   ├── bayesxmed_pipeline.py     # Main training & evaluation pipeline
│   ├── bayesxmed_advanced_xum.py # XUM experiments & noise sweeps
│   ├── bayesxmed_baselines.py    # Deterministic & MC Dropout baselines
│   └── bayesxmed_external.py     # External validation
│
├── results/                      # Metrics and figures
│   ├── plots/
│   │   ├── reliability_diagram.png
│   │   ├── xum_visualization.png
│   │   └── external_validation.png
│   └── *.json                    # e.g. final_5fold_results.json
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Acknowledgments

The authors are thankful to the DST FIST project, ref. no. SR/FST/ET-I/2022/1059(C), and to CMATER, Department of Computer Science and Engineering, Jadavpur University, Kolkata-700032.

---

## Contact

**Arpan Chakraborty**
Email: chakrabortyarpan151@gmail.com
GitHub: [@ArpanC6](https://github.com/ArpanC6)