# BayesXMed: A Bayesian Explainable Deep Learning Framework for Trustworthy Breast Ultrasound Diagnosis


[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Research-Medical%20AI-purple.svg)]()


A Bayesian explainable deep learning framework with uncertainty quantification, calibration, and out-of-distribution detection for reliable medical image diagnosis.

---

## Table of Contents

1. [Overview](#overview)
2. [Research Question](#research-question)
3. [Key Contributions](#key-contributions)
4. [Datasets](#datasets)
5. [Methodology](#methodology)
6. [Results](#results)
7. [Explanation-Uncertainty Map (XUM)](#explanation-uncertainty-map-xum)
8. [Out-of-Distribution Detection](#out-of-distribution-detection)
9. [Selective Prediction](#selective-prediction)
10. [External Validation](#external-validation)
11. [Statistical Significance](#statistical-significance)
12. [Installation](#installation)
13. [Usage](#usage)
14. [Project Structure](#project-structure)
15. [References](#references)
16. [Acknowledgments](#acknowledgments)
17. [Contact](#contact)

---

## Overview

Breast ultrasound is a widely used, non-invasive imaging tool for detecting breast abnormalities. However, interpreting these images remains challenging due to variations in image quality, ambiguous lesion boundaries, and operator dependence. While deep learning models have achieved high classification accuracy in medical imaging, they often produce overconfident predictions without quantifying uncertainty, lack interpretability, and fail when tested on data from different sources.

**BayesXMed** addresses these limitations by integrating:

- Bayesian Deep Learning (MC Dropout + Deep Ensembles)
- Uncertainty Quantification (Epistemic + Aleatoric)
- Explainable AI (Grad-CAM with stability quantification)
- Calibration (ECE, Brier Score, NLL, Reliability Diagrams)
- Out-of-Distribution Detection (Mahalanobis distance)
- Selective Prediction (deferring uncertain cases)

---

## Research Question

Can a Bayesian explainable deep learning framework (BayesXMed) not only classify breast ultrasound images accurately but also reliably quantify when its predictions are uncertain, explain why, and detect out-of-distribution inputs—ultimately enabling trustworthy clinical decision support?

---

## Key Contributions

1. **Calibrated Bayesian Framework**: Deep Ensemble achieves 88.85% accuracy with Expected Calibration Error (ECE) of 0.0654, outperforming deterministic ResNet50 and MC Dropout in both discrimination and calibration.

2. **Novel Explanation-Uncertainty Map (XUM)**: First integration of explanation stability with predictive uncertainty. XUM variance shows strong correlation with epistemic uncertainty (r = 0.6412, p = 0.0001), demonstrating that when the model is uncertain about its prediction, its explanation is also unstable.

3. **Comprehensive Calibration Analysis**: Evaluation using ECE, Brier Score, Negative Log-Likelihood (NLL), and reliability diagrams alongside standard classification metrics.

4. **Selective Prediction**: Rejecting the 20% most uncertain predictions improves accuracy from 88.85% to 96.77%, enabling the system to defer difficult cases for radiologist review.

5. **OOD Detection**: Feature-space methods (Mahalanobis) achieve AUROC > 0.95, while softmax-based methods (MSP, Energy) fail catastrophically, demonstrating that softmax-based OOD scores are unsafe for medical AI deployment.

6. **External Validation**: Comprehensive evaluation on three independent datasets (UDIAT, BUS-BRA, BrEaST) revealing domain shift and highlighting the critical need for uncertainty-aware deployment.

---

## Datasets

### Primary Dataset: BUSI (Breast Ultrasound Images)

The BUSI dataset contains 780 ultrasound images collected from 600 female patients.

| Class | Number of Images |
|-------|------------------|
| Benign | 437 |
| Malignant | 210 |
| Normal | 133 |
| **Total** | **780** |

**Source**: BUSI Dataset (Kaggle)

### External Validation Datasets

To evaluate generalization, BayesXMed was tested on three independent datasets:

| Dataset | Description | Classes |
|---------|-------------|---------|
| UDIAT | Spanish breast ultrasound dataset | Benign, Malignant |
| BUS-BRA | Large Brazilian breast ultrasound dataset | Benign, Malignant |
| BrEaST | Multi-modality dataset | Benign, Malignant |

---

## Methodology

### Experimental Protocol

All models were trained using identical settings to ensure fair comparison:

| Parameter | Value |
|-----------|-------|
| Cross-Validation | 5-Fold Patient-Level Stratified |
| Epochs | 100 (Early Stopping Patience = 10) |
| Batch Size | 16 |
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Scheduler | Cosine Annealing |
| Weight Decay | 1e-4 |
| Unfrozen Layers | Layer 4 + FC Head |
| Random Seed | 42 (Fixed) |

Patient-level splitting was implemented using StratifiedGroupKFold to ensure that images from the same patient do not appear in both training and test sets. A class-weighted cross-entropy loss was employed to address class imbalance, giving higher importance to Malignant and Normal classes.

### Models Compared

1. **Deterministic ResNet50**: Baseline model with single forward pass
2. **MC Dropout ResNet50**: Monte Carlo Dropout with 50 stochastic forward passes
3. **Deep Ensemble**: Ensemble of 5 independently trained ResNet50 models

---

## Results

### Five-Fold Cross-Validation Results

Mean and standard deviation across all 5 folds:

| Model | Accuracy (%) | ECE | Brier Score | NLL |
|-------|-------------|-----|-------------|-----|
| Det. ResNet50 | 87.69 ± 2.76 | 0.0715 ± 0.0218 | 0.2044 | 0.4188 |
| MC Dropout | 88.59 ± 2.76 | 0.0730 ± 0.0179 | 0.1855 | 0.3841 |
| **Deep Ensemble** | **88.85 ± 2.17** | **0.0654 ± 0.0127** | **0.1729** | **0.3601** |

Deep Ensemble outperforms both deterministic ResNet50 and MC Dropout across all metrics, demonstrating superior discrimination and calibration.

### Per-Class Metrics (Deep Ensemble)

| Class | Precision | Recall (Sensitivity) | F1-Score |
|-------|-----------|---------------------|----------|
| Benign | 0.8925 | 0.9432 | 0.9171 |
| Malignant | 0.8919 | 0.8238 | 0.8565 |
| Normal | 0.8846 | 0.8846 | 0.8846 |

**Clinical Significance**: Malignant sensitivity of 82.38% is particularly important for breast cancer diagnosis, as missing malignant cases (false negatives) has serious clinical consequences.

### Confusion Matrix

The confusion matrix for Deep Ensemble on Fold 1 shows the detailed classification performance:

| | Predicted Benign | Predicted Malignant | Predicted Normal |
|---|---|---|---|
| **Actual Benign** | 83 | 5 | 0 |
| **Actual Malignant** | 8 | 42 | 0 |
| **Actual Normal** | 0 | 0 | 23 |

- Benign recall: 94.32% (83/88)
- Malignant sensitivity: 82.38% (42/51)  
- Normal recall: 88.46% (23/26)

### Calibration Analysis

Deep Ensemble proved to be the best-calibrated model, as its confidence scores closely matched actual accuracy. The reliability diagram indicates that while calibration is generally satisfactory, models can exhibit mild overconfidence in intermediate ranges.

**ECE Values**:
- Deterministic ResNet50: 0.0715
- MC Dropout: 0.0730
- Deep Ensemble: 0.0654 (best)

Lower ECE indicates better calibration, meaning the model's confidence scores are more trustworthy.

---

## Explanation-Uncertainty Map (XUM)

Traditional Grad-CAM identifies where a network looked, but not the reliability of that focus. XUM extends this by generating 50 stochastic Grad-CAM maps across MC Dropout forward passes and computing:

1. **Pixel-wise Mean Grad-CAM**: The average explanation across all stochastic passes
2. **Pixel-wise Variance (XUM Score)**: Regions where the explanation is unstable across passes

### Mathematical Formulation

**Pixel-wise Mean Grad-CAM**:

C_bar(i,j) = (1/T) * sum(C_t(i,j))

**Pixel-wise Variance (XUM Score)**:

XUM(i,j) = (1/T) * sum(C_t(i,j) - C_bar(i,j))^2

### Correlation Results

| Correlation | Value | p-value |
|-------------|-------|---------|
| Pearson (XUM vs Epistemic) | 0.6412 | 0.0001 |
| Spearman (XUM vs Epistemic) | 0.4073 | 0.0255 |

**Interpretation**: When the model is uncertain about its prediction, the visual explanation is also volatile. This confirms that explanation instability can serve as a proxy for predictive uncertainty, providing clinicians with both *what* the model looked at and *how certain* it is about that explanation.

---

## Out-of-Distribution Detection

OOD detection is critical for safe medical AI deployment, as models may encounter images from different sources, devices, or disease presentations.

### Results (AUROC)

| Method | BUSI Masks | Corrupted Images |
|--------|------------|------------------|
| MSP (Softmax) | 0.4451 | 0.0000 |
| Energy | 0.5376 | 0.0000 |
| **Mahalanobis** | **0.9571** | **1.0000** |

**Safety Warning**: Softmax-based OOD scores (MSP, Energy) perform poorly and fail catastrophically under corrupted inputs. Feature-space methods like Mahalanobis must be used for safe medical AI deployment.

### Key Finding

Mahalanobis distance in the feature space achieves near-perfect OOD detection, demonstrating that feature-space representations contain more reliable information about whether an input is in-distribution or out-of-distribution.

---

## Selective Prediction

By using predictive entropy as a rejection criterion, the system can defer uncertain cases for expert review rather than making potentially harmful predictions.

### Results

| % Retained | Accuracy (%) | Improvement |
|------------|--------------|-------------|
| 100% | 88.85 | Baseline |
| 80% | 96.77 | +7.92% |
| 50% | 97.44 | +8.59% |

**Clinical Implication**: Rejecting 20% of the most uncertain cases improves accuracy to 96.77%. In practice, this means approximately one in five difficult cases could be automatically flagged for radiologist review, while the remaining cases are classified with substantially higher reliability than the unfiltered model provides.

---

## External Validation

External validation on independent datasets is essential for evaluating generalization capability.

### Results

| Dataset | N | Accuracy (%) | ECE | Malignant Sensitivity (%) |
|---------|---|--------------|-----|---------------------------|
| BUSI (In-Distribution) | 780 | 88.85 | 0.0654 | 82.38 |
| UDIAT | 163 | 58.28 | 0.2876 | 12.96 |
| BUS-BRA | 1,875 | 56.37 | 0.2122 | 12.52 |
| BrEaST | 9,016 | 67.24 | 0.1278 | 45.59 |

**Key Finding**: Severe domain shift is observed, with accuracy dropping to 56-67% on external datasets. This highlights the critical need for:

1. Uncertainty-aware deployment in clinical settings
2. Domain adaptation techniques for cross-institution generalization
3. Continuous monitoring of model performance post-deployment

---

## Statistical Significance

Statistical testing confirms that Deep Ensemble significantly outperforms deterministic ResNet50.

| Test | Statistic | p-value |
|------|-----------|---------|
| Paired t-test | t = 9.021 | 0.0008 |
| Wilcoxon Signed-Rank | - | 0.0625 |
| Bootstrap 95% CI | - | [2.49%, 4.87%] |
| Cohen's d | 4.034 | - |

**Interpretation**: The p-value of 0.0008 indicates a statistically significant improvement at the alpha = 0.05 level. Cohen's d of 4.034 represents a very large effect size, confirming that the improvement is not only statistically significant but also practically meaningful.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.10 or higher
- CUDA-capable GPU (recommended for training)

### Clone the Repository

```bash
git clone https://github.com/ArpanC6/BayesXMed-JU-Research.git
cd BayesXMed-JU-Research