# BayesXMed: Calibration & Temperature Scaling
# Author: Arpan Chakraborty

import os
import gc
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

from bayesxmed_pipeline import (
    BUSIDataset, MCDropoutResNet50, BUSI_PATH, CHECKPOINT_DIR, DEVICE
)

def compute_ece(probs, labels, n_bins=15):
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(np.float32)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(labels)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confs > lo) & (confs <= hi) if i > 0 else (confs >= lo) & (confs <= hi)
        if mask.sum() == 0: continue
        ece += (mask.sum() / N) * abs(correct[mask].mean() - confs[mask].mean())
    return float(ece)

def find_optimal_temperature(logits, labels):
    best_T = 1.0
    best_ece = float("inf")
    for T in np.arange(0.5, 2.0, 0.05):
        scaled = logits / T
        probs = F.softmax(torch.tensor(scaled), dim=1).numpy()
        ece = compute_ece(probs, labels)
        if ece < best_ece:
            best_ece = ece
            best_T = T
    return best_T, best_ece

def main():
    print("=" * 70)
    print(" BayesXMed Calibration (Temperature Scaling)")
    print("=" * 70)

    dataset = BUSIDataset(BUSI_PATH, train=False)
    max_samples = 200
    indices = torch.randperm(len(dataset))[:max_samples]
    eval_subset = Subset(dataset, indices)
    eval_loader = DataLoader(eval_subset, batch_size=16, shuffle=False, num_workers=0)

    mc_model = MCDropoutResNet50(num_classes=3)
    ckpt = torch.load(f"{CHECKPOINT_DIR}/mc/fold0_best.pth", map_location=DEVICE, weights_only=False)
    mc_model.load_state_dict(ckpt["model_state"])
    mc_model.to(DEVICE).eval()

    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in eval_loader:
            x, y, _ = batch
            x = x.to(DEVICE)
            all_logits.append(mc_model(x).cpu().numpy())
            all_labels.extend(y.numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.array(all_labels)

    baseline_probs = F.softmax(torch.tensor(all_logits), dim=1).numpy()
    baseline_ece = compute_ece(baseline_probs, all_labels)
    opt_T, opt_ece = find_optimal_temperature(all_logits, all_labels)

    print(f"\nCalibration Results:")
    print(f"  Baseline ECE: {baseline_ece:.4f}")
    print(f"  Optimal T: {opt_T:.2f}")
    print(f"  Calibrated ECE: {opt_ece:.4f}")

if __name__ == "__main__":
    main()