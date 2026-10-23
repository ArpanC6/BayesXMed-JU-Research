# BayesXMed: Ablation Study (XUM vs PE vs MI)
# Author: Arpan Chakraborty
# Supervisor: Prof. Debotosh Bhattacharjee

import os
import gc
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings("ignore")

from bayesxmed_pipeline import (
    BUSIDataset, MCDropoutResNet50, GradCAM, BUSI_PATH, CHECKPOINT_DIR, OUTPUT_DIR,
    DEVICE, IMG_SIZE, SEED, set_seed, CAM_PASSES
)

set_seed()

def main():
    print("=" * 70)
    print(" BayesXMed Ablation Study (Corruption Detection)")
    print("=" * 70)

    dataset = BUSIDataset(BUSI_PATH, train=False)
    max_samples = 200
    indices = torch.randperm(len(dataset))[:max_samples]
    eval_subset = Subset(dataset, indices)
    eval_loader = DataLoader(eval_subset, batch_size=1, shuffle=False, num_workers=0)

    mc_model = MCDropoutResNet50(num_classes=3)
    ckpt_path = f"{CHECKPOINT_DIR}/mc/fold0_best.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        mc_model.load_state_dict(ckpt["model_state"])
    
    mc_model.to(DEVICE).eval()
    mc_model.enable_mc_dropout()

    sigma = 0.4 # From paper Table 12
    xum_scores, pe_scores, mi_scores, labels = [], [], [], []

    for idx, (img, label, mask) in enumerate(eval_loader):
        img = img.to(DEVICE)
        noisy_img = img + torch.randn_like(img) * sigma
        
        with torch.no_grad():
            pred = mc_model(img).argmax(1).item()
            
        # 1. XUM Score
        cams = [GradCAM(mc_model, mc_model.backbone.layer4[-1]).compute(noisy_img, pred) for _ in range(CAM_PASSES)]
        xum_scores.append(np.var(np.stack(cams), axis=0).mean())
        
        # 2. Predictive Entropy (PE) & Mutual Information (MI)
        with torch.no_grad():
            mc_probs = [F.softmax(mc_model(noisy_img), dim=1).cpu().numpy()[0] for _ in range(20)]
        mc_probs = np.stack(mc_probs)
        mean_probs = mc_probs.mean(axis=0)
        
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-12))
        expected_entropy = np.mean([-np.sum(p * np.log(p + 1e-12)) for p in mc_probs])
        
        pe_scores.append(float(entropy))
        mi_scores.append(float(entropy - expected_entropy))
        labels.append(1) # 1 means corrupted/noisy

    # Add clean samples for comparison
    for idx, (img, label, mask) in enumerate(eval_loader):
        img = img.to(DEVICE)
        with torch.no_grad():
            pred = mc_model(img).argmax(1).item()
            
        cams = [GradCAM(mc_model, mc_model.backbone.layer4[-1]).compute(img, pred) for _ in range(CAM_PASSES)]
        xum_scores.append(np.var(np.stack(cams), axis=0).mean())
        
        with torch.no_grad():
            mc_probs = [F.softmax(mc_model(img), dim=1).cpu().numpy()[0] for _ in range(20)]
        mc_probs = np.stack(mc_probs)
        mean_probs = mc_probs.mean(axis=0)
        
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-12))
        expected_entropy = np.mean([-np.sum(p * np.log(p + 1e-12)) for p in mc_probs])
        
        pe_scores.append(float(entropy))
        mi_scores.append(float(entropy - expected_entropy))
        labels.append(0) # 0 means clean

    print(f"\nAblation Study Results (sigma = {sigma}):")
    print(f"{'Method':<20} | {'AUROC':<8} | {'AUPRC':<8}")
    print("-" * 45)
    
    for name, scores in [("XUM (Ours)", xum_scores), ("Predictive Entropy", pe_scores), ("Mutual Info", mi_scores)]:
        auroc = roc_auc_score(labels, scores)
        auprc = average_precision_score(labels, scores)
        print(f"{name:<20} | {auroc:.4f}   | {auprc:.4f}")

if __name__ == "__main__":
    main()