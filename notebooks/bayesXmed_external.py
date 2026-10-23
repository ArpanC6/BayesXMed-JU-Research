# BayesXMed: External Validation
# Author: Arpan Chakraborty

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings("ignore")

from bayesxmed_pipeline import (
    MCDropoutResNet50, CHECKPOINT_DIR, OUTPUT_DIR,
    DEVICE, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD
)

UDIAT_PATH = "/kaggle/input/datasets/iftekharahmmed/udiat-dataset/UDIAT"
BUS_BRA_PATH = "/kaggle/input/datasets/orvile/bus-bra-a-breast-ultrasound-dataset/BUSBRA/BUSBRA"
BREAST_PATH = "/kaggle/input/datasets/orvile/breast-lesions-usg/breast-lesions-usg/breast-lesions-usg"

class ExternalDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try: img = Image.open(path).convert("RGB")
        except: img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        return self.transform(img), label

def load_udiat(root):
    samples = []
    for cls in ["benign", "malignant"]:
        cls_path = os.path.join(root, cls)
        label = 0 if cls == "benign" else 1
        if os.path.exists(cls_path):
            for fn in os.listdir(cls_path):
                if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                    samples.append((os.path.join(cls_path, fn), label))
    return samples

def load_busbra(root):
    csv_path = os.path.join(root, "bus_data.csv")
    images_dir = os.path.join(root, "Images")
    df = pd.read_csv(csv_path)
    samples = []
    for _, row in df.iterrows():
        img_id = str(row["ID"])
        pathology = str(row["Pathology"]).lower()
        if pathology not in ["benign", "malignant"]: continue
        label = 0 if pathology == "benign" else 1
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = os.path.join(images_dir, img_id + ext)
            if os.path.exists(candidate):
                samples.append((candidate, label))
                break
    return samples

def load_breast(root):
    samples = []
    for cls in ["benign", "malignant"]:
        cls_path = os.path.join(root, "classification", cls)
        label = 0 if cls == "benign" else 1
        if os.path.exists(cls_path):
            for fn in os.listdir(cls_path):
                if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                    samples.append((os.path.join(cls_path, fn), label))
    return samples

@torch.no_grad()
def evaluate(model, loader, name):
    model.eval()
    model.enable_mc_dropout()
    all_probs, all_labels = [], []

    for img, label in loader:
        img = img.to(DEVICE)
        mc_probs = [F.softmax(model(img), dim=1).cpu().numpy() for _ in range(10)]
        all_probs.append(np.stack(mc_probs).mean(axis=0))
        all_labels.extend(label.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.array(all_labels)
    
    # We use binary classification (Benign vs Malignant) for external datasets
    binary_probs = all_probs[:, :2]
    binary_probs = binary_probs / binary_probs.sum(axis=1, keepdims=True)
    preds = binary_probs.argmax(axis=1)

    acc = accuracy_score(all_labels, preds)
    sens = ((preds == 1) & (all_labels == 1)).sum() / max((all_labels == 1).sum(), 1)
    spec = ((preds == 0) & (all_labels == 0)).sum() / max((all_labels == 0).sum(), 1)
    
    try: auroc = roc_auc_score(all_labels, binary_probs[:, 1])
    except: auroc = 0.0

    print(f"\n{name}:")
    print(f"  N = {len(all_labels)}")
    print(f"  Accuracy = {acc * 100:.2f}%")
    print(f"  Sensitivity = {sens * 100:.2f}%")
    print(f"  Specificity = {spec * 100:.2f}%")
    print(f"  AUROC = {auroc:.4f}")

def main():
    print("=" * 70)
    print(" BayesXMed Real External Validation")
    print("=" * 70)

    model = MCDropoutResNet50(num_classes=3)
    ckpt = torch.load(f"{CHECKPOINT_DIR}/mc/fold0_best.pth", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()

    if os.path.exists(UDIAT_PATH):
        samples = load_udiat(UDIAT_PATH)
        if samples:
            loader = DataLoader(ExternalDataset(samples), batch_size=8, shuffle=False)
            evaluate(model, loader, "UDIAT")

    if os.path.exists(BUS_BRA_PATH):
        samples = load_busbra(BUS_BRA_PATH)
        if samples:
            loader = DataLoader(ExternalDataset(samples), batch_size=8, shuffle=False)
            evaluate(model, loader, "BUS-BRA")

    if os.path.exists(BREAST_PATH):
        samples = load_breast(BREAST_PATH)
        if samples:
            loader = DataLoader(ExternalDataset(samples), batch_size=8, shuffle=False)
            evaluate(model, loader, "BrEaST")

if __name__ == "__main__":
    main()