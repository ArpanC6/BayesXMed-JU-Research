# BayesXMed: Complete Research Pipeline & Ablation
# Author: Arpan Chakraborty
# Supervisor: Prof. Debotosh Bhattacharjee

import os
import gc
import json
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from torchvision import models as tvm
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from scipy import stats as scistats
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# CONFIGURATION
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
PATIENCE = 5
LR = 1e-4
WEIGHT_DECAY = 1e-4
N_FOLDS = 5
MC_SAMPLES = 20
CAM_PASSES = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BUSI_PATH = "/kaggle/input/datasets/sabahesaraki/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT"
CHECKPOINT_DIR = "/kaggle/working/checkpoints"
OUTPUT_DIR = "/kaggle/working/results"
os.makedirs(f"{CHECKPOINT_DIR}/mc", exist_ok=True)
os.makedirs(f"{CHECKPOINT_DIR}/ensemble", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# DATASET
class BUSIDataset(Dataset):
    def __init__(self, root_path, train=True):
        self.samples = []
        self.masks = []
        self.class_to_idx = {"benign": 0, "malignant": 1, "normal": 2}

        for cls_name in ["benign", "malignant", "normal"]:
            cls_path = os.path.join(root_path, cls_name)
            if os.path.exists(cls_path):
                for fn in os.listdir(cls_path):
                    if fn.lower().endswith(('.png', '.jpg', '.jpeg')) and "_mask" not in fn.lower():
                        img_path = os.path.join(cls_path, fn)
                        mask_path = img_path.replace('.png', '_mask.png').replace('.jpg', '_mask.png')
                        if os.path.exists(mask_path):
                            self.samples.append((img_path, cls_name))
                            self.masks.append(mask_path)

        tf_list = [T.Resize((IMG_SIZE, IMG_SIZE))]
        if train:
            tf_list += [T.RandomHorizontalFlip(0.5), T.RandomRotation(10)]
        tf_list += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]

        self.transform = T.Compose(tf_list)
        self.mask_transform = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        mask_path = self.masks[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        try:
            mask = Image.open(mask_path).convert("L")
        except:
            mask = Image.new("L", (IMG_SIZE, IMG_SIZE))
        return self.transform(img), self.class_to_idx[label], self.mask_transform(mask)

# MODELS
class MCDropoutResNet50(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone.fc.in_features, num_classes)
        )

    def forward(self, x):
        b = self.backbone
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3(x)
        x = F.dropout(x, p=0.3, training=True)
        x = b.layer4(x)
        x = F.dropout(x, p=0.5, training=True)
        x = b.avgpool(x)
        x = torch.flatten(x, 1)
        return self.backbone.fc(x)

    def enable_mc_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

# GRAD-CAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self._fwd = target_layer.register_forward_hook(self._save_act)
        self._bwd = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        self.activations = out.detach()

    def _save_grad(self, module, gin, gout):
        self.gradients = gout[0].detach()

    def compute(self, x, target_class):
        self.model.zero_grad()
        out = self.model(x)
        score = out[0, target_class]
        score.backward(retain_graph=False)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        return cam.squeeze().cpu().numpy()

    def close(self):
        self._fwd.remove()
        self._bwd.remove()

# MAIN PIPELINE
def main():
    print("=" * 70)
    print(" BayesXMed Pipeline Execution")
    print("=" * 70)

    dataset = BUSIDataset(BUSI_PATH, train=True)
    labels_arr = np.array([dataset.class_to_idx[s[1]] for s in dataset.samples])
    patient_ids = np.array([s[0].split("/")[-1].split("_")[0] for s in dataset.samples])

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(sgkf.split(np.zeros(len(labels_arr)), labels_arr, groups=patient_ids))
    train_idx, val_idx = folds[0]

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    mc_model = MCDropoutResNet50(num_classes=3)
    ckpt_path = f"{CHECKPOINT_DIR}/mc/fold0_best.pth"
    
    # Load pre-trained weights (assuming you have trained it)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        mc_model.load_state_dict(ckpt["model_state"])
    else:
        print("Warning: Pretrained model not found. Using randomly initialized weights for demonstration.")
    
    mc_model.to(DEVICE).eval()
    mc_model.enable_mc_dropout()

    # 2. Evaluation & XUM Noise Sweep (Table 9 in Paper)
    print("\nRunning XUM Noise Sweep (sigma=0.1 to 0.5)...")
    max_samples = 200
    indices = torch.randperm(len(dataset))[:max_samples]
    eval_subset = Subset(dataset, indices)
    eval_loader = DataLoader(eval_subset, batch_size=1, shuffle=False, num_workers=0)

    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    xum_noise_auroc = []

    for sigma in noise_levels:
        clean_xum, noisy_xum = [], []
        labels = []
        for idx, (img, label, mask) in enumerate(eval_loader):
            img = img.to(DEVICE)
            noisy_img = img + torch.randn_like(img) * sigma
            
            with torch.no_grad():
                pred = mc_model(img).argmax(1).item()
            
            cams = [GradCAM(mc_model, mc_model.backbone.layer4[-1]).compute(img, pred) for _ in range(CAM_PASSES)]
            clean_xum.append(np.var(np.stack(cams), axis=0).mean())
            
            cams_n = [GradCAM(mc_model, mc_model.backbone.layer4[-1]).compute(noisy_img, pred) for _ in range(CAM_PASSES)]
            noisy_xum.append(np.var(np.stack(cams_n), axis=0).mean())
            
            labels.extend([0, 1]) 

        all_xum = clean_xum + noisy_xum
        auroc = roc_auc_score(labels, all_xum)
        xum_noise_auroc.append(auroc)
        print(f"  Sigma {sigma}: XUM AUROC = {auroc:.4f}")

    # 3. XUM Localization (Table 10 in Paper)
    print("\nRunning XUM Inside vs Outside Lesion...")
    inside_vars, outside_vars = [], []
    for idx, (img, label, mask) in enumerate(eval_loader):
        img = img.to(DEVICE)
        mask_np = (mask.squeeze().numpy() > 0.5)
        
        with torch.no_grad():
            pred = mc_model(img).argmax(1).item()
            
        cams = [GradCAM(mc_model, mc_model.backbone.layer4[-1]).compute(img, pred) for _ in range(CAM_PASSES)]
        var_map = np.var(np.stack(cams), axis=0)
        
        inside_vars.append(var_map[mask_np].mean())
        outside_vars.append(var_map[~mask_np].mean())

    stat, p_val = scistats.mannwhitneyu(inside_vars, outside_vars, alternative='greater')
    print(f"  Inside Lesion XUM Variance: {np.mean(inside_vars):.5f}")
    print(f"  Outside Lesion XUM Variance: {np.mean(outside_vars):.5f}")
    print(f"  Mann-Whitney U p-value: {p_val:.2e}")

if __name__ == "__main__":
    main()