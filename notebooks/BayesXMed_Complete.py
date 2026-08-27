# BayesXMed: Complete Research Pipeline
# Bayesian Explainable Deep Learning Framework for Breast Ultrasound Diagnosis
# 
# Supervisor: Prof. Debotosh Bhattacharjee
# Author: Arpan Chakraborty
# Department: CSE, Jadavpur University


import os
import re
import glob
import json
import random
import warnings
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from torchvision import models as tvm
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, average_precision_score
)
from sklearn.covariance import LedoitWolf
from sklearn.utils.class_weight import compute_class_weight
from scipy import stats as scistats
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


# 1. CONFIGURATION - Professor's Settings


SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 100
PATIENCE = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4
N_FOLDS = 5
MC_SAMPLES = 50
ENSEMBLE_SIZE = 5
NUM_WORKERS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

OUTPUT_DIR = "./bayesxmed_results"
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()

print("Configuration loaded!")

# 2. DATASET LOADER

def extract_patient_id(filepath):
    fn = os.path.basename(filepath)
    match = re.search(r"([a-zA-Z]+)\s*\((.*?)\)", fn)
    if match:
        cls_name = match.group(1).lower()
        patient_num = match.group(2).split("_")[0].split("-")[0]
        return f"{cls_name}_{patient_num}"
    stem = os.path.splitext(fn)[0]
    return stem.split("_")[0].split("-")[0]

def load_busi_samples(root):
    samples, masks = [], []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith(IMG_EXTS):
                continue
            full = os.path.join(dirpath, fn)
            low = (dirpath + fn).lower()
            if "benign" in low:
                label = "benign"
            elif "malignant" in low:
                label = "malignant"
            elif "normal" in low:
                label = "normal"
            else:
                continue
            if "mask" in fn.lower():
                masks.append((full, label, extract_patient_id(full)))
            else:
                samples.append((full, label, extract_patient_id(full)))
    return samples, masks

class BreastUSDataset(Dataset):
    def __init__(self, samples, class_names, train=False):
        self.samples = samples
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        tf_list = [T.Resize((IMG_SIZE, IMG_SIZE))]
        if train:
            tf_list += [T.RandomHorizontalFlip(0.5), T.RandomRotation(10)]
        tf_list += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        self.transform = T.Compose(tf_list)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx][0], self.samples[idx][1]
        try:
            img = Image.open(path).convert("RGB")
        except:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        return self.transform(img), self.class_to_idx.get(label, 0)

def make_loaders(samples, class_names, train_idx, val_idx):
    train_ds = Subset(BreastUSDataset(samples, class_names, train=True), train_idx)
    val_ds = Subset(BreastUSDataset(samples, class_names, train=False), val_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader

print("Dataset loader ready!")

# 3. MODEL DEFINITIONS

class DetResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x, return_penultimate=False):
        b = self.backbone
        x = b.conv1(x); x = b.bn1(x); x = b.relu(x); x = b.maxpool(x)
        x = b.layer1(x); x = b.layer2(x); x = b.layer3(x); x = b.layer4(x)
        x = b.avgpool(x)
        pen = torch.flatten(x, 1)
        logits = b.fc(pen)
        return (logits, pen) if return_penultimate else logits

class MCDropoutResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone.fc.in_features, num_classes)
        )

    def forward(self, x, return_penultimate=False):
        b = self.backbone
        x = b.conv1(x); x = b.bn1(x); x = b.relu(x); x = b.maxpool(x)
        x = b.layer1(x); x = b.layer2(x); x = b.layer3(x)
        x = F.dropout(x, p=0.3, training=True)
        x = b.layer4(x)
        x = F.dropout(x, p=0.5, training=True)
        x = b.avgpool(x)
        pen = torch.flatten(x, 1)
        logits = self.backbone.fc(pen)
        return (logits, pen) if return_penultimate else logits

    def enable_mc_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

class DeepEnsemble(nn.Module):
    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x, return_penultimate=False):
        if return_penultimate:
            ls, ps = [], []
            for m in self.members:
                l, p = m(x, return_penultimate=True)
                ls.append(l); ps.append(p)
            return torch.stack(ls).mean(0), torch.stack(ps).mean(0)
        return torch.stack([m(x) for m in self.members]).mean(0)

print("Models defined!")

# 4. METRICS

def compute_accuracy(probs, labels):
    return float((probs.argmax(1) == labels).mean())

def compute_ece(probs, labels, n_bins=15):
    preds = probs.argmax(1)
    confs = probs.max(1)
    correct = (preds == labels).astype(np.float32)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(labels)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confs > lo) & (confs <= hi) if i > 0 else (confs >= lo) & (confs <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / N) * abs(correct[mask].mean() - confs[mask].mean())
    return float(ece)

def compute_brier(probs, labels):
    onehot = np.eye(probs.shape[1])[labels]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))

def compute_nll(probs, labels, eps=1e-12):
    p = np.clip(probs[np.arange(len(labels)), labels], eps, 1.0)
    return float(-np.mean(np.log(p)))

@torch.no_grad()
def get_predictions(model, loader, mc_samples=1):
    model.eval()
    if mc_samples > 1 and hasattr(model, "enable_mc_dropout"):
        model.enable_mc_dropout()

    all_probs = []
    all_labels = None

    for _ in range(mc_samples):
        run_probs = []
        labels = []
        for x, y in loader:
            x = x.to(DEVICE)
            out = model(x)
            run_probs.append(F.softmax(out, dim=1).cpu().numpy())
            labels.append(y.numpy())
        all_probs.append(np.concatenate(run_probs))
        if all_labels is None:
            all_labels = np.concatenate(labels)

    mean_probs = np.stack(all_probs).mean(0)
    return mean_probs, all_labels

print("Metrics ready!")

# 5. TRAINING WITH CLASS WEIGHTS

def train_one_fold_weighted(model, train_loader, val_loader, fold_idx, tag=""):
    """Training with Class Weighted Loss - Improves Malignant Sensitivity"""
    model.to(DEVICE)

    train_labels = []
    for x, y in train_loader:
        train_labels.extend(y.numpy())
    train_labels = np.array(train_labels)

    weights = compute_class_weight('balanced', classes=np.arange(len(MAIN_CLASSES)), y=train_labels)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    print(f"[INFO] Class weights: {dict(zip(MAIN_CLASSES, weights.round(3)))}")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_acc = 0.0
    patience_ctr = 0
    save_path = f"{CHECKPOINT_DIR}/fold{fold_idx}_weighted_best.pth"

    print(f"\n[TRAIN] {tag} with Class Weights... (EPOCHS={EPOCHS})")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = model(x).argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        acc = (np.array(all_preds) == np.array(all_labels)).mean()

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: Acc={acc*100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            patience_ctr = 0
            torch.save({"model_state": model.state_dict(), "best_acc": best_acc}, save_path)
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    ckpt = torch.load(save_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model, best_acc

print("Training function ready!")

# 6. XUM - Explanation–Uncertainty Map

class StochasticGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.h1 = target_layer.register_forward_hook(self._fwd_hook)
        self.h2 = target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.activations = out

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate_raw(self, x, target_class):
        self.model.zero_grad()
        out = self.model(x)
        score = out[0, target_class]
        score.backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        return cam.squeeze().detach().cpu().numpy()

    def close(self):
        self.h1.remove()
        self.h2.remove()

def compute_xum(model, loader, mc_passes=50, max_samples=30):
    model.eval()
    if hasattr(model, "enable_mc_dropout"):
        model.enable_mc_dropout()

    target_layer = model.backbone.layer4[-1]
    engine = StochasticGradCAM(model, target_layer)

    xum_scores, entropies, epistemics = [], [], []

    for x, y in loader:
        for i in range(min(x.size(0), max_samples - len(xum_scores))):
            xi = x[i:i+1].to(DEVICE)

            with torch.no_grad():
                tmp = [F.softmax(model(xi), 1).cpu().numpy()[0] for _ in range(10)]
                fixed_cls = int(np.argmax(np.mean(tmp, axis=0)))

            cams = []
            for _ in range(mc_passes):
                cams.append(engine.generate_raw(xi, fixed_cls))
            cams = np.stack(cams)

            cams_norm = (cams - cams.min()) / (cams.max() - cams.min() + 1e-8)
            xum_var = float(cams_norm.var(axis=0).mean())

            mean_p = np.mean(tmp, axis=0)
            ent = float(-np.sum(mean_p * np.log(mean_p + 1e-12)))
            alea = float(np.mean([-np.sum(p * np.log(p + 1e-12)) for p in tmp]))
            epi = float(ent - alea)

            xum_scores.append(xum_var)
            entropies.append(ent)
            epistemics.append(epi)

            if len(xum_scores) >= max_samples:
                break
        if len(xum_scores) >= max_samples:
            break

    engine.close()

    xum_scores = np.array(xum_scores)
    epistemics = np.array(epistemics)

    pr = scistats.pearsonr(xum_scores, epistemics)
    sr = scistats.spearmanr(xum_scores, epistemics)

    print(f"\n[XUM RESULTS]")
    print(f"  Pearson r = {pr[0]:.4f}, p = {pr.pvalue:.4g}")
    print(f"  Spearman r = {sr.statistic:.4f}, p = {sr.pvalue:.4g}")

    return {
        "pearson_r": pr[0], "pearson_p": pr.pvalue,
        "spearman_r": sr.statistic, "spearman_p": sr.pvalue,
        "n_samples": len(xum_scores)
    }

print("XUM ready!")

# 7. OOD DETECTION

def msp_score(probs):
    return 1.0 - probs.max(axis=1)

def energy_score(logits, T=1.0):
    lt = torch.tensor(logits, dtype=torch.float32)
    return (-T * torch.logsumexp(lt / T, dim=1)).numpy()

def mahalanobis_scores_ledoit(train_feats, train_labels, test_feats, n_classes):
    lw = LedoitWolf().fit(train_feats)
    inv = np.linalg.inv(lw.covariance_ + 1e-6 * np.eye(lw.covariance_.shape[0]))
    means = []
    for c in range(n_classes):
        fc = train_feats[train_labels == c]
        means.append(fc.mean(0) if len(fc) else train_feats.mean(0))
    return np.array([min(float((f - m) @ inv @ (f - m)) for m in means) for f in test_feats])

def ood_metrics(id_scores, ood_scores):
    y = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    s = np.concatenate([id_scores, ood_scores])
    auroc = roc_auc_score(y, s)
    aupr = average_precision_score(y, s)
    return {"auroc": auroc, "aupr": aupr}

@torch.no_grad()
def collect_features(model, loader):
    feats, labels = [], []
    for x, y in loader:
        _, pen = model(x.to(DEVICE), return_penultimate=True)
        feats.append(pen.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)

@torch.no_grad()
def collect_logits(model, loader):
    logits, labels = [], []
    for x, y in loader:
        out = model(x.to(DEVICE))
        logits.append(out.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)

def run_ood_detection(model, id_loader, train_loader, ood_loader, num_classes):
    model.eval()

    id_logits, _ = collect_logits(model, id_loader)
    id_probs = F.softmax(torch.tensor(id_logits), dim=1).numpy()
    id_feats, _ = collect_features(model, id_loader)

    train_feats, train_labels = collect_features(model, train_loader)

    ood_logits, _ = collect_logits(model, ood_loader)
    ood_probs = F.softmax(torch.tensor(ood_logits), dim=1).numpy()
    ood_feats, _ = collect_features(model, ood_loader)

    results = {
        "MSP": ood_metrics(msp_score(id_probs), msp_score(ood_probs)),
        "Energy": ood_metrics(energy_score(id_logits), energy_score(ood_logits)),
        "Mahalanobis": ood_metrics(
            mahalanobis_scores_ledoit(train_feats, train_labels, id_feats, num_classes),
            mahalanobis_scores_ledoit(train_feats, train_labels, ood_feats, num_classes)
        )
    }

    print(f"\n[OOD RESULTS]")
    for method, res in results.items():
        print(f"  {method}: AUROC={res['auroc']:.4f}, AUPR={res['aupr']:.4f}")

    return results

print("OOD functions ready!")

# 8. PLOTTING FUNCTIONS

def plot_reliability_diagram(probs, labels, title, save_path):
    preds = probs.argmax(1)
    confs = probs.max(1)
    correct = (preds == labels).astype(float)
    bins = np.linspace(0, 1, 11)
    centers, accs = [], []
    for i in range(10):
        lo, hi = bins[i], bins[i+1]
        mask = (confs > lo) & (confs <= hi) if i > 0 else (confs >= lo) & (confs <= hi)
        if mask.sum() == 0:
            continue
        centers.append((lo + hi) / 2)
        accs.append(correct[mask].mean())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.plot(centers, accs, "o-", color="#2ecc71", linewidth=2)
    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Fraction of Positives (Accuracy)")
    ax.set_title(f"Reliability Diagram: {title}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] {save_path}")

def plot_confusion_matrix(cm, class_names, save_path, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix: {title}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] {save_path}")

def plot_ood_bar(results, save_path):
    data = []
    for method, res in results.items():
        data.append({"Method": method, "AUROC": res["auroc"]})
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="Method", y="AUROC", palette="viridis", ax=ax)
    ax.set_ylim(0, 1.1)
    ax.set_title("OOD Detection AUROC")
    ax.set_ylabel("AUROC (Higher is Better)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] {save_path}")

print("Plotting functions ready!")

# 9. MAIN PIPELINE - 5-FOLD CROSS-VALIDATION

print("\n" + "="*70)
print(" LOADING BUSI DATASET")
print("="*70)

#  IMPORTANT: Change this path to your BUSI dataset location

BUSI_PATH = "./Dataset_BUSI_with_GT"  # ← Change this path!

main_samples, busi_masks = load_busi_samples(BUSI_PATH)
MAIN_CLASSES = sorted(set(s[1] for s in main_samples))
NUM_CLASSES = len(MAIN_CLASSES)
c2i = {c: i for i, c in enumerate(MAIN_CLASSES)}

print(f"BUSI: {len(main_samples)} images | Classes: {MAIN_CLASSES}")
for c in MAIN_CLASSES:
    count = sum(1 for s in main_samples if s[1] == c)
    print(f"  {c}: {count}")

# Create patient-level folds
labels_arr = np.array([c2i[s[1]] for s in main_samples])
groups_arr = np.array([s[2] for s in main_samples])
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(sgkf.split(np.zeros(len(labels_arr)), labels_arr, groups=groups_arr))

print("\n" + "="*70)
print(" RUNNING 5-FOLD CROSS-VALIDATION")
print("="*70)

# Result lists
det_accs, det_eces, det_briers, det_nlls = [], [], [], []
mc_accs, mc_eces, mc_briers, mc_nlls = [], [], [], []
ens_accs, ens_eces, ens_briers, ens_nlls = [], [], [], []
malignant_sensitivities = []

for fold_idx in range(N_FOLDS):
    print(f"\n{'='*70}")
    print(f" FOLD {fold_idx+1}/{N_FOLDS}")
    print('='*70)
    
    train_idx, val_idx = folds[fold_idx]
    train_loader, val_loader = make_loaders(main_samples, MAIN_CLASSES, train_idx, val_idx)
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val: {len(val_idx)} samples")
    
   
    # 1. Deterministic ResNet50

    print("\n[1] Training Det. ResNet50...")
    model = DetResNet50(NUM_CLASSES)
    for name, p in model.named_parameters():
        p.requires_grad = ("layer4" in name or "fc" in name)
    model, _ = train_one_fold_weighted(model, train_loader, val_loader, fold_idx, f"Det. ResNet50 Fold {fold_idx+1}")
    
    probs, labels = get_predictions(model, val_loader, mc_samples=1)
    det_accs.append(compute_accuracy(probs, labels))
    det_eces.append(compute_ece(probs, labels))
    det_briers.append(compute_brier(probs, labels))
    det_nlls.append(compute_nll(probs, labels))
    print(f"  Acc: {det_accs[-1]*100:.2f}%, ECE: {det_eces[-1]:.4f}")
    
    # Malignant Sensitivity
    mal_idx = MAIN_CLASSES.index("malignant")
    mal_mask = labels == mal_idx
    mal_recall = (probs[mal_mask].argmax(1) == mal_idx).mean() if mal_mask.sum() > 0 else 0
    malignant_sensitivities.append(mal_recall)
    
    
    # 2. MC Dropout

    print("\n[2] Training MC Dropout...")
    mc_model = MCDropoutResNet50(NUM_CLASSES)
    mc_model, _ = train_one_fold_weighted(mc_model, train_loader, val_loader, fold_idx, f"MC Dropout Fold {fold_idx+1}")
    
    probs_mc, labels_mc = get_predictions(mc_model, val_loader, mc_samples=MC_SAMPLES)
    mc_accs.append(compute_accuracy(probs_mc, labels_mc))
    mc_eces.append(compute_ece(probs_mc, labels_mc))
    mc_briers.append(compute_brier(probs_mc, labels_mc))
    mc_nlls.append(compute_nll(probs_mc, labels_mc))
    print(f"  Acc: {mc_accs[-1]*100:.2f}%, ECE: {mc_eces[-1]:.4f}")
    
    
    # 3. Deep Ensemble
    
    print(f"\n[3] Training Deep Ensemble ({ENSEMBLE_SIZE} Members)...")
    members = []
    for mi in range(ENSEMBLE_SIZE):
        set_seed(SEED + mi * 10 + fold_idx * 100)
        member = DetResNet50(NUM_CLASSES)
        for name, p in member.named_parameters():
            p.requires_grad = ("layer4" in name or "fc" in name)
        member, _ = train_one_fold_weighted(member, train_loader, val_loader, fold_idx, f"Ensemble Member {mi+1} Fold {fold_idx+1}")
        members.append(member)
    
    set_seed(SEED)
    ensemble = DeepEnsemble(members).to(DEVICE)
    
    probs_ens, labels_ens = get_predictions(ensemble, val_loader, mc_samples=1)
    ens_accs.append(compute_accuracy(probs_ens, labels_ens))
    ens_eces.append(compute_ece(probs_ens, labels_ens))
    ens_briers.append(compute_brier(probs_ens, labels_ens))
    ens_nlls.append(compute_nll(probs_ens, labels_ens))
    print(f"  Acc: {ens_accs[-1]*100:.2f}%, ECE: {ens_eces[-1]:.4f}")

# FINAL RESULTS - Mean ± SD
print("\n" + "="*70)
print(" FINAL 5-FOLD CROSS-VALIDATION RESULTS (Mean ± SD)")
print("="*70)

print("\n| Model | Accuracy (%) | ECE | Brier | NLL |")
print("|-------|-------------|-----|-------|-----|")
print(f"| Det. ResNet50 | {np.mean(det_accs)*100:.2f} ± {np.std(det_accs)*100:.2f} | {np.mean(det_eces):.4f} ± {np.std(det_eces):.4f} | {np.mean(det_briers):.4f} ± {np.std(det_briers):.4f} | {np.mean(det_nlls):.4f} ± {np.std(det_nlls):.4f} |")
print(f"| MC Dropout | {np.mean(mc_accs)*100:.2f} ± {np.std(mc_accs)*100:.2f} | {np.mean(mc_eces):.4f} ± {np.std(mc_eces):.4f} | {np.mean(mc_briers):.4f} ± {np.std(mc_briers):.4f} | {np.mean(mc_nlls):.4f} ± {np.std(mc_nlls):.4f} |")
print(f"| Deep Ensemble | {np.mean(ens_accs)*100:.2f} ± {np.std(ens_accs)*100:.2f} | {np.mean(ens_eces):.4f} ± {np.std(ens_eces):.4f} | {np.mean(ens_briers):.4f} ± {np.std(ens_briers):.4f} | {np.mean(ens_nlls):.4f} ± {np.std(ens_nlls):.4f} |")

print(f"\n>>> MALIGNANT SENSITIVITY (Mean ± SD): {np.mean(malignant_sensitivities)*100:.2f} ± {np.std(malignant_sensitivities)*100:.2f}%")

# Save results
final_results = {
    "det_resnet50": {
        "accuracy_mean": np.mean(det_accs), "accuracy_std": np.std(det_accs),
        "ece_mean": np.mean(det_eces), "ece_std": np.std(det_eces),
        "brier_mean": np.mean(det_briers), "brier_std": np.std(det_briers),
        "nll_mean": np.mean(det_nlls), "nll_std": np.std(det_nlls)
    },
    "mc_dropout": {
        "accuracy_mean": np.mean(mc_accs), "accuracy_std": np.std(mc_accs),
        "ece_mean": np.mean(mc_eces), "ece_std": np.std(mc_eces),
        "brier_mean": np.mean(mc_briers), "brier_std": np.std(mc_briers),
        "nll_mean": np.mean(mc_nlls), "nll_std": np.std(mc_nlls)
    },
    "deep_ensemble": {
        "accuracy_mean": np.mean(ens_accs), "accuracy_std": np.std(ens_accs),
        "ece_mean": np.mean(ens_eces), "ece_std": np.std(ens_eces),
        "brier_mean": np.mean(ens_briers), "brier_std": np.std(ens_briers),
        "nll_mean": np.mean(ens_nlls), "nll_std": np.std(ens_nlls)
    },
    "malignant_sensitivity": {
        "mean": np.mean(malignant_sensitivities),
        "std": np.std(malignant_sensitivities)
    }
}

with open(f"{OUTPUT_DIR}/final_5fold_results.json", "w") as f:
    json.dump(final_results, f, indent=2)

print(f"\n[SAVED] Results saved to: {OUTPUT_DIR}/final_5fold_results.json")

# 10. XUM EVALUATION
print("\n" + "="*70)
print(" XUM EVALUATION")
print("="*70)

mc_model = MCDropoutResNet50(NUM_CLASSES)
mc_path = f"{CHECKPOINT_DIR}/fold{N_FOLDS-1}_weighted_best.pth"
if os.path.exists(mc_path):
    ckpt = torch.load(mc_path, map_location=DEVICE, weights_only=False)
    mc_model.load_state_dict(ckpt["model_state"])
    mc_model.to(DEVICE)
    
    _, val_idx = folds[-1]
    val_samples = [main_samples[i] for i in val_idx]
    val_loader = DataLoader(
        BreastUSDataset(val_samples, MAIN_CLASSES, train=False),
        batch_size=8, shuffle=False
    )
    
    xum_result = compute_xumtot