# activationcomparison_converted.py
"""
Activation + Classifier Evaluation Suite (converted to Option C: single dataset folder)
Saves outputs to ./activationcomparison/

Usage example:
    python activationcomparison_converted.py --data_dir dataset --model_dir ddpmodels
"""

import os
import argparse
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from skimage.metrics import structural_similarity as ssim_metric
import json

# Import new model definitions / transforms from your model.py
from model import (
    UNetDenoiser,
    DDPM,
    EfficientNetClassifier,
    get_eval_transform,
    get_eval_transform_ddpm,
)

# ------------------------
# Utilities: PSNR / SSIM
# ------------------------
def PSNR(x, y):
    mse = torch.mean((x - y) ** 2).item()
    if mse == 0:
        return 100.0
    # images are in [-1,1], peak-to-peak = 2
    return 20 * np.log10(2.0 / np.sqrt(mse))

def SSIM(x, y):
    # x,y: torch tensors in [-1,1], shape [C,H,W] or [1,C,H,W] handled outside
    x_np = x.squeeze().permute(1,2,0).cpu().numpy()
    y_np = y.squeeze().permute(1,2,0).cpu().numpy()
    # channel_axis=2, data_range=2.0 because [-1,1] range
    return ssim_metric(x_np, y_np, channel_axis=2, data_range=2.0)

# ------------------------
# split indices (same logic as train.py)
# ------------------------
def split_indices(n, val_ratio=0.2, test_ratio=0.1, seed=42):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx

# ------------------------
# TransformSubset (wrap ImageFolder + indices)
# ------------------------
class TransformSubset(Dataset):
    def __init__(self, base_ds, indices, transform):
        self.base_ds = base_ds
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img, label = self.base_ds[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

# ------------------------
# Activation replacement (deepcopy model, replace activations)
# ------------------------
def replace_activation(model, activation_cls):
    """
    Return a deep-copied model with standard activation modules replaced.
    Replaces instances of common activations used in the UNet blocks.
    """
    new_model = copy.deepcopy(model)
    # activation types to replace
    target_types = (nn.ReLU, nn.LeakyReLU, nn.GELU, nn.SiLU)
    # Mish may not exist in some torch versions; handle if available
    if hasattr(nn, "Mish"):
        target_types = target_types + (nn.Mish,)
    # iterate over named_modules and replace attributes on parent
    for name, module in list(new_model.named_modules()):
        if isinstance(module, target_types):
            # find parent by walking name parts
            parent = new_model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            # attempt to construct activation with inplace arg if possible
            try:
                new_act = activation_cls(inplace=True)
            except Exception:
                try:
                    new_act = activation_cls()
                except Exception:
                    continue
            setattr(parent, parts[-1], new_act)
    return new_model

# ------------------------
# plotting helpers
# ------------------------
def plot_confusion_matrix(cm, class_names, title, save_path):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor="black")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ------------------------
# classifier evaluation
# ------------------------
def eval_classifier(model, loader, class_names, device, prefix, out_dir):
    model.eval()
    y_true, y_pred, prob_list = [], [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"Eval {prefix}"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            outs = model(imgs)
            probs = torch.softmax(outs, dim=1)
            pred = probs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            prob_list.extend(probs.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, f"{prefix} Confusion Matrix", f"{out_dir}/{prefix}_confusion_matrix.png")

    # ROC per class (one-vs-rest)
    y_true_onehot = np.eye(len(class_names))[y_true]
    plt.figure(figsize=(7,6))
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], np.array(prob_list)[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.3f})")
    plt.plot([0,1],[0,1],"k--")
    plt.title(f"{prefix} ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{prefix}_roc_curve.png")
    plt.close()

    print(f"\n{prefix} classification report:\n", classification_report(y_true, y_pred, target_names=class_names))

    return cm

# ------------------------
# Evaluate activation function on DDPM denoising metrics
# ------------------------
def evaluate_activation(model, ddpm, loader, device, noise_step, gen_steps):
    psnr_vals, ssim_vals, mse_vals = [], [], []
    model.eval()
    ddpm_eval = ddpm  # wrapper

    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="DDPM eval"):
            imgs = imgs.to(device)
            # sample noise and create x_t at single timestep
            noise = torch.randn_like(imgs)
            t = torch.full((imgs.shape[0],), noise_step, dtype=torch.long, device=device)
            x_t = ddpm_eval.q_sample(imgs, t, noise)
            # run partial reverse chain starting from t for gen_steps
            x0_hat = ddpm_eval.p_sample_from(model, x_t, start_t=noise_step, steps=gen_steps)
            out = x0_hat.clamp(-1,1)
            for i in range(out.shape[0]):
                psnr_vals.append(PSNR(out[i], imgs[i]))
                ssim_vals.append(SSIM(out[i], imgs[i]))
                mse_vals.append(torch.mean((out[i] - imgs[i])**2).item())

    return float(np.mean(psnr_vals)), float(np.mean(ssim_vals)), float(np.mean(mse_vals))

# ------------------------
# main
# ------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = "activationcomparison"
    os.makedirs(out_dir, exist_ok=True)

    # --- load dataset (single folder with class subfolders) ---
    base_ds = datasets.ImageFolder(args.data_dir, transform=None)
    n = len(base_ds)
    if n == 0:
        raise RuntimeError(f"No images found in {args.data_dir} (expect class subfolders).")
    print(f"Found {n} images across {len(base_ds.classes)} classes.")
    class_names = base_ds.classes

    # split indices (same default ratios used in train.py)
    train_idx, val_idx, test_idx = split_indices(n, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed)
    # build datasets with evaluation transform for classifier and ddpm transform for ddpm eval
    clf_tf = get_eval_transform(image_size=args.clf_image_size)
    ddpm_tf = get_eval_transform_ddpm(image_size=args.ddpm_image_size)

    train_ds = TransformSubset(base_ds, train_idx, clf_tf)
    val_ds = TransformSubset(base_ds, val_idx, clf_tf)
    test_ds = TransformSubset(base_ds, test_idx, clf_tf)

    # loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # --- load classifier ---
    clf_path = os.path.join(args.model_dir, args.classifier_file)
    if not os.path.exists(clf_path):
        raise FileNotFoundError(f"Classifier file not found: {clf_path}")
    classifier = EfficientNetClassifier(num_classes=args.num_classes, pretrained=False).to(device)
    classifier.load_state_dict(torch.load(clf_path, map_location=device))
    classifier.eval()

    print("\nEvaluating classifier on train / val / test ...")
    eval_classifier(classifier, train_loader, class_names, device, "train", out_dir)
    eval_classifier(classifier, val_loader,   class_names, device, "val",   out_dir)
    eval_classifier(classifier, test_loader,  class_names, device, "test",  out_dir)

    # --- load ddpm unet ---
    ddpm_path = os.path.join(args.model_dir, args.ddpm_file)
    if not os.path.exists(ddpm_path):
        raise FileNotFoundError(f"DDPM file not found: {ddpm_path}")
    unet = UNetDenoiser(in_ch=3, base_ch=args.base_ch).to(device)
    state = torch.load(ddpm_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        unet.load_state_dict(state["model_state"])
    else:
        unet.load_state_dict(state)
    unet.eval()

    ddpm = DDPM(timesteps=args.timesteps, device=device)

    # prepare ddpm loader (use test split but transform suited for ddpm)
    test_ds_ddpm = TransformSubset(base_ds, test_idx, ddpm_tf)
    ddpm_loader = DataLoader(test_ds_ddpm, batch_size=args.eval_batch, shuffle=True, num_workers=args.workers)

    # activations to compare
    activations = {
        "ReLU": nn.ReLU,
        "LeakyReLU": nn.LeakyReLU,
        "GELU": nn.GELU,
        "SiLU": nn.SiLU,
    }
    if hasattr(nn, "Mish"):
        activations["Mish"] = nn.Mish

    results = {"act": [], "psnr": [], "ssim": [], "mse": []}
    print("\nComparing activations (DDPM denoising metrics)...")
    for name, act in activations.items():
        print("Testing:", name)
        model_with_act = replace_activation(unet, act).to(device)
        p, s, m = evaluate_activation(model_with_act, ddpm, ddpm_loader, device, args.noise_step, args.gen_steps)
        results["act"].append(name); results["psnr"].append(p); results["ssim"].append(s); results["mse"].append(m)

    # save bar charts
    plt.figure(figsize=(8,5)); plt.bar(results["act"], results["psnr"]); plt.title("PSNR Comparison"); plt.savefig(f"{out_dir}/psnr_comparison.png"); plt.close()
    plt.figure(figsize=(8,5)); plt.bar(results["act"], results["ssim"]); plt.title("SSIM Comparison"); plt.savefig(f"{out_dir}/ssim_comparison.png"); plt.close()
    plt.figure(figsize=(8,5)); plt.bar(results["act"], results["mse"]); plt.title("MSE Comparison"); plt.savefig(f"{out_dir}/mse_comparison.png"); plt.close()

    # radar
    angles = np.linspace(0, 2*np.pi, len(results["act"]), endpoint=False).tolist()
    angles += angles[:1]
    psnr_r = results["psnr"] + results["psnr"][:1]; ssim_r = results["ssim"] + results["ssim"][:1]
    plt.figure(figsize=(7,7)); ax = plt.subplot(111, polar=True)
    ax.plot(angles, psnr_r, "o-", label="PSNR"); ax.fill(angles, psnr_r, alpha=0.25)
    ax.plot(angles, ssim_r, "o-", label="SSIM"); ax.fill(angles, ssim_r, alpha=0.25)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(results["act"]); plt.legend(); plt.title("Activation Radar"); plt.savefig(f"{out_dir}/radar_comparison.png"); plt.close()

    # save results JSON
    with open(os.path.join(out_dir, "activation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nAll saved in:", out_dir)
    print("Done.")

# ------------------------
# CLI args
# ------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="dataset")
    p.add_argument("--model_dir", type=str, default="ddpmodels")
    p.add_argument("--classifier_file", type=str, default="classifier_best.pth")
    p.add_argument("--ddpm_file", type=str, default="ddpm_final.pth")
    p.add_argument("--num_classes", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch", type=int, default=1)
    p.add_argument("--timesteps", type=int, default=200)
    p.add_argument("--noise_step", type=int, default=50)
    p.add_argument("--gen_steps", type=int, default=50)
    p.add_argument("--base_ch", type=int, default=64)
    p.add_argument("--ddpm_image_size", type=int, default=64)
    p.add_argument("--clf_image_size", type=int, default=224)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)
