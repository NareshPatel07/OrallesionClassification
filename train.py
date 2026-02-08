# train.py
# DDPM training, denoising, and EfficientNet classifier training
# Uses internal split function (no external script).

import os
import json
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from model import (
    UNetDenoiser,
    DDPM,
    EfficientNetClassifier,
    get_ddpm_train_transform,
    get_eval_transform_ddpm,
    get_classifier_train_transform,
    get_eval_transform,
)


# ======================================================
# UTILITIES
# ======================================================

def set_seed(s=42):
    random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def split_indices(n, val_ratio=0.2, test_ratio=0.1):
    """
    Return train/val/test index splits for n samples.
    """
    idx = list(range(n))
    random.shuffle(idx)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


class TransformSubset(Dataset):
    """
    Wraps a base ImageFolder and a list of indices,
    applying a given transform to the PIL image.
    """
    def __init__(self, base_ds, indices, transform):
        self.base_ds = base_ds
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img, label = self.base_ds[idx]  # img is PIL, label is int
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ======================================================
# DDPM TRAINING
# ======================================================

def train_ddpm(
    data_dir,
    out_dir="ddpm_out",
    epochs=20,
    batch_size=16,
    lr=2e-4,
    timesteps=200,
    image_size=64,
    num_workers=4,
    base_ch=64,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    transform = get_ddpm_train_transform(image_size)
    ds = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers)

    model = UNetDenoiser(in_ch=3, base_ch=base_ch).to(device)
    ddpm = DDPM(timesteps=timesteps, device=device)

    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    ensure_dir(out_dir)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        total = 0

        pbar = tqdm(loader, desc=f"DDPM Epoch {epoch}/{epochs}")
        for imgs, _ in pbar:
            imgs = imgs.to(device)  # in [-1,1]
            B = imgs.size(0)
            t = torch.randint(0, timesteps, (B,), device=device).long()
            noise = torch.randn_like(imgs)

            x_t = ddpm.q_sample(imgs, t, noise)   # forward noising
            pred_noise = model(x_t, t)            # predict noise

            loss = criterion(pred_noise, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item() * B
            total += B
            pbar.set_postfix({"loss": running / total})

        avg_loss = running / len(ds)
        print(f"Epoch {epoch} avg loss: {avg_loss:.6f}")

        # save per epoch
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
            },
            os.path.join(out_dir, f"ddpm_epoch{epoch}.pth"),
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(),
                       os.path.join(out_dir, "ddpm_final.pth"))
            print("💾 Saved best DDPM UNet -> ddpm_final.pth")

    print("✅ DDPM training complete. Best loss:", best_loss)


# ======================================================
# DDPM DENOISING / GENERATION
# ======================================================

def generate_denoised(
    data_dir,
    ddpm_model,
    out_dir="denoised",
    timesteps=200,
    image_size=64,
    start_t=50,
    gen_steps=50,
    batch_size=4,
    num_workers=4,
    base_ch=64,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    if not os.path.exists(ddpm_model):
        raise FileNotFoundError(f"DDPM model not found: {ddpm_model}")

    transform = get_eval_transform_ddpm(image_size)
    ds = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)

    model = UNetDenoiser(in_ch=3, base_ch=base_ch).to(device)
    state = torch.load(ddpm_model, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.eval()

    ddpm = DDPM(timesteps=timesteps, device=device)

    ensure_dir(out_dir)
    for cls in ds.classes:
        ensure_dir(os.path.join(out_dir, cls))

    from torchvision.transforms.functional import to_pil_image

    idx = 0
    for imgs, labels in tqdm(loader, desc="Denoising"):
        imgs = imgs.to(device)  # normalized [-1,1]
        B = imgs.size(0)

        # corrupt each image at fixed t0 then run partial reverse chain
        t0 = min(start_t, timesteps - 1)
        t_vec = torch.full((B,), t0, device=device, dtype=torch.long)
        noise = torch.randn_like(imgs)
        x_t = ddpm.q_sample(imgs, t_vec, noise)

        x0_hat = ddpm.p_sample_from(model, x_t, start_t=t0, steps=gen_steps)
        # map back from [-1,1] to [0,1]
        out = (x0_hat.clamp(-1, 1) + 1.0) / 2.0

        for b in range(B):
            pil = to_pil_image(out[b].cpu())
            cls_name = ds.classes[labels[b].item()]
            pil.save(os.path.join(out_dir, cls_name,
                                  f"den_{idx:06d}.png"))
            idx += 1

    print("✅ Denoised images saved to:", out_dir)


# ======================================================
# CLASSIFIER TRAINING (WITH INTERNAL SPLIT)
# ======================================================

def train_classifier(
    data_dir,
    out_dir="clf_out",
    epochs=25,
    batch_size=32,
    lr=1e-4,
    weight_decay=1e-5,
    num_workers=4,
    val_ratio=0.2,
    test_ratio=0.1,
):
    """
    Train EfficientNet classifier on images in data_dir (no pre-split).
    Internally splits dataset into train/val/test using random indices.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    base_ds = datasets.ImageFolder(data_dir, transform=None)
    n = len(base_ds)
    train_idx, val_idx, test_idx = split_indices(
        n, val_ratio=val_ratio, test_ratio=test_ratio
    )

    train_tf = get_classifier_train_transform(224)
    eval_tf = get_eval_transform(224)

    train_ds = TransformSubset(base_ds, train_idx, train_tf)
    val_ds = TransformSubset(base_ds, val_idx, eval_tf)
    test_ds = TransformSubset(base_ds, test_idx, eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    class_names = base_ds.classes
    print("✅ Classes:", class_names)

    ensure_dir(out_dir)
    # Save class names for inference (Flask app)
    with open(os.path.join(out_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    model = EfficientNetClassifier(num_classes=len(class_names),
                                   pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3, verbose=True
    )

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # ---------- TRAIN ----------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"CLF Epoch {epoch}/{epochs}")
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                "loss": running_loss / total,
                "acc": correct / total
            })

        train_loss = running_loss / total
        train_acc = correct / total

        # ---------- VALIDATION ----------
        model.eval()
        v_loss = 0.0
        v_corr = 0
        v_tot = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                v_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(1)
                v_corr += (preds == labels).sum().item()
                v_tot += labels.size(0)

        val_loss = v_loss / v_tot
        val_acc = v_corr / v_tot
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(out_dir, "classifier_best.pth"))
            print("💾 Saved best classifier -> classifier_best.pth")

    # ---------- TEST EVAL ----------
    print("🔍 Evaluating best classifier on test set...")
    best_path = os.path.join(out_dir, "classifier_best.pth")
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())

    print("✅ Test Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n",
          classification_report(y_true, y_pred,
                                target_names=class_names))


# ======================================================
# CLI / ENTRYPOINT
# ======================================================

def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True,
                  choices=["ddpm", "gen", "clf"])
    p.add_argument("--data_dir", required=True,
                  help="Root folder with class subfolders")
    p.add_argument("--out_dir", default="output")
    p.add_argument("--ddpm_model", default=None)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--workers", type=int, default=4)

    p.add_argument("--ddpm_image_size", type=int, default=64)
    p.add_argument("--clf_image_size", type=int, default=224)
    p.add_argument("--base_ch", type=int, default=64)
    p.add_argument("--timesteps", type=int, default=200)
    p.add_argument("--start_t", type=int, default=50)
    p.add_argument("--gen_steps", type=int, default=50)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ensure_dir(args.out_dir)

    if args.mode == "ddpm":
        train_ddpm(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            timesteps=args.timesteps,
            image_size=args.ddpm_image_size,
            num_workers=args.workers,
            base_ch=args.base_ch,
        )

    elif args.mode == "gen":
        ddpm_model_path = (
            args.ddpm_model
            if args.ddpm_model is not None
            else os.path.join(args.out_dir, "ddpm_final.pth")
        )
        generate_denoised(
            data_dir=args.data_dir,
            ddpm_model=ddpm_model_path,
            out_dir=args.out_dir,
            timesteps=args.timesteps,
            image_size=args.ddpm_image_size,
            start_t=args.start_t,
            gen_steps=args.gen_steps,
            batch_size=args.batch_size,
            num_workers=args.workers,
            base_ch=args.base_ch,
        )

    elif args.mode == "clf":
        train_classifier(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_workers=args.workers,
        )

    else:
        raise ValueError("mode must be one of: ddpm, gen, clf")
