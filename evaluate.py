"""
evaluate.py
------------------------------------------
Evaluation tool for Oral Lesion classification model.

Features:
- Load classifier model automatically
- Load class names
- Evaluate Train / Validation / Test splits
- Compute confusion matrix
- Classification report
- Plot accuracy curves
- Compare activation functions
- Save everything in /evaluation_results/

Author: Naresh Project Final
------------------------------------------
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from model import EfficientNetClassifier, get_eval_transform


# ======================================================
# CONFIG
# ======================================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "ddpmodel")
DATA_DIR = os.path.join(BASE_DIR, "dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "evaluation_results")

MODEL_PATH = os.path.join(MODEL_DIR, "classifier_best.pth")
CLASS_PATH = os.path.join(MODEL_DIR, "class_names.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================
# UTILITIES
# ======================================================

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_classifier():
    """Load model + class names, auto-fix missing files."""
    ensure_dir(MODEL_DIR)

    # ---------- class names ----------
    if not os.path.exists(CLASS_PATH):
        default_classes = ["cancerous", "non_cancerous", "precancerous"]
        with open(CLASS_PATH, "w") as f:
            json.dump(default_classes, f)
        print("⚠ Created default class_names.json")

    with open(CLASS_PATH, "r") as f:
        classes = json.load(f)

    # ---------- model ----------
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Cannot find classifier file at {MODEL_PATH}\n"
            f"👉 Train first using:\n"
            f"     python train.py --mode clf --data_dir dataset --out_dir ddpmodel"
        )

    model = EfficientNetClassifier(num_classes=len(classes), pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("✔ Loaded classifier model")
    print("✔ Class names:", classes)

    return model, classes


# ======================================================
# DATASET & LOADER
# ======================================================

def load_dataset():
    transform = get_eval_transform()

    full = datasets.ImageFolder(DATA_DIR, transform=transform)
    n = len(full)

    val_size = int(n * 0.2)
    test_size = int(n * 0.1)
    train_size = n - val_size - test_size

    train_ds, val_ds, test_ds = random_split(full, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    print("✔ Dataset loaded:")
    print("Train:", len(train_ds))
    print("Val:", len(val_ds))
    print("Test:", len(test_ds))

    return train_loader, val_loader, test_loader, full.classes


# ======================================================
# EVAL FUNCTION
# ======================================================

def evaluate_split(model, loader):
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = model(x)
            pred = out.argmax(1).cpu()

            y_true.extend(y.tolist())
            y_pred.extend(pred.tolist())

    return y_true, y_pred


# ======================================================
# CONFUSION MATRIX PLOT
# ======================================================

def plot_conf_matrix(cm, classes, title, fname):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


# ======================================================
# MAIN EVALUATION
# ======================================================

def evaluate():
    print("🚀 Starting evaluation...\n")

    ensure_dir(RESULTS_DIR)

    model, classes = load_classifier()
    train_loader, val_loader, test_loader, class_list = load_dataset()

    results = {}

    # ---------- Train ----------
    yt, yp = evaluate_split(model, train_loader)
    cm = confusion_matrix(yt, yp)
    acc = accuracy_score(yt, yp)
    results["train_acc"] = acc

    plot_conf_matrix(cm, class_list,
                     "Train Confusion Matrix",
                     os.path.join(RESULTS_DIR, "train_confusion.png"))

    # ---------- Validation ----------
    yt, yp = evaluate_split(model, val_loader)
    cm = confusion_matrix(yt, yp)
    acc = accuracy_score(yt, yp)
    results["val_acc"] = acc

    plot_conf_matrix(cm, class_list,
                     "Validation Confusion Matrix",
                     os.path.join(RESULTS_DIR, "val_confusion.png"))

    # ---------- Test ----------
    yt, yp = evaluate_split(model, test_loader)
    cm = confusion_matrix(yt, yp)
    acc = accuracy_score(yt, yp)
    results["test_acc"] = acc

    report = classification_report(yt, yp, target_names=class_list)
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    plot_conf_matrix(cm, class_list,
                     "Test Confusion Matrix",
                     os.path.join(RESULTS_DIR, "test_confusion.png"))

    # SAVE RESULTS JSON
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("\n✔ Evaluation complete!")
    print("Results saved in /evaluation_results/")
    print(results)


# ======================================================
# RUN
# ======================================================

if __name__ == "__main__":
    evaluate()
