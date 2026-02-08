import os
import json
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model import EfficientNetClassifier, get_eval_transform

# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "dataset"  # change if needed
MODEL_DIR = "ddpmodel"
MODEL_FILE = os.path.join(MODEL_DIR, "classifier_best.pth")
CLASS_FILE = os.path.join(MODEL_DIR, "class_names.json")

BATCH_SIZE = 16


# ============================================================
# LOAD CLASSIFIER
# ============================================================

def load_model(activation):
    """Loads EfficientNet classifier with new activation function"""

    # Read classes
    if not os.path.exists(CLASS_FILE):
        raise FileNotFoundError("class_names.json missing!")

    with open(CLASS_FILE, "r") as f:
        class_names = json.load(f)

    # Load base classifier
    model = EfficientNetClassifier(num_classes=len(class_names), pretrained=False)

    # Replace activation in classifier head
    model.backbone.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.backbone.classifier[1].in_features, len(class_names)),
        activation
    )

    # Load weights
    state = torch.load(MODEL_FILE, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    model.eval()

    return model, class_names


# ============================================================
# LOAD TEST DATA
# ============================================================

def get_test_loader():
    tf = get_eval_transform()
    ds = datasets.ImageFolder(DATA_DIR, transform=tf)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    return loader, ds.classes


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate(model, loader):
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = model(x)
            pred = out.argmax(1).cpu().tolist()
            y_pred.extend(pred)
            y_true.extend(y.tolist())

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return acc, cm, y_true, y_pred


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment():
    loader, class_names = get_test_loader()

    activations = {
        "ReLU": nn.ReLU(),
        "LeakyReLU": nn.LeakyReLU(0.1),
        "ELU": nn.ELU(),
        "SiLU": nn.SiLU(),
        "GELU": nn.GELU()
    }

    results = {}
    cms = {}

    for name, act in activations.items():
        print(f"\n🔍 Testing activation: {name}")

        model, _ = load_model(act)
        acc, cm, _, _ = evaluate(model, loader)

        results[name] = acc
        cms[name] = cm

        print(f"Accuracy: {acc:.4f}")
        print("Confusion Matrix:")
        print(cm)

    # =====================================================
    # Plot accuracy bar chart
    # =====================================================

    plt.figure(figsize=(10, 5))
    plt.bar(results.keys(), results.values(), color=["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f1c40f"])
    plt.title("Activation Function Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.savefig("activation_accuracy.png")
    plt.show()

    # =====================================================
    # Plot confusion matrices individually
    # =====================================================

    fig, axes = plt.subplots(1, 5, figsize=(25, 4))

    for i, (name, cm) in enumerate(cms.items()):
        axes[i].imshow(cm, cmap="Blues")
        axes[i].set_title(name)
        axes[i].set_xticks(np.arange(len(class_names)))
        axes[i].set_yticks(np.arange(len(class_names)))
        axes[i].set_xticklabels(class_names)
        axes[i].set_yticklabels(class_names)

        for (x, y), value in np.ndenumerate(cm):
            axes[i].text(y, x, value, ha='center', va='center', color="black")

    plt.tight_layout()
    plt.savefig("activation_confusion_matrix.png")
    plt.show()

    print("\n📊 Saved:")
    print(" - activation_accuracy.png")
    print(" - activation_confusion_matrix.png")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_experiment()
