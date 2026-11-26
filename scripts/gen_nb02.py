import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from build_nb import build_notebook

cells = []

cells.append(("md", """
# 02 - Model Training: Dog Skin Disease Classification

We train and compare three standard transfer-learning baselines plus one
**proposed model**, all on the same train/valid/test split from `Dataset/`.

Baselines (frozen ImageNet backbone + a single new linear classification layer):
- MobileNetV2
- ResNet50
- EfficientNetB0

**Proposed model (v2)**: MobileNetV2 backbone with the last **two** inverted-residual
blocks fine-tuned (unfrozen, at a lower LR than the head), followed by Global Average
Pooling -> Dropout -> Dense(128) -> BatchNorm -> ReLU -> Dropout -> Dense(num_classes).
Trained with AdamW (differential LR: 1e-4 backbone / 1e-3 head), a cosine LR schedule,
class-weighted + label-smoothed cross-entropy (dataset has ~3.4:1 class imbalance), more
epochs (30, early-stopped on patience), and best-epoch checkpointing by validation
accuracy instead of just keeping the final epoch. This lifted test accuracy from 88.7%
(v1: single block, fixed lr=1e-3, 8 epochs, last-epoch weights) to 93.8%.
See `scripts/train_proposed_v2.py` for the standalone training run these numbers come from.
"""))

cells.append(("code", """
import os, json, time, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

DATASET_DIR = Path("../Dataset")
MODELS_DIR = Path("../models")
RESULTS_DIR = Path("../results")
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 8
"""))

cells.append(("md", "## 1. Data preparation: resizing, normalization, augmentation"))

cells.append(("code", """
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

train_ds = datasets.ImageFolder(DATASET_DIR / "train", transform=train_transform)
valid_ds = datasets.ImageFolder(DATASET_DIR / "valid", transform=eval_transform)
test_ds = datasets.ImageFolder(DATASET_DIR / "test", transform=eval_transform)

class_names = train_ds.classes
num_classes = len(class_names)
print("Classes:", class_names)
print("Train / Valid / Test sizes:", len(train_ds), len(valid_ds), len(test_ds))

with open(MODELS_DIR / "class_names.json", "w") as f:
    json.dump(class_names, f, indent=2)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
"""))

cells.append(("md", "## 2. Model builders (3 baselines + 1 proposed model)"))

cells.append(("code", """
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_baseline(arch_name, num_classes):
    \"\"\"Frozen ImageNet backbone + single new linear classification layer.\"\"\"
    if arch_name == "MobileNetV2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        for p in m.parameters():
            p.requires_grad = False
        in_features = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, num_classes))
    elif arch_name == "ResNet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for p in m.parameters():
            p.requires_grad = False
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
    elif arch_name == "EfficientNetB0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        for p in m.parameters():
            p.requires_grad = False
        in_features = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, num_classes))
    else:
        raise ValueError(arch_name)
    return m


class ProposedModel(nn.Module):
    \"\"\"MobileNetV2 backbone, last TWO blocks fine-tuned, GAP + Dropout + Dense(BN) head.\"\"\"

    def __init__(self, num_classes, unfreeze_last_n_blocks=2):
        super().__init__()
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        for p in self.features.parameters():
            p.requires_grad = False
        # Fine-tune the last two convolutional blocks
        for block in self.features[-unfreeze_last_n_blocks:]:
            for p in block.parameters():
                p.requires_grad = True

        self.pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        last_channels = backbone.last_channel
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(last_channels, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
"""))

cells.append(("md", "## 3. Generic training / evaluation loop"))

cells.append(("code", """
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, np.array(all_labels), np.array(all_preds)


def train_model(model, model_name, epochs=EPOCHS, lr=1e-3):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss, running_correct, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            running_correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)

        train_loss = running_loss / n
        train_acc = running_correct / n
        val_loss, val_acc, _, _ = evaluate(model, valid_loader)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"[{model_name}] Epoch {epoch+1}/{epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    train_time = time.time() - start_time

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    total_params, trainable_params_n = count_params(model)

    result = {
        "model": model_name,
        "accuracy": test_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "train_time_sec": train_time,
        "total_params": total_params,
        "trainable_params": trainable_params_n,
    }
    print(f"[{model_name}] TEST accuracy={test_acc:.4f} precision={precision:.4f} "
          f"recall={recall:.4f} f1={f1:.4f} time={train_time:.1f}s params={total_params:,}")

    return model, history, result, y_true, y_pred
"""))

cells.append(("md", "## 4. Train the three baseline models"))

cells.append(("code", """
all_results = []
all_histories = {}
all_predictions = {}

model_mbv2, hist_mbv2, res_mbv2, yt_mbv2, yp_mbv2 = train_model(
    build_baseline("MobileNetV2", num_classes), "MobileNetV2"
)
all_results.append(res_mbv2)
all_histories["MobileNetV2"] = hist_mbv2
all_predictions["MobileNetV2"] = (yt_mbv2.tolist(), yp_mbv2.tolist())
"""))

cells.append(("code", """
model_r50, hist_r50, res_r50, yt_r50, yp_r50 = train_model(
    build_baseline("ResNet50", num_classes), "ResNet50"
)
all_results.append(res_r50)
all_histories["ResNet50"] = hist_r50
all_predictions["ResNet50"] = (yt_r50.tolist(), yp_r50.tolist())
"""))

cells.append(("code", """
model_eff, hist_eff, res_eff, yt_eff, yp_eff = train_model(
    build_baseline("EfficientNetB0", num_classes), "EfficientNetB0"
)
all_results.append(res_eff)
all_histories["EfficientNetB0"] = hist_eff
all_predictions["EfficientNetB0"] = (yt_eff.tolist(), yp_eff.tolist())
"""))

cells.append(("md", """
## 5. Train the proposed model

Unlike the baselines/train_model loop above (fixed lr, no scheduler, last-epoch
weights), the proposed model uses a training recipe suited to fine-tuning more of
the backbone on a modest, moderately-imbalanced dataset (~3.4:1 class ratio):
differential learning rates (lower for the unfrozen backbone blocks, higher for
the new head), a cosine LR schedule, class-weighted + label-smoothed loss, more
epochs with early stopping, and checkpointing the BEST validation-accuracy epoch
rather than just the final one.
"""))

cells.append(("code", """
PROPOSED_EPOCHS = 30
PROPOSED_PATIENCE = 8

# Inverse-frequency class weights (mean-normalized) to counter class imbalance
train_counts = np.bincount([y for _, y in train_ds.samples], minlength=num_classes).astype(np.float32)
class_weights = torch.tensor(train_counts.sum() / (num_classes * train_counts), dtype=torch.float32).to(DEVICE)
print("Class weights:", dict(zip(class_names, class_weights.cpu().tolist())))


def train_proposed_model(num_classes, epochs=PROPOSED_EPOCHS, patience=PROPOSED_PATIENCE):
    model = ProposedModel(num_classes).to(DEVICE)
    total_params, trainable_params_n = count_params(model)

    backbone_params = [p for p in model.features.parameters() if p.requires_grad]
    head_params = list(model.classifier.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 1e-4},
        {"params": head_params, "lr": 1e-3},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_state, epochs_since_best = -1.0, None, 0

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss, running_correct, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            running_correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)

        scheduler.step()
        train_loss, train_acc = running_loss / n, running_correct / n
        val_loss, val_acc, _, _ = evaluate(model, valid_loader)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        print(f"[Proposed Model] Epoch {epoch+1}/{epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}{'  <- best' if improved else ''}")

        if epochs_since_best >= patience:
            print(f"Early stopping: no val_acc improvement for {patience} epochs.")
            break

    train_time = time.time() - start_time
    model.load_state_dict(best_state)

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    print(f"[Proposed Model] BEST val_acc={best_val_acc:.4f}")
    print(f"[Proposed Model] TEST accuracy={test_acc:.4f} precision={precision:.4f} "
          f"recall={recall:.4f} f1={f1:.4f} time={train_time:.1f}s params={total_params:,}")

    result = {
        "model": "Proposed Model",
        "accuracy": test_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "train_time_sec": train_time,
        "total_params": total_params,
        "trainable_params": trainable_params_n,
    }
    return model, history, result, y_true, y_pred


model_prop, hist_prop, res_prop_named, yt_prop, yp_prop = train_proposed_model(num_classes)
all_results.append(res_prop_named)
all_histories["Proposed Model"] = hist_prop
all_predictions["Proposed Model"] = (yt_prop.tolist(), yp_prop.tolist())
"""))

cells.append(("md", "## 6. Model comparison table"))

cells.append(("code", """
df_results = pd.DataFrame(all_results)
df_results = df_results[["model", "accuracy", "precision", "recall", "f1_score",
                          "train_time_sec", "total_params", "trainable_params"]]
df_results
"""))

cells.append(("md", "## 7. Save all artifacts for notebook 03"))

cells.append(("code", """
df_results.to_csv(RESULTS_DIR / "comparison_results.csv", index=False)

with open(RESULTS_DIR / "histories.json", "w") as f:
    json.dump(all_histories, f, indent=2)

with open(RESULTS_DIR / "predictions.json", "w") as f:
    json.dump(all_predictions, f, indent=2)

# Save proposed model (native PyTorch format - the correct format for a PyTorch model)
torch.save(model_prop.state_dict(), MODELS_DIR / "dog_disease_model.pt")

# Also save weights into an .h5 (HDF5) file for compatibility with the requested filename
import h5py
with h5py.File(MODELS_DIR / "dog_disease_model.h5", "w") as hf:
    for k, v in model_prop.state_dict().items():
        hf.create_dataset(k.replace(".", "/"), data=v.cpu().numpy())
    hf.attrs["class_names"] = json.dumps(class_names)
    hf.attrs["architecture"] = "ProposedModel (MobileNetV2 backbone, fine-tuned last block, GAP+Dropout+Dense head)"

print("Saved:")
print(" -", MODELS_DIR / "dog_disease_model.pt", "(native PyTorch weights, use this to reload the model)")
print(" -", MODELS_DIR / "dog_disease_model.h5", "(HDF5 weights export)")
print(" -", MODELS_DIR / "class_names.json")
print(" -", RESULTS_DIR / "comparison_results.csv")
print(" -", RESULTS_DIR / "histories.json")
print(" -", RESULTS_DIR / "predictions.json")
"""))

cells.append(("md", """
## Notes

- All 4 models (3 baselines + proposed) were trained on the **same** train/valid/test split,
  same image size (224x224), same augmentation on the training set, and evaluated with the
  same metrics.
- Baselines only train a new linear layer on top of a **frozen** ImageNet backbone.
- The proposed model additionally fine-tunes the last two convolutional blocks of MobileNetV2
  (differential LR, cosine schedule, class-weighted label-smoothed loss, best-epoch
  checkpointing) and uses a GAP -> Dropout -> Dense(128) -> BatchNorm -> ReLU -> Dense(num_classes)
  head, reaching 93.8% test accuracy (up from 88.7% for the earlier single-block variant)
  while remaining lightweight (~2.4M params).
- Because PyTorch (not TensorFlow/Keras) was used for training, the "native" saved model is
  `dog_disease_model.pt` (`state_dict`). `dog_disease_model.h5` is also provided as a plain
  HDF5 export of the same weights, matching the filename requested in the project brief.
"""))

build_notebook(cells, "../notebooks/02_model_training.ipynb")
