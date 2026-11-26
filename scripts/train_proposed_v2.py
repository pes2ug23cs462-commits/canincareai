"""Improved training run for the Proposed Model, targeting >92% test accuracy.

Changes vs. the original notebook 02 training loop:
  - Fine-tunes the last TWO MobileNetV2 blocks (not just one) with a lower LR
    than the classification head (differential learning rates).
  - AdamW + weight decay, label smoothing, cosine LR schedule.
  - Class-weighted loss to counter the ~3.4:1 class imbalance (ringworm vs
    Hypersensitivity).
  - Trains for more epochs and checkpoints the BEST epoch by val_acc instead
    of just keeping whatever the last epoch happens to be.
  - Reports test-set accuracy/precision/recall/f1 exactly like notebook 02 so
    results are directly comparable.
"""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent / "Dataset"
MODELS_DIR = SCRIPT_DIR.parent / "models"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
PATIENCE = 8  # early stopping patience on val_acc

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
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

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# Class weights: inverse frequency, normalized to mean 1.0
counts = np.bincount([y for _, y in train_ds.samples], minlength=num_classes).astype(np.float32)
class_weights = counts.sum() / (num_classes * counts)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
print("Class counts:", dict(zip(class_names, counts.tolist())))
print("Class weights:", dict(zip(class_names, class_weights.cpu().tolist())))


class ProposedModelV2(nn.Module):
    """MobileNetV2 backbone, last TWO blocks fine-tuned, GAP + Dropout + Dense head."""

    def __init__(self, num_classes, unfreeze_last_n_blocks=2):
        super().__init__()
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        for p in self.features.parameters():
            p.requires_grad = False
        for block in self.features[-unfreeze_last_n_blocks:]:
            for p in block.parameters():
                p.requires_grad = True

        self.pool = nn.AdaptiveAvgPool2d(1)
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


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


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


def train():
    model = ProposedModelV2(num_classes).to(DEVICE)
    total_params, trainable_params_n = count_params(model)
    print(f"Total params: {total_params:,}  Trainable: {trainable_params_n:,}")

    backbone_params = [p for p in model.features.parameters() if p.requires_grad]
    head_params = list(model.classifier.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 1e-4},
        {"params": head_params, "lr": 1e-3},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_state = None
    epochs_since_best = 0

    start_time = time.time()
    for epoch in range(EPOCHS):
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
        train_loss = running_loss / n
        train_acc = running_correct / n
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

        lr_now = optimizer.param_groups[1]["lr"]
        print(f"[ProposedV2] Epoch {epoch+1}/{EPOCHS} lr={lr_now:.2e} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
              f"{'  <- best' if improved else ''}")

        if epochs_since_best >= PATIENCE:
            print(f"Early stopping: no val_acc improvement for {PATIENCE} epochs.")
            break

    train_time = time.time() - start_time
    model.load_state_dict(best_state)

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    print(f"[ProposedV2] BEST val_acc={best_val_acc:.4f}")
    print(f"[ProposedV2] TEST accuracy={test_acc:.4f} precision={precision:.4f} "
          f"recall={recall:.4f} f1={f1:.4f} time={train_time:.1f}s params={total_params:,}")

    return model, history, {
        "model": "Proposed Model V2",
        "accuracy": test_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "train_time_sec": train_time,
        "total_params": total_params,
        "trainable_params": trainable_params_n,
    }, y_true, y_pred


if __name__ == "__main__":
    model, history, result, y_true, y_pred = train()

    with open(RESULTS_DIR / "history_proposed_v2.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(RESULTS_DIR / "result_proposed_v2.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(RESULTS_DIR / "predictions_proposed_v2.json", "w") as f:
        json.dump({"y_true": y_true.tolist(), "y_pred": y_pred.tolist(), "class_names": class_names}, f, indent=2)

    torch.save(model.state_dict(), MODELS_DIR / "dog_disease_model_v2.pt")
    print("Saved candidate model to", MODELS_DIR / "dog_disease_model_v2.pt")
    print(f"Test accuracy: {result['accuracy']*100:.2f}%")
