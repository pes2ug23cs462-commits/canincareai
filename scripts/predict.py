"""
Simple command-line inference script for the trained dog skin disease model.

Usage:
    python scripts/predict.py path/to/image.jpg
"""
import sys
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ProposedModel(nn.Module):
    """Same architecture as defined/trained in 02_model_training.ipynb."""

    def __init__(self, num_classes):
        super().__init__()
        backbone = models.mobilenet_v2(weights=None)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        last_channels = backbone.last_channel
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(last_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def load_model():
    with open(MODELS_DIR / "class_names.json") as f:
        class_names = json.load(f)
    model = ProposedModel(len(class_names))
    state_dict = torch.load(MODELS_DIR / "dog_disease_model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model, class_names


def predict(image_path):
    model, class_names = load_model()
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    top_idx = int(probs.argmax())
    print(f"Predicted class: {class_names[top_idx]} ({probs[top_idx]:.2%} confidence)")
    print("\nAll class probabilities:")
    for name, p in sorted(zip(class_names, probs.tolist()), key=lambda t: -t[1]):
        print(f"  {name:20s} {p:.2%}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/predict.py path/to/image.jpg")
        sys.exit(1)
    predict(sys.argv[1])
