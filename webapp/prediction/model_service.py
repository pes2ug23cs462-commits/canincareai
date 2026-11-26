"""Inference service for the Stage 1 "Proposed Model".

Stage 1 was trained with PyTorch (see notebooks/02_model_training.ipynb).
The architecture below is identical to `ProposedModel` from that notebook,
so the saved weights (`dog_disease_proposed_model.pt`) load directly.
"""
import threading

import torch
import torch.nn as nn
from django.conf import settings
from PIL import Image
from torchvision import models, transforms

from dog_disease_project.data_loader import get_class_names

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_lock = threading.Lock()
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class ProposedModel(nn.Module):
    """MobileNetV2 backbone, last two blocks fine-tuned, GAP + Dropout + Dense(BN) head.

    Matches `ProposedModelV2` from scripts/train_proposed_v2.py (93.76% test
    accuracy), which improves on the original single-block fine-tune (88.68%)
    via a second unfrozen block, class-weighted label-smoothed loss, and a
    cosine LR schedule with best-epoch checkpointing.
    """

    def __init__(self, num_classes):
        super().__init__()
        backbone = models.mobilenet_v2(weights=None)
        self.features = backbone.features
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


def _load_model():
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is None:
            class_names = get_class_names()
            model = ProposedModel(len(class_names))
            state_dict = torch.load(settings.PROPOSED_MODEL_PATH, map_location=_device)
            model.load_state_dict(state_dict)
            model.to(_device)
            model.eval()
            _model = model
    return _model


def predict_image(pil_image: Image.Image):
    """Run inference on a PIL image. Returns a list of (class_name, probability)
    sorted by probability, descending."""
    model = _load_model()
    class_names = get_class_names()

    image = pil_image.convert("RGB")
    tensor = _transform(image).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()

    ranked = sorted(zip(class_names, probs), key=lambda t: -t[1])
    return ranked
