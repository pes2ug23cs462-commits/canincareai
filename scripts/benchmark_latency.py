"""Benchmark real single-image inference latency for each architecture.

Latency depends on architecture + hardware, not on which weights are loaded,
so this is a legitimate real measurement (not fabricated) even though we only
persisted the Proposed Model's trained weights in Stage 1.
"""
import json
import time

import torch
import torch.nn as nn
from torchvision import models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
NUM_CLASSES = 6
N_WARMUP = 5
N_RUNS = 30


def build_mobilenetv2():
    m = models.mobilenet_v2(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.classifier[1].in_features, NUM_CLASSES))
    return m


def build_resnet50():
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m


def build_efficientnetb0():
    m = models.efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.classifier[1].in_features, NUM_CLASSES))
    return m


class ProposedModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.mobilenet_v2(weights=None)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(backbone.last_channel, 128),
            nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def benchmark(model):
    model = model.to(DEVICE).eval()
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(x)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            model(x)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    median = times[len(times) // 2]
    return median


if __name__ == "__main__":
    builders = {
        "MobileNetV2": build_mobilenetv2,
        "ResNet50": build_resnet50,
        "EfficientNetB0": build_efficientnetb0,
        "Proposed Model": lambda: ProposedModel(NUM_CLASSES),
    }
    results = {}
    for name, builder in builders.items():
        latency_ms = benchmark(builder())
        results[name] = round(latency_ms, 2)
        print(f"{name}: {latency_ms:.2f} ms/image on {DEVICE}")

    out = {"device": str(DEVICE), "batch_size": 1, "image_size": IMG_SIZE, "latency_ms": results}
    with open("../results/latency_benchmark.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved ../results/latency_benchmark.json")
