"""Small helpers for loading the Stage 1 dataset/training artifacts.

All functions read straight from the JSON/CSV files copied from Stage 1 -
nothing here is fabricated or hard-coded.
"""
import csv
import json
from functools import lru_cache

from django.conf import settings


@lru_cache(maxsize=1)
def get_class_names():
    with open(settings.CLASS_NAMES_PATH, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_disease_info():
    with open(settings.DISEASE_INFO_PATH, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_dataset_stats():
    with open(settings.DATASET_STATS_PATH, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_model_comparison():
    rows = []
    with open(settings.MODEL_COMPARISON_CSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "model": row["model"],
                "accuracy": float(row["accuracy"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1_score": float(row["f1_score"]),
                "train_time_sec": float(row["train_time_sec"]),
            })
    return rows


@lru_cache(maxsize=1)
def get_training_history():
    with open(settings.TRAINING_HISTORY_JSON, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_predictions():
    with open(settings.PREDICTIONS_JSON, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_latency_benchmark():
    with open(settings.LATENCY_BENCHMARK_JSON, encoding="utf-8") as f:
        return json.load(f)
