from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

from dog_disease_project.data_loader import (
    get_dataset_stats,
    get_latency_benchmark,
    get_model_comparison,
)


def _as_percent_rows(comparison):
    """Convert 0-1 fractions to 0-100 percentages and attach measured latency."""
    latency = get_latency_benchmark()["latency_ms"]
    rows = []
    for m in comparison:
        row = dict(m)
        for key in ("accuracy", "precision", "recall", "f1_score"):
            row[key] = row[key] * 100
        row["latency_ms"] = latency.get(row["model"])
        rows.append(row)
    return rows


def home_view(request):
    stats = get_dataset_stats()
    comparison = _as_percent_rows(get_model_comparison())
    proposed = next((m for m in comparison if m["model"] == "Proposed Model"), None)

    if not request.user.is_authenticated:
        return render(request, "intro.html", {
            "stats": stats,
            "proposed_accuracy": proposed["accuracy"] if proposed else 0,
            "public_nav": True,
        })

    # The dashboard and dataset pages are staff-only; pet owners land on screening.
    if not request.user.is_staff:
        return redirect("prediction:predict")

    context = {
        "num_classes": stats["num_classes"],
        "class_names": stats["class_names"],
        "total_images": stats["total_images"],
        "proposed": proposed,
        "comparison": comparison,
    }
    return render(request, "home.html", context)


@login_required
def dataset_view(request):
    if not request.user.is_staff:
        return redirect("prediction:predict")

    stats = get_dataset_stats()
    class_names = stats["class_names"]

    per_class = []
    for cls in class_names:
        counts = stats["counts_per_class_split"][cls]
        per_class.append({
            "name": cls,
            "train": counts["train"],
            "valid": counts["valid"],
            "test": counts["test"],
            "total": counts["train"] + counts["valid"] + counts["test"],
        })

    context = {
        "stats": stats,
        "per_class": per_class,
        "image_size": stats["image_sizes_found"][0] if stats["image_sizes_found"] else None,
    }
    return render(request, "dataset.html", context)
