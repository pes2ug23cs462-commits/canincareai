import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from build_nb import build_notebook

cells = []

cells.append(("md", """
# 03 - Results Analysis: Dog Skin Disease Classification

This notebook loads the results produced by `02_model_training.ipynb` and builds the
final results package for the **proposed model**: accuracy/precision/recall/F1,
confusion matrix, training curves, classification report, and the Excel results file.
"""))

cells.append(("code", """
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

RESULTS_DIR = Path("../results")
MODELS_DIR = Path("../models")

with open(MODELS_DIR / "class_names.json") as f:
    class_names = json.load(f)

with open(RESULTS_DIR / "dataset_stats.json") as f:
    dataset_stats = json.load(f)

df_results = pd.read_csv(RESULTS_DIR / "comparison_results.csv")

with open(RESULTS_DIR / "histories.json") as f:
    all_histories = json.load(f)

with open(RESULTS_DIR / "predictions.json") as f:
    all_predictions = json.load(f)

class_names, df_results
"""))

cells.append(("md", "## 1. Model comparison table (all models, actual experiment results)"))

cells.append(("code", """
df_results_display = df_results.copy()
for col in ["accuracy", "precision", "recall", "f1_score"]:
    df_results_display[col] = df_results_display[col].round(4)
df_results_display["train_time_sec"] = df_results_display["train_time_sec"].round(1)
df_results_display
"""))

cells.append(("code", """
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(df_results_display))
width = 0.2
metrics = ["accuracy", "precision", "recall", "f1_score"]
for i, m in enumerate(metrics):
    ax.bar(x + i * width, df_results_display[m], width, label=m)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(df_results_display["model"], rotation=20, ha="right")
ax.set_ylim(0, 1.05)
ax.set_title("Model comparison")
ax.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "plots" / "model_comparison.png", dpi=120)
plt.show()
"""))

cells.append(("md", "## 2. Proposed model - training / validation curves"))

cells.append(("code", """
hist = all_histories["Proposed Model"]
epochs = range(1, len(hist["train_acc"]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(epochs, hist["train_acc"], marker="o", label="Train accuracy")
axes[0].plot(epochs, hist["val_acc"], marker="o", label="Validation accuracy")
axes[0].set_title("Proposed Model - Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(epochs, hist["train_loss"], marker="o", label="Train loss")
axes[1].plot(epochs, hist["val_loss"], marker="o", label="Validation loss")
axes[1].set_title("Proposed Model - Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig(RESULTS_DIR / "plots" / "proposed_model_curves.png", dpi=120)
plt.show()
"""))

cells.append(("md", "## 3. Confusion matrix (proposed model, test set)"))

cells.append(("code", """
y_true, y_pred = all_predictions["Proposed Model"]
y_true, y_pred = np.array(y_true), np.array(y_pred)

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Proposed Model - Confusion Matrix (Test set)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "plots" / "confusion_matrix.png", dpi=120)
plt.show()
"""))

cells.append(("md", "## 4. Classification report (proposed model, test set)"))

cells.append(("code", """
report_dict = classification_report(
    y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
)
df_report = pd.DataFrame(report_dict).T
df_report
"""))

cells.append(("md", "## 5. Final proposed-model metrics"))

cells.append(("code", """
proposed_row = df_results[df_results["model"] == "Proposed Model"].iloc[0]
final_metrics = {
    "Accuracy": round(float(proposed_row["accuracy"]), 4),
    "Precision (macro)": round(float(proposed_row["precision"]), 4),
    "Recall (macro)": round(float(proposed_row["recall"]), 4),
    "F1-score (macro)": round(float(proposed_row["f1_score"]), 4),
    "Training time (sec)": round(float(proposed_row["train_time_sec"]), 1),
    "Total parameters": int(proposed_row["total_params"]),
    "Trainable parameters": int(proposed_row["trainable_params"]),
}
final_metrics
"""))

cells.append(("md", "## 6. Build Excel results file: dog_disease_results.xlsx"))

cells.append(("code", """
excel_path = RESULTS_DIR / "dog_disease_results.xlsx"

# --- Sheet 1: Dataset Summary ---
rows = []
for cls in class_names:
    c = dataset_stats["counts_per_class_split"][cls]
    rows.append({
        "Class": cls,
        "Train": c["train"],
        "Valid": c["valid"],
        "Test": c["test"],
        "Total": c["train"] + c["valid"] + c["test"],
    })
df_dataset_summary = pd.DataFrame(rows)
df_dataset_summary.loc[len(df_dataset_summary)] = [
    "TOTAL",
    df_dataset_summary["Train"].sum(),
    df_dataset_summary["Valid"].sum(),
    df_dataset_summary["Test"].sum(),
    df_dataset_summary["Total"].sum(),
]

# --- Sheet 2: Model Comparison ---
df_model_comparison = df_results_display.rename(columns={
    "model": "Model",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1-score",
    "train_time_sec": "Training Time (sec)",
    "total_params": "Total Parameters",
    "trainable_params": "Trainable Parameters",
})

# --- Sheet 3: Proposed Model Results ---
df_proposed_results = pd.DataFrame(list(final_metrics.items()), columns=["Metric", "Value"])

# --- Sheet 4: Class-wise Results ---
df_classwise = df_report.reset_index().rename(columns={"index": "Class"})
df_classwise = df_classwise[~df_classwise["Class"].isin(["accuracy"])]
df_classwise = df_classwise.round(4)

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    df_dataset_summary.to_excel(writer, sheet_name="Dataset Summary", index=False)
    df_model_comparison.to_excel(writer, sheet_name="Model Comparison", index=False)
    df_proposed_results.to_excel(writer, sheet_name="Proposed Model Results", index=False)
    df_classwise.to_excel(writer, sheet_name="Class-wise Results", index=False)

print("Saved Excel results file to:", excel_path)
"""))

cells.append(("md", """
## Summary

- `results/dog_disease_results.xlsx` contains four sheets: Dataset Summary, Model Comparison,
  Proposed Model Results, and Class-wise Results.
- `models/dog_disease_model.pt` / `models/dog_disease_model.h5` hold the trained proposed
  model's weights, and `models/class_names.json` holds the class label order.
- All numbers in this notebook come directly from the training run in
  `02_model_training.ipynb` - nothing here is fabricated or hand-entered.
"""))

build_notebook(cells, "../notebooks/03_results_analysis.ipynb")
