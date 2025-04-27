import torch
from torchmetrics import ConfusionMatrix
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import os
import numpy as np
import pandas as pd

# === LOAD MODEL ===
model = YOLO("../runs/train/s_4k12/weights/best.pt")

# === CONFIG ===
img_dir = "../data/valid/images"
label_dir = "../data/valid/labels"
save_folder = "misclassification_reports"
os.makedirs(save_folder, exist_ok=True)

true_labels = []
pred_labels = []

classnames = [
    "Looking away",
    "Looking forward",
    "Phone use",
    "Raising hand",
    "Sleeping",
    "Turning around",
    "Writing-Reading",
]

# === COLLECT VALID PAIRS ===
for img_file in os.listdir(img_dir):
    if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(img_dir, img_file)
    label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")

    if not os.path.exists(label_path):
        continue

    results = model.predict(img_path, conf=0.001, iou=0.5)
    pred_classes = (
        results[0].boxes.cls.cpu().numpy().astype(int)
        if len(results[0].boxes) > 0
        else []
    )

    with open(label_path, "r") as f:
        gt_classes = [int(line.split()[0]) for line in f.readlines()]

    for gt_cls in gt_classes:
        if len(pred_classes) == 0:
            continue  # no prediction at all, skip this instance

        true_labels.append(gt_cls)
        pred_labels.append(
            pred_classes[0]
        )  # assumes 1 pred per image — adjust if needed

# === ENSURE ALL CLASSES ARE COVERED ===
# Add any classes that were never predicted or seen in the ground truth
all_classes = set(range(len(classnames)))  # All possible class indices
observed_classes = set(
    true_labels + pred_labels
)  # Classes that appeared in either gt or preds

missing_classes = all_classes - observed_classes
if missing_classes:
    print(
        f"⚠️ Missing predictions for classes: {', '.join([classnames[i] for i in missing_classes])}"
    )

# === COMPUTE CONFUSION MATRIX ===
if true_labels:
    num_classes = len(classnames)

    cm_metric = ConfusionMatrix(
        task="multiclass", num_classes=num_classes, normalize="true"
    )
    cm_tensor = cm_metric(torch.tensor(pred_labels), torch.tensor(true_labels))
    cm = cm_tensor.numpy()
    cm = np.nan_to_num(cm, nan=0.0)  # Replace NaN with 0

    # === SAVE HEATMAP ===
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classnames,
        yticklabels=classnames,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("Normalized Confusion Matrix")
    png_path = os.path.join(save_folder, "confusion_matrix_clean.png")
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"✅ Confusion matrix image saved at: {png_path}")

    # === SAVE CSV ===
    csv_path = os.path.join(save_folder, "confusion_matrix_clean.csv")
    pd.DataFrame(cm, index=classnames, columns=classnames).to_csv(csv_path)
    print(f"✅ Confusion matrix CSV saved at: {csv_path}")
else:
    print("⚠️ No valid predictions to compute confusion matrix.")
