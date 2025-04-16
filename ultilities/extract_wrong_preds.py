import os
import cv2
from ultralytics import YOLO
import numpy as np


def compute_iou(box1, box2):
    # Boxes in xyxy format
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = interArea / float(box1Area + box2Area - interArea)
    return iou


# Load your trained model
model = YOLO("../runs/train/s_2.6k_relabeled2/weights/last.pt")

# Paths
image_folder = "../relabeled/valid/images"
label_folder = "../relabeled/valid/labels"
output_folder = "../misclassified/s_2.6k_relabeled2"

os.makedirs(output_folder, exist_ok=True)

# Predict
results = model.predict(source=image_folder, save=False, conf=0.01)

for result in results:
    image_name = os.path.basename(result.path)
    label_file = os.path.join(label_folder, os.path.splitext(image_name)[0] + ".txt")

    if not os.path.exists(label_file):
        continue

    # Load ground truth in xyxy format
    gt_boxes = []
    h, w = cv2.imread(result.path).shape[:2]
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])
            x, y, bw, bh = map(float, parts[1:5])
            x1 = (x - bw / 2) * w
            y1 = (y - bh / 2) * h
            x2 = (x + bw / 2) * w
            y2 = (y + bh / 2) * h
            gt_boxes.append((cls, [x1, y1, x2, y2]))

    pred_classes = result.boxes.cls.cpu().numpy()
    pred_coords = result.boxes.xyxy.cpu().numpy()  # xyxy format

    img = cv2.imread(result.path)
    wrong_detected = False

    for pred_class, pred_bbox in zip(pred_classes, pred_coords):
        best_iou = 0
        closest_gt_class = None

        for gt_class, gt_bbox in gt_boxes:
            iou = compute_iou(pred_bbox, gt_bbox)
            if iou > best_iou:
                best_iou = iou
                closest_gt_class = gt_class

        # If closest GT class is different or IoU too small
        if closest_gt_class is None or int(pred_class) != int(closest_gt_class):
            wrong_detected = True
            x1, y1, x2, y2 = map(int, pred_bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            label = f"{int(pred_class)}/{closest_gt_class if closest_gt_class is not None else 'None'}"
            cv2.putText(
                img,
                label,
                (x1, max(y1 - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

    if wrong_detected:
        cv2.imwrite(os.path.join(output_folder, image_name), img)

print("Done! Misclassified images saved to:", output_folder)
print(model.names)
