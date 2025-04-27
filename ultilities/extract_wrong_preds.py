from ultralytics import YOLO
import cv2
import os

# === CONFIGURATION ===
model = YOLO("../runs/train/s_3k3_50/weights/best.pt")  # Your trained model
img_dir = "../data/valid/images"  # Path to validation images
label_dir = "../data/valid/labels"  # Path to ground-truth YOLO labels
output_dir = "../wrong_predictions"  # Root output folder
os.makedirs(output_dir, exist_ok=True)

# === PROCESSING ===
for img_file in os.listdir(img_dir):
    if not img_file.endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(img_dir, img_file)
    label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")

    if not os.path.exists(label_path):
        continue

    # Predict with your model
    results = model.predict(img_path, conf=0.001, iou=0.5)
    pred_boxes = results[0].boxes
    pred_classes = (
        pred_boxes.cls.cpu().numpy().astype(int) if len(pred_boxes) > 0 else []
    )

    # Read ground truth labels
    with open(label_path, "r") as f:
        gt_classes = [int(line.split()[0]) for line in f.readlines()]

    # Load the image for drawing
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    misclassified = False

    # Check for any class mismatch
    for gt_class in gt_classes:
        if gt_class not in pred_classes:
            misclassified = True

            # Draw predicted boxes (red) with pred_label / true_label
            for box in pred_boxes:
                xyxy = box.xyxy.cpu().numpy().astype(int)[0]
                pred_label = int(box.cls.cpu().numpy())
                cv2.rectangle(img, xyxy[:2], xyxy[2:], (0, 0, 255), 2)
                cv2.putText(
                    img,
                    f"{pred_label}/{gt_class}",
                    (xyxy[0], xyxy[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            # # Draw ground truth boxes (green)
            # with open(label_path, "r") as f:
            #     for line in f.readlines():
            #         parts = line.strip().split()
            #         cls_id = int(parts[0])
            #         x_center, y_center, w_rel, h_rel = map(float, parts[1:5])
            #         x1 = int((x_center - w_rel / 2) * w)
            #         y1 = int((y_center - h_rel / 2) * h)
            #         x2 = int((x_center + w_rel / 2) * w)
            #         y2 = int((y_center + h_rel / 2) * h)
            #         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #         cv2.putText(
            #             img,
            #             f"TRUTH:{cls_id}",
            #             (x1, y1 - 5),
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             0.6,
            #             (0, 255, 0),
            #             2,
            #         )

            # Save image to correct folder
            class_folder = os.path.join(output_dir, f"class_{gt_class}")
            os.makedirs(class_folder, exist_ok=True)
            save_path = os.path.join(class_folder, img_file)
            cv2.imwrite(save_path, img)

            break  # Save once per image, skip checking further ground truths

print("✅ Misclassified images saved into folders by class!")
print(model.names)
