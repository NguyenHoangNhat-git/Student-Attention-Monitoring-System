import cv2
from ultralytics import YOLO
from tqdm import tqdm
import torch

# Load your fine-tuned YOLO model
model = YOLO("runs/train/s_4k12/weights/best.pt")

# Input/output video paths
video_path = "demo_vid.mp4"
output_path = "annotated_video.mp4"

# Open input video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError("Cannot open video")

# Video properties
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# VideoWriter setup
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Drawing parameters
line_thickness = 1
font_scale = 0.8
font_thickness = 1
class_names = model.names

print(torch.cuda.is_available())

# Process video frame-by-frame
with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        # Inference
        results = model(frame, verbose=False)[0]

        # Draw predictions
        for box, cls, conf in zip(
            results.boxes.xyxy, results.boxes.cls, results.boxes.conf
        ):
            x1, y1, x2, y2 = map(int, box.tolist())
            label = f"{class_names[int(cls)]} {conf:.2f}"
            color = (0, 255, 0)

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness)

            # Label
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                font_thickness,
                cv2.LINE_AA,
            )

        # Save frame
        out.write(frame)
        pbar.update(1)

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\n✅ Annotated video saved to {output_path}")
