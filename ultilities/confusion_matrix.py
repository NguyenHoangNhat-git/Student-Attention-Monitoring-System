from ultralytics import YOLO
import ultralytics
import ultralytics.utils
import ultralytics.utils.metrics

model = YOLO("../runs/train/s_3k3_50/weights/best.pt")
results = model.predict(source="../data/test/images", save=False, conf=0.001)

ultralytics.utils.metrics.ConfusionMatrix()