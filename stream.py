from ultralytics import models
import cv2

model = models.YOLO("runs/train/s_3k3_50/weights/best.pt")
for result in model.predict(source=0, stream=True):
    frame = result.plot()  # this gives you an annotated frame
    cv2.imshow("YOLOv11 Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

results = model.predict(source=0, stream=True, show=True)
