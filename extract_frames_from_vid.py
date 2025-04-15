import cv2
import os


def extract_frames_fast(video_path, output_folder, interval_sec):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps * interval_sec)

    print(f"Video FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(
        f"Extracting every {interval_sec} second(s) ({frame_interval} frames interval)"
    )

    saved = 0
    for frame_id in range(0, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            break

        frame_name = os.path.join(output_folder, f"frame_{saved:04d}.jpg")
        cv2.imwrite(frame_name, frame)
        print(f"Saved {frame_name}")
        saved += 1

    cap.release()
    print("Extraction completed!")


if __name__ == "__main__":
    video_path = "vid.MOV"
    output_folder = "manual data"
    interval_sec = 2

    extract_frames_fast(video_path, output_folder, interval_sec)
