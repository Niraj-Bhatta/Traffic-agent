
from pathlib import Path
import cv2
from ultralytics import YOLO


# Project paths
ROOT_DIR = Path(__file__).resolve().parents[2]
VIDEO_PATH = ROOT_DIR / "data" / "traffic.mp4"
OUTPUT_PATH = ROOT_DIR / "outputs" / "detected_traffic.mp4"


def detect_video():

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    # Load YOLO model
    model = YOLO("yolo11n.pt")

    # Open video
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError("Could not open the input video.")

    # Read video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid video properties.")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video.")

    frame_count = 0

    print("Starting traffic video detection...")

    try:
        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # Run YOLO on current frame
            results = model.predict(
                source=frame,
                device="cpu",
                conf=0.10,
                iou=0.50,
                imgsz=960,
                max_det=1000,
                verbose=False
            )

            # Draw bounding boxes
            annotated_frame = results[0].plot()

            # Save processed frame
            writer.write(annotated_frame)

            frame_count += 1

            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")

    finally:
        cap.release()
        writer.release()

    print("\nVideo processing completed!")
    print(f"Total frames processed: {frame_count}")
    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    detect_video()
