
from pathlib import Path
from ultralytics import YOLO
from src.traffic_analytics import TrafficAnalytics
from src.vehicle_counter import VehicleCounter
import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]

VIDEO_PATH = ROOT_DIR / "data" / "traffic.mp4"
OUTPUT_PATH = ROOT_DIR / "outputs" / "tracked_traffic.mp4"


def track_video():

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    model = YOLO("yolo11n.pt")

    analytics = TrafficAnalytics()
    counter = VehicleCounter(
    line_start=(200, 400),
    line_end=(1000, 400)
)

    print("Starting YOLO + ByteTrack...")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps <= 0:
        raise RuntimeError("Could not determine video FPS.")

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    output_path = ROOT_DIR / "outputs" / "traffic_counted.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video_writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    results = model.track(
        source=str(VIDEO_PATH),
        tracker="bytetrack.yaml",
        device="cpu",
        conf=0.10,
        iou=0.50,
        imgsz=960,
        max_det=1000,
        persist=True,
        stream=True,
        save=True,
        project=str(ROOT_DIR / "outputs"),
        name="tracking_run",
        exist_ok=True,
        verbose=False
    )

    total_frames = 0
    tracked_ids = set()
    frame_index = 0

    for result in results:

        frame_index += 1

        analytics.update(result, frame_index, fps)
        counter.update(result)

        # Get YOLO annotated frame
        annotated_frame = result.plot()

        # Draw virtual line and vehicle counts
        annotated_frame = counter.draw(annotated_frame)

        # Save annotated frame
        video_writer.write(annotated_frame)


        total_frames += 1

        if result.boxes is not None and result.boxes.id is not None:

            boxes = result.boxes
            ids = boxes.id.int().cpu().tolist()
            classes = boxes.cls.int().cpu().tolist()

            for track_id, class_id in zip(ids, classes):
                tracked_ids.add((track_id, class_id))

        if total_frames % 30 == 0:
            print(f"Processed {total_frames} frames")

        video_writer.release()

        print(f"Counted video saved at: {output_path}")

    

    analytics.display_summary()
    analytics.display_density_summary()
    counter.display_summary()

    print("\nTracking completed!")
    print(f"Total frames processed: {total_frames}")
    print(f"Unique tracked IDs observed: {len(tracked_ids)}")
    print(f"Tracked video saved inside: {OUTPUT_PATH.parent / 'tracking_run'}")

    analytics.display_summary()
    analytics.display_density_summary()


if __name__ == "__main__":
    track_video()
