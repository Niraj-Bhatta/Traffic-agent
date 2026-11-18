
from pathlib import Path
from ultralytics import YOLO
from src.traffic_analytics import TrafficAnalytics


ROOT_DIR = Path(__file__).resolve().parents[2]

VIDEO_PATH = ROOT_DIR / "data" / "traffic.mp4"
OUTPUT_PATH = ROOT_DIR / "outputs" / "tracked_traffic.mp4"


def track_video():

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    model = YOLO("yolo11n.pt")

    analytics = TrafficAnalytics()

    print("Starting YOLO + ByteTrack...")

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

    for result in results:

        analytics.update(result)

        total_frames += 1

        if result.boxes is not None and result.boxes.id is not None:

            boxes = result.boxes
            ids = boxes.id.int().cpu().tolist()
            classes = boxes.cls.int().cpu().tolist()

            for track_id, class_id in zip(ids, classes):
                tracked_ids.add((track_id, class_id))

        if total_frames % 30 == 0:
            print(f"Processed {total_frames} frames")

    print("\nTracking completed!")
    print(f"Total frames processed: {total_frames}")
    print(f"Unique tracked IDs observed: {len(tracked_ids)}")
    print(f"Tracked video saved inside: {OUTPUT_PATH.parent / 'tracking_run'}")

    analytics.display_summary()


if __name__ == "__main__":
    track_video()
