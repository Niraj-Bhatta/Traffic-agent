
from pathlib import Path
from ultralytics import YOLO


# Project paths
ROOT_DIR = Path(__file__).resolve().parents[2]
IMAGE_PATH = ROOT_DIR / "data" / "traffic.jpg"
OUTPUT_DIR = ROOT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def detect_objects():
    # Load a lightweight YOLO model
    model = YOLO("yolo11n.pt")

    # Run object detection using CPU
  
    results = model.predict(
        source=str(IMAGE_PATH),
        device="cpu",
        conf=0.10,
        iou=0.50,
        imgsz=960,
        max_det=1000,
        verbose=False
    )

    # Process detection results
    for result in results:
        print("\nDetected objects:")

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            print(
                f"Object: {class_name} | "
                f"Confidence: {confidence:.2f}"
            )

        # Save image with bounding boxes
        output_path = OUTPUT_DIR / "detected_traffic.jpg"
        result.save(filename=str(output_path))

        print(f"\nAnnotated image saved to: {output_path}")


if __name__ == "__main__":
    detect_objects()
