
from collections import Counter


class TrafficAnalytics:

    def __init__(self):
        self.tracked_vehicles = {}
        self.total_detections = 0

    def update(self, result):

        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return

        if boxes.id is None:
            return

        track_ids = boxes.id.int().cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()

        for track_id, class_id in zip(track_ids, class_ids):

            class_name = result.names[class_id]

            self.total_detections += 1
            self.tracked_vehicles[track_id] = class_name

    def get_summary(self):

        vehicle_counts = Counter(
            self.tracked_vehicles.values()
        )

        return {
            "unique_tracked_objects": len(self.tracked_vehicles),
            "vehicle_breakdown": dict(vehicle_counts),
            "total_frame_detections": self.total_detections
        }

    def display_summary(self):

        summary = self.get_summary()

        print("\n" + "=" * 40)
        print("       ROADSIGHT TRAFFIC REPORT")
        print("=" * 40)

        print(
            f"\nUnique tracked objects: "
            f"{summary['unique_tracked_objects']}"
        )

        print("\nObject classification:")

        for name, count in summary["vehicle_breakdown"].items():
            print(f"{name}: {count}")

        print(
            f"\nTotal detections across frames: "
            f"{summary['total_frame_detections']}"
        )

        print("=" * 40)
