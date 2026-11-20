
from collections import Counter, defaultdict


class TrafficAnalytics:
    def __init__(self):
        self.tracked_vehicles = {}
        self.total_detections = 0

        self.vehicle_classes = {
            "car",
            "motorcycle",
            "bus",
            "truck"
        }

        self.density_data = defaultdict(list)
        self.interval_seconds = 10

    def update(self, result, frame_index, fps):
        boxes = result.boxes

        # Get detected class IDs
        if boxes is not None and len(boxes) > 0:
            class_ids = boxes.cls.int().cpu().tolist()
        else:
            class_ids = []

        # Count vehicles visible in the current frame
        current_vehicle_count = sum(
            1 for class_id in class_ids
            if result.names[class_id] in self.vehicle_classes
        )

        # Calculate timestamp using actual video FPS
        timestamp = (frame_index - 1) / fps
        interval = int(timestamp // self.interval_seconds)

        self.density_data[interval].append(current_vehicle_count)

        # Update tracking analytics when track IDs are available
        if boxes is None or len(boxes) == 0 or boxes.id is None:
            return

        track_ids = boxes.id.int().cpu().tolist()

        for track_id, class_id in zip(track_ids, class_ids):
            class_name = result.names[class_id]

            if class_name in self.vehicle_classes:
                self.total_detections += 1
                self.tracked_vehicles[(class_name, track_id)] = class_name

    def get_summary(self):
        vehicle_counts = Counter(self.tracked_vehicles.values())

        return {
            "unique_tracked_vehicles": len(self.tracked_vehicles),
            "vehicle_breakdown": dict(vehicle_counts),
            "total_tracked_detections": self.total_detections
        }

    def display_summary(self):
        summary = self.get_summary()

        print("\n" + "=" * 45)
        print("       TRAFFIC ANALYTICS REPORT")
        print("=" * 45)

        print(f"Unique tracked vehicles: {summary['unique_tracked_vehicles']}")
        print(f"Total tracked detections: {summary['total_tracked_detections']}")

        print("\nVehicle breakdown:")

        for vehicle, count in summary["vehicle_breakdown"].items():
            print(f"{vehicle}: {count}")

        print("=" * 45)

    def get_density_summary(self):
        density_summary = {}

        for interval, counts in self.density_data.items():
            average_count = sum(counts) / len(counts)

            if average_count < 10:
                level = "Low"
            elif average_count < 25:
                level = "Medium"
            else:
                level = "High"

            start_time = interval * self.interval_seconds
            end_time = start_time + self.interval_seconds

            density_summary[f"{start_time}-{end_time}s"] = {
                "average_vehicles": round(average_count, 2),
                "traffic_level": level
            }

        return density_summary

    def display_density_summary(self):
        summary = self.get_density_summary()

        print("\n" + "=" * 45)
        print("       TRAFFIC DENSITY REPORT")
        print("=" * 45)

        for interval, data in summary.items():
            print(f"\nTime: {interval}")
            print(f"Average vehicles: {data['average_vehicles']}")
            print(f"Traffic level: {data['traffic_level']}")

        print("=" * 45)
