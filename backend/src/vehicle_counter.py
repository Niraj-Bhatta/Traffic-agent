import cv2



class VehicleCounter:
    def __init__(self, line_start, line_end):
        self.line_start = line_start
        self.line_end = line_end

        self.previous_positions = {}
        self.counted_ids = set()

        self.vehicle_counts = {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0
        }

    def _get_side(self, point):
        x1, y1 = self.line_start
        x2, y2 = self.line_end
        px, py = point

        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    def update(self, result):
        boxes = result.boxes

        if boxes is None or len(boxes) == 0 or boxes.id is None:
            return

        track_ids = boxes.id.int().cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()
        coordinates = boxes.xyxy.cpu().tolist()

        for track_id, class_id, box in zip(
            track_ids, class_ids, coordinates
        ):
            class_name = result.names[class_id]

            if class_name not in self.vehicle_counts:
                continue

            x1, y1, x2, y2 = box

            # Bottom-center point of the vehicle
            center = (
                int((x1 + x2) / 2),
                int(y2)
            )

            current_side = self._get_side(center)

            if track_id in self.previous_positions:
                previous_side = self.previous_positions[track_id]

                crossed = (
                    previous_side * current_side < 0
                )

                if crossed and track_id not in self.counted_ids:
                    self.vehicle_counts[class_name] += 1
                    self.counted_ids.add(track_id)

                    print(
                        f"Vehicle crossed: {class_name} "
                        f"| Track ID: {track_id}"
                    )

            self.previous_positions[track_id] = current_side

    def get_summary(self):
        return {
            "total": sum(self.vehicle_counts.values()),
            "by_class": self.vehicle_counts.copy()
        }

    def display_summary(self):
        summary = self.get_summary()

        print("\n" + "=" * 45)
        print("       VIRTUAL LINE COUNT REPORT")
        print("=" * 45)

        print(f"Total vehicles crossed: {summary['total']}")

        for vehicle, count in summary["by_class"].items():
            print(f"{vehicle}: {count}")

        print("=" * 45)



    def draw(self, frame):
    

        # Draw virtual counting line
        cv2.line(
            frame,
            self.line_start,
            self.line_end,
            (0, 255, 255),
            3
        )

        # Display total count
        total = sum(self.vehicle_counts.values())

        cv2.putText(
            frame,
            f"Vehicles Counted: {total}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Display individual vehicle counts
        y_position = 90

        for vehicle, count in self.vehicle_counts.items():
            cv2.putText(
                frame,
                f"{vehicle}: {count}",
                (30, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            y_position += 35

        return frame
