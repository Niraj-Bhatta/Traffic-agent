
from src.vision import detect_objects
from src.video_detection import track_video



def main():
    print("=" * 45)
    print("       RoadSight - Traffic Intelligence")
    print("=" * 45)

    while True:
        print("\nSelect an operation:")
        print("1. Detect objects in an image")
        print("2. Detect objects in a video")
        print("0. Exit")

        choice = input("\nEnter your choice: ").strip()

        if choice == "1":
            print("\nStarting image detection...")
            detect_objects()

        elif choice == "2":
            print("\nStarting video detection...")
            track_video()

        elif choice == "0":
            print("\nExiting RoadSight. Goodbye!")
            break

        else:
            print("\nInvalid choice. Please select 0, 1, or 2.")


if __name__ == "__main__":
    main()
