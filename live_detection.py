import cv2
from pathlib import Path
from ultralytics import YOLO

# Update this path if your best.pt is stored elsewhere
MODEL_PATH = Path("/Users/mac1067/Documents/exam_cheating_ai/Exam Hall Monitoring.v3i.yolov8/runs/detect/runs/detect/merged_2class/weights/best.pt")

CONFIDENCE = 0.35
IMAGE_SIZE = 640
CAMERA_INDEX = 0

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH.resolve()}\n"
            "Run: find . -name 'best.pt'"
        )

    print("Loading your trained YOLOv8 model...")
    model = YOLO(str(MODEL_PATH))

    print("Model classes:", model.names)
    print("Opening camera...")

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open camera. Check macOS camera permissions "
            "and try CAMERA_INDEX = 1."
        )

    # Request a reasonable camera resolution
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read camera frame.")
                break

            # Run detection using your trained weights
            results = model.predict(
                source=frame,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                device="mps",
                verbose=False
            )

            result = results[0]
            annotated_frame = result.plot()

            # Display detections in the terminal
            detection_count = len(result.boxes) if result.boxes is not None else 0

            cv2.putText(
                annotated_frame,
                f"Detections: {detection_count}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            cv2.imshow("Exam Cheating AI - Live Detection", annotated_frame)

            # Press Q to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()