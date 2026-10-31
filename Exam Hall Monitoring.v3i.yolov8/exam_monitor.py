from ultralytics import YOLO
import cv2
from pathlib import Path

MODEL_PATH = "runs/detect/train/weights/best.pt"
VIDEO_PATH = "Exam Hall Monitoring.v3i.yolov8/3891949475-preview.mp4"
OUTPUT_DIR = Path("exam_monitor_output")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "exam_video_result.mp4"

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    fps,
    (width, height)
)

CLASS_NAMES = {
    0: "CHEATING",
    1: "NOT CHEATING",
    2: "CHEATING",
    3: "NOT CHEATING"
}

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(
        frame,
        imgsz=640,
        device="mps",
        conf=0.25,
        verbose=False
    )

    annotated = frame.copy()

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = CLASS_NAMES.get(class_id, f"CLASS_{class_id}")

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 0, 255) if label == "CHEATING" else (0, 255, 0),
                2
            )

            text = f"{label} {confidence:.2f}"

            cv2.putText(
                annotated,
                text,
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255) if label == "CHEATING" else (0, 255, 0),
                2
            )

    writer.write(annotated)

cap.release()
writer.release()

print(f"\nDone!")
print(f"Output saved to: {OUTPUT_VIDEO}")