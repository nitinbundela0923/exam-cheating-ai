from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="Exam_Cheating/data.yaml",
    epochs=50,
    imgsz=640,
    batch=4,
    patience=20
)
