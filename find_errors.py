
from pathlib import Path
from collections import Counter
import csv

import cv2
from ultralytics import YOLO

ROOT = Path.home() / "Documents" / "exam_cheating_ai"
DATASET = ROOT / "clean_dataset_2class_v1"
MODEL_PATH = ROOT / "runs/detect/two_class_setup_test/weights/best.pt"
OUT = ROOT / "runs/detect/two_class_error_review_v2"


CONF = 0.25
IOU_MATCH = 0.50
IMAGE_SIZE = 640
NAMES = {0: "Cheating", 1: "Not Cheating"}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

if not MODEL_PATH.is_file():
    raise SystemExit(f"Model not found: {MODEL_PATH}")

if not (DATASET / "test/images").is_dir():
    raise SystemExit(f"Test images folder missing: {DATASET}")

if OUT.exists():
    raise SystemExit(
        f"Output already exists: {OUT}\n"
        "Change OUT to a new folder name before rerunning."
    )

for folder in ("missed", "false_positive", "class_mismatch"):
    (OUT / folder).mkdir(parents=True)

def read_gt(path, width, height):
    """Read YOLO normalized labels and convert boxes to pixels."""
    items = []

    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        parts = line.split()
        if not parts:
            continue

        if len(parts) != 5:
            raise ValueError(
                f"Expected class and 4 box values at {path}:{line_no}"
            )

        cls = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:])

        if cls not in NAMES:
            raise ValueError(f"Unexpected class {cls} in {path}")

        if not all(0 <= v <= 1 for v in (xc, yc, bw, bh)):
            raise ValueError(f"Invalid normalized coordinates in {path}")

        items.append({
            "cls": cls,
            "box": [
                (xc - bw / 2) * width,
                (yc - bh / 2) * height,
                (xc + bw / 2) * width,
                (yc + bh / 2) * height,
            ],
        })

    return items

def box_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0

def match_boxes(gt, preds):
    # Class-agnostic, one-to-one greedy IoU matching.
    candidates = []

    for gi, g in enumerate(gt):
        for pi, p in enumerate(preds):
            overlap = box_iou(g["box"], p["box"])
            if overlap >= IOU_MATCH:
                candidates.append((overlap, gi, pi))

    candidates.sort(reverse=True)
    used_gt, used_pred, matches = set(), set(), []

    for overlap, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append((gi, pi, overlap))

    return used_gt, used_pred, matches

def draw_box(image, box, color, text):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(round, box)
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    ty = y1 - 5 if y1 > 22 else min(h - 5, y1 + 18)
    cv2.putText(
        image, text, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX,
        0.45, color, 1, cv2.LINE_AA
    )

model = YOLO(str(MODEL_PATH))
image_dir = DATASET / "test/images"
label_dir = DATASET / "test/labels"

images = sorted(
    p for p in image_dir.iterdir()
    if p.is_file() and p.suffix.lower() in EXTENSIONS
)

if not images:
    raise SystemExit("No test images found.")

rows = []
totals = Counter()
checked = 0

for number, image_path in enumerate(images, 1):
    label_path = label_dir / f"{image_path.stem}.txt"

    if not label_path.is_file():
        print("Missing label:", label_path.name)
        totals["missing_label_files"] += 1
        continue

    image = cv2.imread(str(image_path))
    if image is None:
        print("Unreadable image:", image_path.name)
        totals["unreadable_images"] += 1
        continue

    height, width = image.shape[:2]
    gt = read_gt(label_path, width, height)

    result = model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        device="mps",
        conf=CONF,
        verbose=False,
    )[0]

    preds = []
    if result.boxes is not None:
        for b in result.boxes:
            preds.append({
                "cls": int(b.cls.item()),
                "conf": float(b.conf.item()),
                "box": b.xyxy[0].cpu().tolist(),
            })

    used_gt, used_pred, matches = match_boxes(gt, preds)
    errors = []

    for gi, item in enumerate(gt):
        if gi not in used_gt:
            errors.append({
                "error": "missed",
                "gt": item,
                "pred": None,
                "iou": None,
            })
            totals["missed"] += 1
            totals[f"missed_{item['cls']}"] += 1

    for pi, item in enumerate(preds):
        if pi not in used_pred:
            errors.append({
                "error": "false_positive",
                "gt": None,
                "pred": item,
                "iou": None,
            })
            totals["false_positive"] += 1
            totals[f"false_positive_{item['cls']}"] += 1

    for gi, pi, overlap in matches:
        if gt[gi]["cls"] != preds[pi]["cls"]:
            errors.append({
                "error": "class_mismatch",
                "gt": gt[gi],
                "pred": preds[pi],
                "iou": overlap,
            })
            totals["class_mismatch"] += 1

    checked += 1

    if errors:
        annotated = image.copy()

        # Green = ground truth; blue = matched prediction;
        # red = unmatched prediction.
        for item in gt:
            draw_box(
                annotated, item["box"], (0, 200, 0),
                f"GT: {NAMES[item['cls']]}"
            )

        for pi, item in enumerate(preds):
            color = (255, 120, 0) if pi in used_pred else (0, 0, 255)
            draw_box(
                annotated, item["box"], color,
                f"P: {NAMES[item['cls']]} {item['conf']:.2f}"
            )

        annotated_path = OUT / f"{image_path.stem}_review.jpg"
        cv2.imwrite(str(annotated_path), annotated)

        for error in errors:
            gt_item = error["gt"]
            pred_item = error["pred"]

            rows.append({
                "image": image_path.name,
                "error": error["error"],
                "gt_class": (
                    NAMES[gt_item["cls"]] if gt_item else ""
                ),
                "pred_class": (
                    NAMES[pred_item["cls"]] if pred_item else ""
                ),
                "confidence": (
                    round(pred_item["conf"], 4) if pred_item else ""
                ),
                "iou": (
                    round(error["iou"], 4)
                    if error["iou"] is not None else ""
                ),
            })

            category = OUT / error["error"]
            cv2.imwrite(
                str(category / annotated_path.name), annotated
            )

    if number % 50 == 0 or number == len(images):
        print(f"Processed {number}/{len(images)} images")

fields = [
    "image", "error", "gt_class",
    "pred_class", "confidence", "iou"
]

csv_path = OUT / "error_summary.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

summary = [
    "YOLO TEST ERROR REVIEW",
    f"Images discovered: {len(images)}",
    f"Images checked: {checked}",
    f"Confidence threshold: {CONF}",
    f"IoU threshold: {IOU_MATCH}",
    f"Missed boxes: {totals['missed']}",
    f"  Cheating: {totals['missed_0']}",
    f"  Not Cheating: {totals['missed_1']}",
    f"Unmatched predictions: {totals['false_positive']}",
    f"  Cheating: {totals['false_positive_0']}",
    f"  Not Cheating: {totals['false_positive_1']}",
    f"Class mismatches: {totals['class_mismatch']}",
    f"Missing labels: {totals['missing_label_files']}",
    f"Unreadable images: {totals['unreadable_images']}",
    f"CSV report: {csv_path}",
]

(OUT / "summary.txt").write_text("\n".join(summary) + "\n")
print("\n" + "\n".join(summary))
print("Output folder:", OUT)
