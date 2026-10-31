from pathlib import Path
from collections import Counter

original = Path("Exam Hall Monitoring.v3i.yolov8_new")
cleaned = Path("clean_dataset_2class_v1")

# Original IDs -> cleaned IDs
mapping = {"0": "0", "1": "1", "2": "0", "3": "1"}

def read_labels(path, merge=False):
    boxes = []
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue

        cls = mapping.get(p[0], p[0]) if merge else p[0]
        try:
            values = [float(v) for v in p[1:]]
        except ValueError:
            continue

        # Standard YOLO detection box: class + 4 coordinates
        if len(values) == 4:
            xc, yc, w, h = values
            box = (xc - w/2, yc - h/2, xc + w/2, yc + h/2)

        # YOLO segmentation polygon: class + x/y pairs
        elif len(values) >= 6 and len(values) % 2 == 0:
            xs, ys = values[0::2], values[1::2]
            box = (min(xs), min(ys), max(xs), max(ys))
        else:
            continue

        boxes.append((cls, box))
    return boxes

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

for split in ("train", "valid", "test"):
    old_dir = original / split / "labels"
    new_dir = cleaned / split / "labels"

    checked = missing = count_diff = class_diff = 0
    matched = low_iou = 0
    ious = []

    for old_file in sorted(old_dir.glob("*.txt")):
        new_file = new_dir / old_file.name
        if not new_file.exists():
            missing += 1
            continue

        old = read_labels(old_file, merge=True)
        new = read_labels(new_file)
        checked += 1

        if len(old) != len(new):
            count_diff += 1
            continue

        # Greedy matching by highest IoU, independent of row order.
        candidates = []
        for i, (oc, ob) in enumerate(old):
            for j, (nc, nb) in enumerate(new):
                candidates.append((iou(ob, nb), i, j))

        used_old, used_new = set(), set()
        for score, i, j in sorted(candidates, reverse=True):
            if i in used_old or j in used_new:
                continue
            used_old.add(i)
            used_new.add(j)
            matched += 1
            ious.append(score)
            if score < 0.95:
                low_iou += 1
            if old[i][0] != new[j][0]:
                class_diff += 1

    print(f"\n{split.upper()}")
    print("Matching files checked:", checked)
    print("Missing cleaned label files:", missing)
    print("Files with annotation-count differences:", count_diff)
    print("Boxes matched:", matched)
    print("Matched boxes with IoU below 0.95:", low_iou)
    print("Class differences after merge:", class_diff)
    print("Mean box IoU:", round(sum(ious)/len(ious), 4) if ious else "N/A")
