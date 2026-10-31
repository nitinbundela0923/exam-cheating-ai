from pathlib import Path
from collections import Counter
import math

original = Path("Exam Hall Monitoring.v3i.yolov8_new")
cleaned = Path("clean_dataset_2class_v1")
mapping = {"0": "0", "1": "1", "2": "0", "3": "1"}

def read_labels(path, merge=False):
    result = []
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cls = mapping.get(p[0], p[0]) if merge else p[0]
        try:
            coords = tuple(float(v) for v in p[1:5])
        except ValueError:
            continue
        result.append((cls, coords))
    return result

for split in ("train", "valid", "test"):
    old_dir = original / split / "labels"
    new_dir = cleaned / split / "labels"

    checked = missing = count_diff = box_diff = 0
    class_diff = 0
    max_error = 0.0

    for old_file in old_dir.glob("*.txt"):
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

        # Compare boxes without relying on annotation row order.
        unmatched = list(new)
        for old_cls, old_box in old:
            best_i = None
            best_err = float("inf")

            for i, (new_cls, new_box) in enumerate(unmatched):
                err = max(abs(a - b) for a, b in zip(old_box, new_box))
                if err < best_err:
                    best_err, best_i = err, i

            if best_i is None:
                count_diff += 1
                break

            new_cls, _ = unmatched.pop(best_i)
            max_error = max(max_error, best_err)

            if old_cls != new_cls:
                class_diff += 1
            if best_err > 1e-5:
                box_diff += 1

    print(f"\n{split.upper()}")
    print("Matching files checked:", checked)
    print("Missing cleaned label files:", missing)
    print("Files with annotation-count differences:", count_diff)
    print("Boxes with coordinate differences:", box_diff)
    print("Class differences after intended merge:", class_diff)
    print("Largest coordinate difference:", max_error)
