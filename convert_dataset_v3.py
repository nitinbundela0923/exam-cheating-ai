from pathlib import Path
import shutil
import math

BASE = Path.home() / "Documents" / "exam_cheating_ai"
SOURCE = BASE / "Exam Hall Monitoring.v3i.yolov8_new"
DEST = BASE / "clean_dataset_v3"

if not SOURCE.is_dir():
    raise SystemExit(f"Source dataset not found: {SOURCE}")

if DEST.exists():
    raise SystemExit(
        f"{DEST} already exists. Stop and inspect it before continuing."
    )

shutil.copytree(SOURCE, DEST, ignore=shutil.ignore_patterns("data.yaml"))

converted_polygons = 0
existing_boxes = 0

for split in ("train", "valid", "test"):
    for file in (DEST / split / "labels").glob("*.txt"):
        output = []

        for line_no, line in enumerate(file.read_text().splitlines(), 1):
            parts = line.split()
            if not parts:
                continue

            try:
                cls = int(parts[0])
                coords = list(map(float, parts[1:]))

                if cls not in range(4):
                    raise ValueError("class ID must be 0–3")
                if not all(math.isfinite(v) for v in coords):
                    raise ValueError("non-finite coordinate")

                if len(coords) == 4:
                    xc, yc, w, h = coords
                    if not (
                        0 <= xc <= 1 and 0 <= yc <= 1
                        and 0 < w <= 1 and 0 < h <= 1
                        and xc - w / 2 >= -1e-5
                        and yc - h / 2 >= -1e-5
                        and xc + w / 2 <= 1 + 1e-5
                        and yc + h / 2 <= 1 + 1e-5
                    ):
                        raise ValueError("invalid bounding box")
                    existing_boxes += 1

                else:
                    if len(coords) < 6 or len(coords) % 2:
                        raise ValueError("invalid polygon coordinate count")
                    if not all(0 <= v <= 1 for v in coords):
                        raise ValueError("polygon coordinate outside 0–1")

                    xs, ys = coords[0::2], coords[1::2]
                    xc = (min(xs) + max(xs)) / 2
                    yc = (min(ys) + max(ys)) / 2
                    w = max(xs) - min(xs)
                    h = max(ys) - min(ys)

                    if w <= 0 or h <= 0:
                        raise ValueError("zero-area polygon")
                    converted_polygons += 1

                output.append(
                    f"{cls} {xc:.8f} {yc:.8f} {w:.8f} {h:.8f}"
                )

            except ValueError as exc:
                raise SystemExit(
                    f"Stopped at {file}, line {line_no}: {exc}\n{line}"
                )

        file.write_text("\n".join(output) + ("\n" if output else ""))

(DEST / "data.yaml").write_text(
    f"""path: {DEST}
train: train/images
val: valid/images
test: test/images
nc: 4
names:
  0: Cheating
  1: Not Cheating
  2: cheating
  3: not_cheating
"""
)

print("Conversion completed successfully.")
print(f"Clean dataset: {DEST}")
print(f"Existing boxes retained: {existing_boxes}")
print(f"Polygons converted: {converted_polygons}")
print("Original dataset was not modified.")
