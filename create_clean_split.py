from pathlib import Path
import shutil
import yaml

root = Path("Exam Hall Monitoring.v3i.yolov8").resolve()
out = Path("clean_dataset_v1").resolve()

# Keep each source video entirely within one split.
assignment = {
    "IMG_3485": "train",
    "IMG_4719": "valid",
    "IMG_9313": "test",
}

if out.exists():
    raise SystemExit(
        f"{out} already exists. Nothing was changed. "
        "Choose a new output folder name before retrying."
    )

for split in ("train", "valid", "test"):
    (out / split / "images").mkdir(parents=True)
    (out / split / "labels").mkdir(parents=True)

copied = {s: 0 for s in ("train", "valid", "test")}
missing = []

for old_split in ("train", "valid", "test"):
    image_dir = root / old_split / "images"
    label_dir = root / old_split / "labels"

    for img in image_dir.iterdir():
        if not img.is_file():
            continue

        source = img.stem.split("_f")[0]
        if source not in assignment:
            continue

        new_split = assignment[source]
        label = label_dir / f"{img.stem}.txt"

        if not label.is_file():
            missing.append(str(label))
            continue

        target_img = out / new_split / "images" / img.name
        target_label = out / new_split / "labels" / label.name

        # If a frame appears in more than one old split, copy it only once.
        if target_img.exists():
            continue

        shutil.copy2(img, target_img)
        shutil.copy2(label, target_label)
        copied[new_split] += 1

if missing:
    shutil.rmtree(out)
    raise SystemExit(
        "Missing labels found; output removed. Examples: "
        + ", ".join(missing[:5])
    )

data = {
    "path": str(out),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": 2,
    "names": {
        0: "Cheating",
        1: "Not_Cheating",
    },
}

with (out / "data.yaml").open("w") as f:
    yaml.safe_dump(data, f, sort_keys=False)

print("\nCreated separate dataset:", out)
for split, count in copied.items():
    print(f"{split}: {count} images")
print("Original dataset and model were not modified.")
print("Configuration:", out / "data.yaml")
