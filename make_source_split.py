from pathlib import Path
import re
import yaml

ROOT = Path("clean_dataset_2class_v1").resolve()
OUT = Path("source_split_v1").resolve()

# Assign each source group to exactly one split.
ASSIGNMENT = {
    "IMG_3485_MOV": "train",
    "IMG_4719_MOV": "valid",
    "IMG_9313_MOV": "test",
}

if OUT.exists():
    raise SystemExit(
        f"{OUT} already exists. Nothing changed. "
        "Choose a new output folder before continuing."
    )

# Verify source files and class configuration before creating output.
with (ROOT / "data.yaml").open() as f:
    config = yaml.safe_load(f)

if config.get("nc") != 2:
    raise SystemExit("Expected a two-class dataset.")

names = config.get("names")
if isinstance(names, list):
    names = {i: name for i, name in enumerate(names)}

if names != {0: "Cheating", 1: "Not Cheating"}:
    raise SystemExit(f"Unexpected class mapping: {names}")

items = []
pattern = re.compile(r"^(IMG_\d+_MOV)-\d+")

for old_split in ("train", "valid", "test"):
    image_dir = ROOT / old_split / "images"
    label_dir = ROOT / old_split / "labels"

    for image in sorted(image_dir.iterdir()):
        if not image.is_file():
            continue

        match = pattern.match(image.name)
        if not match:
            raise SystemExit(f"Unrecognized source filename: {image.name}")

        source = match.group(1)
        if source not in ASSIGNMENT:
            raise SystemExit(f"Unmapped source group: {source}")

        label = label_dir / f"{image.stem}.txt"
        if not label.is_file():
            raise SystemExit(f"Missing label: {label}")

        items.append((image, label, source, ASSIGNMENT[source]))

if len(items) != 4266:
    raise SystemExit(f"Expected 4266 images; found {len(items)}")

# Ensure the new destination doesn't overlap the original dataset.
if OUT == ROOT or ROOT in OUT.parents or OUT in ROOT.parents:
    raise SystemExit("Unsafe output location; nothing created.")

for split in ("train", "valid", "test"):
    (OUT / split / "images").mkdir(parents=True)
    (OUT / split / "labels").mkdir(parents=True)

    for image, label, source, destination in items:
        if destination != split:
            continue
        (OUT / split / "images" / image.name).symlink_to(image)
        (OUT / split / "labels" / label.name).symlink_to(label)

new_yaml = {
    "path": str(OUT),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": 2,
    "names": {0: "Cheating", 1: "Not Cheating"},
}

with (OUT / "data.yaml").open("w") as f:
    yaml.safe_dump(new_yaml, f, sort_keys=False)

print(f"\nCreated: {OUT}")
for split in ("train", "valid", "test"):
    image_count = len(list((OUT / split / "images").iterdir()))
    label_count = len(list((OUT / split / "labels").iterdir()))
    sources = sorted({
        source for _, _, source, destination in items
        if destination == split
    })
    print(f"{split}: {image_count} images, {label_count} labels, sources={sources}")

print("Original dataset and model were not modified.")
