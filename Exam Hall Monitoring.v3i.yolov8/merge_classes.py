
from pathlib import Path
import shutil

ROOT = Path(".")
SPLITS = ["train", "valid", "test"]

# Original IDs:
# 0 = Cheating
# 1 = Not Cheating
# 2 = cheating
# 3 = not_cheating
#
# New IDs:
# 0 = Cheating
# 1 = Not_Cheating

BACKUP = ROOT / "labels_backup"

# Back up all original labels before changing anything.
if BACKUP.exists():
    raise SystemExit(
        "ERROR: labels_backup already exists. "
        "Stop and inspect it before running this script again."
    )

for split in SPLITS:
    labels_dir = ROOT / split / "labels"

    if not labels_dir.exists():
        raise SystemExit(f"Missing label folder: {labels_dir}")

    for label_file in labels_dir.glob("*.txt"):
        destination = BACKUP / split / label_file.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(label_file, destination)

# Merge class IDs in place.
for split in SPLITS:
    labels_dir = ROOT / split / "labels"

    for label_file in labels_dir.glob("*.txt"):
        updated_lines = []

        for line in label_file.read_text().splitlines():
            if not line.strip():
                continue

            parts = line.split()
            old_id = int(parts[0])

            if old_id == 0 or old_id == 2:
                parts[0] = "0"
            elif old_id == 1 or old_id == 3:
                parts[0] = "1"
            else:
                raise ValueError(
                    f"Unexpected class ID {old_id} in {label_file}"
                )

            updated_lines.append(" ".join(parts))

        label_file.write_text(
            "\n".join(updated_lines) + ("\n" if updated_lines else "")
        )

# Update the dataset configuration.
yaml_file = ROOT / "data.yaml"
yaml_backup = ROOT / "data.yaml.backup"

if not yaml_file.exists():
    raise SystemExit("data.yaml not found.")

if yaml_backup.exists():
    raise SystemExit(
        "data.yaml.backup already exists. Inspect it before proceeding."
    )

shutil.copy2(yaml_file, yaml_backup)

yaml_text = yaml_file.read_text()
yaml_text = yaml_text.replace(
    "nc: 4",
    "nc: 2"
)
yaml_text = yaml_text.replace(
    "names: ['Cheating', 'Not Cheating', 'cheating', 'not_cheating']",
    "names: ['Cheating', 'Not_Cheating']"
)
yaml_file.write_text(yaml_text)

print("Labels merged and original labels backed up.")
print("Review data.yaml before training.")
