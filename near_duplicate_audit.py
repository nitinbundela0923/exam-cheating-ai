from pathlib import Path
from PIL import Image
import imagehash
from collections import defaultdict
import itertools

ROOT = Path("clean_dataset_2class_v1")
SPLITS = ("train", "valid", "test")
HASH_SIZE = 8
MAX_DISTANCE = 6

images = []

for split in SPLITS:
    folder = ROOT / split / "images"
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in {
            ".jpg", ".jpeg", ".png", ".webp", ".bmp"
        }:
            continue
        try:
            with Image.open(path) as im:
                phash = imagehash.phash(im.convert("RGB"), hash_size=HASH_SIZE)
            images.append((split, path, phash))
        except Exception as exc:
            print(f"Could not read {path}: {exc}")

print(f"Images hashed: {len(images)}")
print("Comparing images across different splits...")

# Group candidates by coarse hash prefixes to avoid an all-pairs comparison.
# For 64-bit hashes, use four 16-bit chunks.
def chunks(value):
    number = int(str(value), 16)
    return tuple(
        (number >> (16 * i)) & 0xFFFF
        for i in range(4)
    )
    

buckets = defaultdict(list)
for idx, (_, _, h) in enumerate(images):
    for chunk_idx, chunk in enumerate(chunks(str(h))):
        buckets[(chunk_idx, chunk)].append(idx)

candidate_pairs = set()
for members in buckets.values():
    for a, b in itertools.combinations(members, 2):
        if images[a][0] != images[b][0]:
            candidate_pairs.add(tuple(sorted((a, b))))

matches = []
for a, b in candidate_pairs:
    split_a, path_a, hash_a = images[a]
    split_b, path_b, hash_b = images[b]
    distance = hash_a - hash_b
    if distance <= MAX_DISTANCE:
        matches.append((distance, split_a, str(path_a), split_b, str(path_b)))

matches.sort()

out = Path("near_duplicate_report.csv")
import csv
with out.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["phash_distance", "split_a", "image_a", "split_b", "image_b"])
    writer.writerows(matches)

print(f"Potential cross-split near-duplicate pairs: {len(matches)}")
print(f"Report saved to: {out.resolve()}")
print("\nFirst 20 matches:")
for row in matches[:20]:
    print(row)
print("\nNote: Similarity matches need visual review; they are not automatically confirmed leakage.")

