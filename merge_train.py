import numpy as np
import re
from pathlib import Path

SOURCE_ROOT = Path(r"D:\Research_Internship\data\features\ucf-crime\i3d\deepmil_train")
DEST_DIR = Path(r"D:\Research_Internship\data\features\ucf-crime\i3d\train_merged")
DEST_DIR.mkdir(parents=True, exist_ok=True)

all_npy_files = list(SOURCE_ROOT.rglob("*.npy"))
print(f"Found {len(all_npy_files)} total .npy files to process")

video_groups = {}

for f in all_npy_files:
    match = re.match(r"^(.+)__(\d)\.npy$", f.name)
    if match:
        base_name, crop_idx = match.group(1), int(match.group(2))
    else:
        base_name = f.name[:-4]
        crop_idx = 0
    video_groups.setdefault(base_name, {})[crop_idx] = f

print(f"Found {len(video_groups)} unique videos")

incomplete = []
unfixable_mismatch = []
trimmed_count = 0
merged_count = 0

for base_name, crops in video_groups.items():
    if len(crops) != 10:
        incomplete.append((base_name, sorted(crops.keys())))
        continue

    arrays = [np.load(crops[i]) for i in range(10)]
    shapes = [a.shape for a in arrays]

    if len(set(shapes)) > 1:
        lengths = [s[0] for s in shapes]
        min_len = min(lengths)
        max_len = max(lengths)
        # Only auto-fix if the gap is small (<=2 frames) - trim everything to min_len
        if max_len - min_len <= 2:
            arrays = [a[:min_len] for a in arrays]
            trimmed_count += 1
        else:
            unfixable_mismatch.append((base_name, shapes))
            continue

    stacked = np.stack(arrays, axis=1)
    out_path = DEST_DIR / f"{base_name}_i3d.npy"
    np.save(out_path, stacked)
    merged_count += 1

print(f"Merged {merged_count} videos successfully ({trimmed_count} required trimming to align crop lengths)")

if incomplete:
    print(f"INCOMPLETE ({len(incomplete)} videos missing some crops):")
    for name, have in incomplete[:20]:
        print(f"  {name}: has crops {have}")

if unfixable_mismatch:
    print(f"UNFIXABLE MISMATCH ({len(unfixable_mismatch)} videos with gap > 2 frames):")
    for name, shapes in unfixable_mismatch[:20]:
        print(f"  {name}: shapes = {shapes}")
