from pathlib import Path

TRAIN_DIR = Path(r"D:\Research_Internship\data\features\ucf-crime\i3d\train_merged")
TEST_DIR = Path(r"D:\Research_Internship\data\features\ucf-crime\i3d\test_merged")

train_files = sorted(TRAIN_DIR.glob("*.npy"))
test_files = sorted(TEST_DIR.glob("*.npy"))

abnormal = [p for p in train_files if not p.name.startswith("Normal_Videos")]
normal = [p for p in train_files if p.name.startswith("Normal_Videos")]

print(f"Abnormal count: {len(abnormal)} (expected 810)")
print(f"Normal count: {len(normal)} (expected 800)")

ordered_train = abnormal + normal

with open("baselines/accuracy_baselines/deepmil/list/ucf-i3d.list", "w") as f:
    for p in ordered_train:
        f.write(str(p.resolve()) + "\n")

with open("baselines/accuracy_baselines/deepmil/list/ucf-i3d-test.list", "w") as f:
    for p in test_files:
        f.write(str(p.resolve()) + "\n")

print(f"Wrote {len(ordered_train)} train paths (abnormal first, normal last) and {len(test_files)} test paths")
