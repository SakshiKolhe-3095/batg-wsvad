from pathlib import Path

TRAIN_DIR = Path(r"D:\Research_Internship\data\features\ucf-crime\i3d\train_merged")
TEST_DIR = Path(r"D:\Research_Internship\data\features\ucf-crime\i3d\test_merged")

train_files = sorted(TRAIN_DIR.glob("*.npy"))
test_files = sorted(TEST_DIR.glob("*.npy"))

abnormal = [p for p in train_files if not p.name.startswith("Normal_Videos")]
normal = [p for p in train_files if p.name.startswith("Normal_Videos")]
ordered_train = abnormal + normal

rtfm_list_dir = Path("baselines/accuracy_baselines/rtfm2021/list")
deepmil_list_dir = Path("baselines/accuracy_baselines/deepmil/list")

for list_dir in [rtfm_list_dir, deepmil_list_dir]:
    with open(list_dir / "ucf-i3d.list", "w") as f:
        for p in ordered_train:
            f.write(str(p).replace("\\", "/") + "\n")
    with open(list_dir / "ucf-i3d-test.list", "w") as f:
        for p in test_files:
            f.write(str(p).replace("\\", "/") + "\n")

print(f"Wrote {len(ordered_train)} train and {len(test_files)} test paths to both repos")
print("Sample train path:", str(ordered_train[0]).replace("\\", "/"))
print("Sample test path:", str(test_files[0]).replace("\\", "/"))
