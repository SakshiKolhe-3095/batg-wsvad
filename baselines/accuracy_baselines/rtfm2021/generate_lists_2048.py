from pathlib import Path

TRAIN_DIR = Path(r"D:\Research_Internship\data\features\ucf-crime\i3d_2048\train")
TEST_DIR = Path(r"D:\Research_Internship\data\features\ucf-crime\i3d_2048\test\UCF_Test_ten_i3d")

train_files = sorted(TRAIN_DIR.glob("*.npy"))
abnormal = [p for p in train_files if not p.name.startswith("Normal_Videos")]
normal = [p for p in train_files if p.name.startswith("Normal_Videos")]
ordered_train = abnormal + normal

with open("list/ucf-i3d-2048.list", "w") as f:
    for p in ordered_train:
        f.write(str(p).replace("\\", "/") + "\n")

test_files = sorted(TEST_DIR.glob("*.npy"))
with open("list/ucf-i3d-test-2048.list", "w") as f:
    for p in test_files:
        f.write(str(p).replace("\\", "/") + "\n")

print(f"Train: {len(ordered_train)} ({len(abnormal)} abnormal + {len(normal)} normal)")
print(f"Test: {len(test_files)}")