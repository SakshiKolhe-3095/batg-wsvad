# Data — Download & Setup Instructions

This folder is gitignored (`data/raw/`, `data/features/`) — nothing here gets pushed to GitHub.
Large files (features, raw video) live on our **shared Google Drive folder**: https://drive.google.com/drive/u/0/folders/1ddTzqVIA8Un4Fd49QZTbr-kguEJhEHcC

---

## 1. UCF-Crime — Precomputed I3D Features (primary)

**⚠️ RTFM's official download links are BROKEN — confirmed 2026-07-03.**
GitHub issues #105, #98, #97 on tianyu0207/RTFM all confirm Google Drive and OneDrive
links are dead. Do NOT waste time trying them.

**Working source: DeepMIL / Roc-Ng repo**
- Repo: https://github.com/Roc-Ng/DeepMIL
- UCF-Crime 10-crop I3D features available here
- Downloaded and processed by Mumtaj on 2026-07-03 ✅

**What Mumtaj downloaded and processed:**
- Train: 16,100 raw per-crop files (1610 videos × 10 crops)
- Test: 2,900 raw per-crop files (290 videos × 10 crops)
- Merged into shape (T, 10, 1024) per video
- Fixed 71 shape-mismatch videos (off-by-one frame, trimmed cleanly)
- Generated correct .list files with abnormal-first split
- Fixed forward-slash path encoding in list files

**⚠️ Feature dimension is 1024, NOT 2048.**
DeepMIL uses a different I3D backbone than RTFM's original. Mumtaj patched
`model.py` and `option.py` to use `feature-size=1024`. Our BATG pipeline
must also use 1024-dim features — do not assume 2048 anywhere in src/models/.

**RTFM patches applied by Mumtaj (document here for reproducibility):**
- `model.py` — removed 4 hardcoded 2048s, parameterized via n_features/len_feature
- `option.py` — feature-size=1024, UCF list paths, workers=0, dataset=ucf
- `utils.py` — np.int → int (numpy deprecation fix)
- `test_10crop.py` — correct squeeze/mean dimensions, gt[:len(pred)] trim, CPU tensor handling
- `main.py` — pin_memory=False, CUDA generator for shuffle, num_workers=0

**Features on shared Drive:** upload in progress — check Drive folder link at top of this file.

**Local structure (Mumtaj's machine):**
```
data/features/ucf-crime/
├── i3d_merged/
│   ├── train/    # 1610 .npy files, shape (T, 10, 1024)
│   └── test/     # 290 .npy files, shape (T, 10, 1024)
└── list/
    ├── ucf-i3d.list          # train list, local paths
    └── ucf-i3d-test.list     # test list, local paths
```

**RTFM pipeline confirmed working end-to-end — AUC 0.5513 after 1 epoch (2026-07-03) ✅**
(Low AUC expected at epoch 1 — needs 50-100 epochs to reach published ~84%.
This just confirms pipeline runs without error.)

---

## 2. UCF-Crime — Precomputed C3D / ResNet Features (for Sultani 2018 baseline)

The `ekosman/AnomalyDetectionCVPR2018-Pytorch` repo (our Sultani 2018 baseline) provides
its own precomputed features separately — check that repo's README/releases for download
links once cloned into `baselines/accuracy_baselines/sultani2018/`. Don't assume RTFM's
I3D features are directly swappable here; Sultani's original method expects C3D or
ResNet-based features depending on which variant of the repo we use.

**Confirmed feature download links (from ekosman repo README):**
- C3D features: https://drive.google.com/drive/folders/1rhOuAdUqyJU4hXIhToUnh5XVvYjQiN50?usp=sharing
- ResNet-101 features: https://drive.google.com/file/d/1kQAvOhtL-sGadblfd3NmDirXq8vYQPvf/view?usp=sharing
- ResNet-152 features: https://drive.google.com/file/d/17wdy_DS9UY37J9XTV5XCLqxOFgXiv3ZK/view
- Pre-trained anomaly detector models: see `baselines/accuracy_baselines/sultani2018/exps/` folder (already cloned)

**Note:** These are NOT the same features as DeepMIL/RTFM — different backbone (C3D/ResNet vs I3D).
Not downloaded yet — only needed when actually running Sultani baseline reproduction.

---

## 3. UCF-Crime — Raw Videos (only if needed later)

We are **not planning to extract features ourselves** for the main experiments — using
precomputed features above avoids burning Colab/AI Kosh time on extraction. Raw video
download is only needed if:
- We need a custom feature extraction step for BATG's cheap pre-gating signals
  (motion energy / frame difference) that aren't already captured in precomputed features, or
- A baseline repo requires raw video input directly.

Official UCF-Crime dataset page (if raw video becomes necessary):
https://www.crcv.ucf.edu/projects/real-world/

**Action needed:** confirm with the group before downloading raw video — it's large
(100+ GB) and probably unnecessary for our current plan.

---

## 4. XD-Violence (stretch goal dataset)

Not downloaded yet — only needed if we get to the generalization-check stretch goal
after main UCF-Crime results are done. Official page:
https://roc-ng.github.io/XD-Violence/

---

## Folder structure reminder

```
data/
├── raw/                  # gitignored — raw video, only if absolutely needed
├── features/             # gitignored — all precomputed features go here
│   ├── ucf-crime/
│   │   ├── i3d/           # RTFM-style I3D features (train/test)
│   │   └── c3d_or_resnet/ # Sultani-baseline features (whichever this repo uses)
│   └── xd-violence/       # stretch goal, empty for now
└── README_data.md         # this file
```

## Rules
1. Never commit anything inside `raw/` or `features/` — `.gitignore` already blocks this, don't override it.
2. All large files go to the shared Google Drive folder — link at top of this file.
3. If you download something not listed here, add it to this README immediately so the other person doesn't redo the work or get confused about where files came from.
4. Note exact source URL + date downloaded for anything added — links break, dates help us track if a re-download is needed later.



## C3D Features (Sultani 2018 baseline)

**Source:** ekosman/AnomalyDetectionCVPR2018-Pytorch precomputed C3D features
**Drive link:** https://drive.google.com/drive/folders/1rhOuAdUqyJU4hXIhToUnh5XVvYjQiN50

**Downloaded:** 2026-07-08
**Local path:** `data/features/ucf-crime/c3d/`

**Folder structure** (16 category folders, matches UCF-Crime annotation format):
```
data/features/ucf-crime/c3d/
├── Abuse/
├── Arrest/
├── Arson/
├── Assault/
├── Burglary/
├── Explosion/
├── Fighting/
├── Normal_Videos_for_Event_Recognition/
├── RoadAccidents/
├── Robbery/
├── Shooting/
├── Shoplifting/
├── Stealing/
├── Testing_Normal_Videos_Anomaly/
├── Training_Normal_Videos_Anomaly/
└── Vandalism/
```

**File format:** one `.txt` file per video (e.g. `Abuse001_x264.txt`), containing
space-separated floating-point C3D feature values (verified readable,
non-corrupted 2026-07-08).

**Annotation match confirmed:** paths in `Train_Annotation.txt` /
`Test_Annotation.txt` (e.g. `Abuse/Abuse022_x264.mp4`) match folder/file naming
here exactly (`.mp4` → `.txt` extension swap handled by Sultani's
`features_loader.py`, standard for precomputed-feature loaders).

**Note:** Skipped `desk.zip` (828 MB) present in the shared Drive folder —
different/unverified owner, flagged by Google as unscannable for viruses.
Not part of the official feature set, ignored.

**Status:** Ready for Sultani baseline (Phase 2, Mumtaj's task).