# Data — Download & Setup Instructions

This folder is gitignored (`data/raw/`, `data/features/`) — nothing here gets pushed to GitHub.
Large files (features, raw video) live on our **shared Google Drive folder**: https://drive.google.com/drive/u/0/folders/1ddTzqVIA8Un4Fd49QZTbr-kguEJhEHcC

---

## 1. UCF-Crime — Precomputed I3D Features (primary, via RTFM repo)

RTFM (Tian et al., ICCV 2021) provides precomputed I3D (ResNet-50 backbone, 10-crop augmented)
features for UCF-Crime directly — **no need to extract features ourselves.**

**Download links (from `tianyu0207/RTFM` official repo):**

- UCF-Crime train I3D features (Google Drive): https://drive.google.com/file/d/1i2P9Nn62i0cVil_WS24HKzbzmyUxA9vX/view?usp=drive_link
- UCF-Crime test I3D features (Google Drive): https://drive.google.com/file/d/1KHBLG0-cjSmbqZJc4pjvKzRMI3jeE14T/view?usp=drive_link
- Alternative (OneDrive) versions also listed in their README if Drive link breaks:
  - train: https://uao365-my.sharepoint.com/:f:/g/personal/a1697106_adelaide_edu_au/ErCr6bjDzzZPstgposv1ttYBjv_ZBsAbNTbwyl3yX8QCHA?e=BzNuJ2
  - test: https://uao365-my.sharepoint.com/:f:/g/personal/a1697106_adelaide_edu_au/EsmBEpklrShEjTFOWTd5FooBkJR3DPxp3cIZN-R8b2hhLA?e=hlcZFO
- Pretrained checkpoint for UCF-Crime (if we want a sanity baseline without training):
  https://uao365-my.sharepoint.com/:u:/g/personal/a1697106_adelaide_edu_au/Ed0gS0RZ5hFMqVa8LxcO3sYBqFEmzMU5IsvvLWxioTatKw?e=qHEl5Z

**Note on feature origin:** these I3D features were extracted using the ResNet-50 I3D backbone
from `Tushar-N/pytorch-resnet3d` (https://github.com/Tushar-N/pytorch-resnet3d), with 10-crop
augmentation, following prior WS-VAD convention (Sultani et al., RTFM, etc.).

**Expected local structure after download:**
```
data/features/
└── ucf-crime/
    ├── i3d/
    │   ├── train/         # extracted train features, .npy per video
    │   └── test/           # extracted test features, .npy per video
```

**Action needed after download:** edit the file paths inside
`baselines/accuracy_baselines/rtfm2021/list/ucf-i3d-train-10crop.list` and
`ucf-i3d-test-10crop.list` (or equivalent list files) to point to wherever you
actually put the downloaded features locally. RTFM's code reads paths from these
`.list` files, not from a config — easy to miss, check this first if training crashes
with "file not found."

---

## 2. UCF-Crime — Precomputed C3D / ResNet Features (for Sultani 2018 baseline)

The `ekosman/AnomalyDetectionCVPR2018-Pytorch` repo (our Sultani 2018 baseline) provides
its own precomputed features separately — check that repo's README/releases for download
links once cloned into `baselines/accuracy_baselines/sultani2018/`. Don't assume RTFM's
I3D features are directly swappable here; Sultani's original method expects C3D or
ResNet-based features depending on which variant of the repo we use.

**TODO (whoever sets this up):** paste exact link here once confirmed from that repo.

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