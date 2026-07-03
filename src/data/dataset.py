"""
UCF-Crime dataset loader for BATG-WSVAD.

Feature format (confirmed from Mumtaj's merged files):
  - Files: {VideoName}_i3d.npy  e.g. Abuse001_x264_i3d.npy
  - Shape per file: (T, 10, 1024)
      T   = number of temporal segments (varies: min 18, max 1779, mean ~243)
      10  = number of crops (10-crop augmentation)
      1024 = I3D feature dimension (DeepMIL backbone)
  - Train: 1610 videos (abnormal-first split, index 0-809 abnormal, 810-1609 normal)
  - Test:  290 videos

List file format (RTFM-style, fixed by Mumtaj):
  Each line: /path/to/VideoName_i3d.npy  num_segments  label
  label: 1 = anomaly video, 0 = normal video

Usage:
    from src.data.dataset import UCFCrimeDataset
    train_ds = UCFCrimeDataset(list_file="path/to/ucf-i3d.list", n_segments=32, is_train=True)
    test_ds  = UCFCrimeDataset(list_file="path/to/ucf-i3d-test.list", n_segments=32, is_train=False)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class UCFCrimeDataset(Dataset):
    """UCF-Crime weakly supervised VAD dataset.
    
    Loads precomputed I3D features (T, 10, 1024) and resamples
    to a fixed number of segments for batch consistency.
    
    Args:
        list_file:   path to .list file (one entry per video)
        n_segments:  number of segments to resample each video to (default 32, RTFM convention)
        is_train:    True for training (returns video-level label),
                     False for testing (returns frame-level gt if available)
        gt_file:     path to ground-truth .npy for test set (optional, for AUC eval)
        mean_crop:   if True, average 10 crops → shape (T, 1024); if False keep all crops (T, 10, 1024)
    """

    def __init__(self,
                 list_file: str,
                 n_segments: int = 32,
                 is_train: bool = True,
                 gt_file: str = None,
                 mean_crop: bool = True) -> None:

        self.n_segments = n_segments
        self.is_train   = is_train
        self.mean_crop  = mean_crop

        self.video_paths = []
        self.labels      = []   # video-level: 1=anomaly, 0=normal

        self._load_list(list_file)

        # frame-level ground truth for test AUC (optional)
        self.gt = None
        if gt_file is not None and os.path.exists(gt_file):
            self.gt = np.load(gt_file)

    def _load_list(self, list_file: str) -> None:
        """Parse RTFM-style list file.
        
        Expected format per line (space-separated):
            /path/to/Video_i3d.npy  <num_segments>  <label>
        OR (2-column, label inferred from filename position):
            /path/to/Video_i3d.npy  <num_segments>
        """
        if not os.path.exists(list_file):
            raise FileNotFoundError(f"List file not found: {list_file}")

        with open(list_file, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        for line in lines:
            parts = line.split()
            path = parts[0]

            if len(parts) >= 3:
                label = int(parts[2])
            else:
                # fallback: infer from filename — Normal_Videos = 0, else = 1
                label = 0 if "Normal_Videos" in os.path.basename(path) else 1

            self.video_paths.append(path)
            self.labels.append(label)

    def _load_features(self, path: str) -> np.ndarray:
        """Load .npy feature file and return array of shape (T, 1024) or (T, 10, 1024)."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Feature file not found: {path}")
        return np.load(path)   # shape: (T, 10, 1024)

    def _resample(self, features: np.ndarray, n: int) -> np.ndarray:
        """Uniformly resample T segments to exactly n segments.
        
        Handles both short videos (T < n, upsample) and
        long videos (T > n, downsample) without dropping content.
        """
        T = features.shape[0]
        if T == n:
            return features
        indices = np.linspace(0, T - 1, n, dtype=int)
        return features[indices]

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int):
        path  = self.video_paths[idx]
        label = self.labels[idx]

        # load: (T, 10, 1024)
        feats = self._load_features(path)

        # resample to fixed n_segments: (n_segments, 10, 1024)
        feats = self._resample(feats, self.n_segments)

        # optionally average crops: (n_segments, 1024)
        if self.mean_crop:
            feats = feats.mean(axis=1)

        feats = torch.tensor(feats, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return feats, label

    def get_labels(self) -> list:
        """Return all video-level labels (for MIL bag construction)."""
        return self.labels


def build_dataloaders(train_list: str,
                      test_list: str,
                      n_segments: int = 32,
                      batch_size: int = 32,
                      num_workers: int = 0,
                      gt_file: str = None,
                      mean_crop: bool = True):
    """Convenience function to build train + test DataLoaders.
    
    Args:
        train_list:   path to train .list file
        test_list:    path to test .list file
        n_segments:   segments per video (default 32)
        batch_size:   training batch size (default 32 = 16 abnormal + 16 normal, RTFM convention)
        num_workers:  DataLoader workers (keep 0 on Windows to avoid multiprocessing issues)
        gt_file:      test ground-truth .npy for AUC eval
        mean_crop:    average 10 crops or keep all
    
    Returns:
        (train_loader, test_loader, test_dataset)
    """
    from torch.utils.data import DataLoader

    train_ds = UCFCrimeDataset(train_list, n_segments=n_segments,
                                is_train=True, mean_crop=mean_crop)
    test_ds  = UCFCrimeDataset(test_list,  n_segments=n_segments,
                                is_train=False, gt_file=gt_file, mean_crop=mean_crop)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,   # False on Windows / CPU-only machines
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,       # test one video at a time (variable-length context)
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    print(f"[dataset] Train: {len(train_ds)} videos | Test: {len(test_ds)} videos")
    print(f"[dataset] n_segments={n_segments} | feature_dim=1024 | mean_crop={mean_crop}")
    return train_loader, test_loader, test_ds


if __name__ == "__main__":
    """Smoke test — runs without actual data files, just checks imports and logic."""
    import tempfile, os

    # create a fake .npy feature file
    tmp_dir  = tempfile.mkdtemp()
    fake_npy = os.path.join(tmp_dir, "Abuse001_x264_i3d.npy")
    np.save(fake_npy, np.random.rand(170, 10, 1024).astype(np.float32))

    # create a fake list file
    list_path = os.path.join(tmp_dir, "fake_train.list")
    with open(list_path, "w") as f:
        f.write(f"{fake_npy} 170 1\n")

    ds = UCFCrimeDataset(list_path, n_segments=32, is_train=True, mean_crop=True)
    feats, label = ds[0]

    assert feats.shape == (32, 1024), f"Expected (32, 1024), got {feats.shape}"
    assert label.item() == 1.0
    print(f"Smoke test passed. feats.shape={feats.shape}, label={label.item()}")

    # cleanup
    os.remove(fake_npy)
    os.remove(list_path)
    os.rmdir(tmp_dir)
    print("Temp files cleaned up.")