"""
Non-learned floor baseline for BATG.
Uses ONLY raw motion energy (L2 norm of I3D features) as the anomaly score —
no trained scorer, no trained gate, no trained MIL head. Gives 62.7%-type
numbers a floor for context (per external critique's Red Flag #4).
"""
import torch
import numpy as np
import yaml
from src.data.dataset import build_dataloaders
from src.eval import expand_segment_scores_to_frames, compute_auc

cfg = yaml.safe_load(open("configs/batg_base.yaml"))
ds_cfg = yaml.safe_load(open("configs/dataset_ucf.yaml"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader, test_ds = build_dataloaders(
    train_list=ds_cfg["paths"]["train_list"],
    test_list=ds_cfg["paths"]["test_list"],
    n_segments=cfg["model"]["n_segments"],
    batch_size=1,
    num_workers=0,
    gt_file=ds_cfg["paths"]["gt_file"],
    mean_crop=cfg["model"]["mean_crop"],
)

all_frame_scores = []
for idx in range(len(test_ds)):
    feats, label = test_ds[idx]
    feats = feats.to(device)  # (n_segments, feat_dim)

    # Floor baseline: raw motion energy (L2 norm), min-max normalized per video
    energy = torch.norm(feats, p=2, dim=-1)  # (n_segments,)
    mn, mx = energy.min(), energy.max()
    anomaly_scores = ((energy - mn) / (mx - mn + 1e-8)).cpu().numpy()

    video_path = test_ds.video_paths[idx]
    T_original = np.load(video_path).shape[0]
    frame_scores = expand_segment_scores_to_frames(anomaly_scores, T_original)
    all_frame_scores.append(frame_scores)

auc = compute_auc(all_frame_scores, ds_cfg["paths"]["gt_file"])
print(f"Non-learned floor baseline (motion-energy-only) AUC: {auc:.4f}")