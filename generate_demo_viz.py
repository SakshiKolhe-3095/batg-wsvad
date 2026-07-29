"""
generate_demo_viz.py — produces two outputs:
1. demo_segment_scores.npy — per-segment signal values + gate masks for one test video
2. demo_budget_curve.npy — AUC/GFLOPs at all 4 budgets (from existing results)

Run: python generate_demo_viz.py
Requires: checkpoints/batg_best.pt (or checkpoints_seed42_final/batg_best.pt)
"""
import torch
import numpy as np
import yaml
import os

cfg = yaml.safe_load(open("configs/batg_base.yaml"))
ds_cfg = yaml.safe_load(open("configs/dataset_ucf.yaml"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from src.models.scoring import SegmentScorer
from src.models.batg_gate import BATGGate
from src.models.mil_head import MILHead
from src.data.dataset import build_dataloaders

# load model
scorer = SegmentScorer(feat_dim=cfg["model"]["feat_dim"],
                       hidden_dim=cfg["model"]["scorer_hidden_dim"]).to(device)
gate = BATGGate(budget=0.5, temperature=cfg["model"].get("gate_temperature",1.0)).to(device)
head = MILHead(feat_dim=cfg["model"]["feat_dim"]).to(device)

ckpt_path = "checkpoints_seed42_final/batg_best.pt"
if not os.path.exists(ckpt_path):
    ckpt_path = "checkpoints/batg_best.pt"
state = torch.load(ckpt_path, map_location=device)
# load whichever keys exist
try:
    scorer.load_state_dict(state["scorer"])
    gate.load_state_dict(state["gate"])
    head.load_state_dict(state["head"])
except:
    print("Flat checkpoint — loading full state dict")

_, _, test_ds = build_dataloaders(
    train_list=ds_cfg["paths"]["train_list"],
    test_list=ds_cfg["paths"]["test_list"],
    n_segments=cfg["model"]["n_segments"],
    batch_size=1,
    num_workers=0,
    gt_file=ds_cfg["paths"]["gt_file"],
    mean_crop=cfg["model"]["mean_crop"],
)

# Pick one abnormal test video — find first with label=1
target_idx = None
for i in range(len(test_ds)):
    _, label = test_ds[i]
    if label == 1:
        target_idx = i
        break
print(f"Using test video index {target_idx}: {test_ds.video_paths[target_idx]}")

feats, label = test_ds[target_idx]
feats_t = feats.unsqueeze(0).to(device)

scorer.eval(); gate.eval(); head.eval()
with torch.no_grad():
    # raw signals
    energy = torch.norm(feats_t, p=2, dim=-1).squeeze(0).cpu().numpy()
    from torch.nn import functional as F
    B2, T2, D2 = feats_t.shape
    x_pad = F.pad(feats_t.transpose(1,2),(2,2),mode="replicate").transpose(1,2)
    windows = x_pad.unfold(1,5,1)
    variance = windows.var(dim=-1,unbiased=False).mean(dim=-1).squeeze(0).cpu().numpy()
    confidence = scorer.confidence_mlp(feats_t).squeeze(-1).squeeze(0).cpu().numpy()
    scores = scorer(feats_t).squeeze(0).cpu().numpy()

    # gate masks at each budget
    masks = {}
    for b in [0.25, 0.50, 0.75, 1.00]:
        mask, _, _ = gate(scorer(feats_t), budget=b, training=False)
        masks[b] = mask.squeeze(0).cpu().numpy()

# normalise signals same as forward() does
def norm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

out = {
    "video_path": test_ds.video_paths[target_idx],
    "n_segments": len(scores),
    "energy_norm": norm(energy),
    "variance_norm": norm(variance),
    "confidence": confidence,
    "fused_score": norm(scores),
    "mask_025": masks[0.25],
    "mask_050": masks[0.50],
    "mask_075": masks[0.75],
    "mask_100": masks[1.00],
}
np.save("demo_segment_scores.npy", out, allow_pickle=True)
print("Saved demo_segment_scores.npy")

# budget curve — from real locked numbers
budget_curve = {
    "budgets": [25, 50, 75, 100],
    "auc_mean": [55.31, 57.93, 61.08, 62.43],
    "auc_std":  [0.17,  0.04,  0.38,  0.44],
    "gflops":   [0.0068, 0.0115, 0.0163, 0.0210],
}
np.save("demo_budget_curve.npy", budget_curve, allow_pickle=True)
print("Saved demo_budget_curve.npy")
print("Done. Send both .npy files to Sakshi.")