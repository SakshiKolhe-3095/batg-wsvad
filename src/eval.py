"""
src/eval.py
BATG — Evaluation: AUC / FLOPs / FPS / Latency / Memory at each budget level

Loads a trained BATG checkpoint (scorer+gate+head) and evaluates it at each
budget level in config's `budget_levels` list, producing the core paper
results table:

    Budget B | AUC | GFLOPs | FPS | Latency (ms) | Peak Memory (MB)

IMPORTANT — FLOPs measurement nuance (read before trusting numbers):
    Training-time masking (see train.py) multiplies gated features by a
    soft/hard mask but keeps the tensor shape (B, T, feat_dim) unchanged —
    that's correct for gradient flow, but WRONG for measuring real FLOPs
    savings, since fvcore/ptflops will count compute over the full T
    regardless of zeros. At eval time this script actually SLICES down to
    only the kept T_kept segments before running the MIL head, so measured
    FLOPs reflect real compute reduction, not just zeroed-out waste.

SEGMENT-TO-FRAME ALIGNMENT (resolved, verified against actual dataset.py):
    dataset.py's UCFCrimeDataset._resample() picks n representative indices
    out of the original T via np.linspace — it does NOT average equal-width
    bins. RTFM's own eval (test_10crop.py) does a simple "repeat each score
    by 16" because RTFM has no resampling step. Since BATG resamples every
    video to a fixed n_segments=32, a blind repeat-by-16 would misalign
    scores whenever a video's original T isn't a clean fit. This script
    instead re-loads each video's original T (via test_ds.video_paths) and
    expands each of the 32 scores across its proportional share of T before
    applying the same repeat-by-16 (matching RTFM's frames-per-T-unit
    convention). See expand_segment_scores_to_frames() below.

ONE REMAINING ASSUMPTION (not yet verified — flag if AUC looks wrong):
    Video iteration order in test_ds must match the video order gt-ucf.npy
    was built from. This mirrors RTFM's own assumption (same gt file, same
    list-file convention) — if AUC looks implausible, check this first.

Usage:
    python -m src.eval --config configs/batg_base.yaml --dataset configs/dataset_ucf.yaml --checkpoint checkpoints/batg_epoch099.pt
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score

from src.data.dataset import build_dataloaders
from src.models.scoring import SegmentScorer
from src.models.batg_gate import BATGGate
from src.models.mil_head import MILHead
from src.utils.flops_counter import count_flops_fvcore, measure_fps
from src.utils.logger import CSVLogger, print_log


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class BATGPipeline(nn.Module):
    """Wraps scorer+gate+head into a single nn.Module so fvcore can trace
    the whole pipeline's FLOPs in one call. Fixed budget per instance since
    fvcore traces a single forward path.
    """

    def __init__(self, scorer, gate, head, budget: float):
        super().__init__()
        self.scorer = scorer
        self.gate = gate
        self.head = head
        self.budget = budget

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.scorer(x)
        mask, _, _ = self.gate(scores, budget=self.budget, training=False)
        # Hard mask at eval — actually slice to kept segments for real FLOPs
        # reduction (see module docstring). Assumes batch size 1 for tracing.
        keep_idx = mask[0].nonzero(as_tuple=True)[0]
        kept_feats = x[:, keep_idx, :]
        if kept_feats.shape[1] == 0:  # safety: budget too small, keep at least 1
            kept_feats = x[:, :1, :]
        return self.head(kept_feats)


def load_models_from_checkpoint(checkpoint_path: str, model_cfg: dict, device):
    scorer = SegmentScorer(
        feat_dim=model_cfg["feat_dim"],
        hidden_dim=model_cfg["scorer_hidden_dim"],
        learn_weights=model_cfg.get("scorer_learn_weights", False),
    ).to(device)
    gate = BATGGate(temperature=model_cfg.get("gate_temperature", 1.0)).to(device)
    head = MILHead(
        feat_dim=model_cfg["feat_dim"],
        dropout=model_cfg.get("mil_dropout", 0.7),
        k_ratio=model_cfg.get("mil_k_ratio", 0.2),
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    scorer.load_state_dict(state["model_state_dict"])
    gate.load_state_dict(state["gate_state_dict"])
    head.load_state_dict(state["head_state_dict"])
    print_log(f"Loaded checkpoint {checkpoint_path} (epoch {state.get('epoch', '?')})")

    scorer.eval()
    gate.eval()
    head.eval()
    return scorer, gate, head


def expand_segment_scores_to_frames(scores: np.ndarray,
                                     T_original: int,
                                     frames_per_segment: int = 16) -> np.ndarray:
    """Expand n resampled segment scores back to frame-level predictions.

    dataset.py's _resample() picks n representative indices out of the
    original T via np.linspace — it does NOT average equal-width bins.
    To expand back correctly (rather than RTFM's simple "repeat by 16",
    which only works when there's no resampling step), we split the
    original T into n contiguous chunks (via linspace boundaries) and
    repeat each segment's score across its chunk's frame span.

    frames_per_segment=16 matches RTFM's own convention (confirmed in
    test_10crop.py: pred = np.repeat(pred, 16)) — each original T-unit
    already corresponds to 16 raw video frames in the I3D feature pipeline.

    Args:
        scores: (n,) segment-level anomaly scores (n = n_segments, e.g. 32)
        T_original: original T (before resampling) for this specific video
        frames_per_segment: frames each original T-unit represents (16, RTFM convention)
    Returns:
        (T_original * frames_per_segment,) frame-level predictions
    """
    n = len(scores)
    boundaries = np.linspace(0, T_original, n + 1, dtype=int)
    frame_scores = []
    for i in range(n):
        span = max(boundaries[i + 1] - boundaries[i], 0) * frames_per_segment
        if span > 0:
            frame_scores.append(np.repeat(scores[i], span))
    if not frame_scores:
        return np.array([])
    return np.concatenate(frame_scores)


@torch.no_grad()
def run_inference_at_budget(scorer, gate, head, test_ds, budget: float, device):
    """Run inference over the full test set at a fixed budget.

    Returns frame-level predictions, correctly expanded from each video's
    resampled segment scores back to its original frame span (see
    expand_segment_scores_to_frames), concatenated across the whole test set
    in the same order as gt-ucf.npy expects (video order must match RTFM's
    original ordering — verify if AUC looks wrong).
    """
    all_frame_scores = []

    for idx in range(len(test_ds)):
        feats, label = test_ds[idx]
        feats = feats.unsqueeze(0).to(device)  # (1, n_segments, feat_dim)

        scores = scorer(feats)
        mask, kept_fraction, _ = gate(scores, budget=budget, training=False)
        gated_feats = feats * mask.unsqueeze(-1)
        anomaly_scores = head(gated_feats).squeeze(0).cpu().numpy()  # (n_segments,)

        # Get this video's ORIGINAL T (before resampling) to expand correctly
        video_path = test_ds.video_paths[idx]
        T_original = np.load(video_path).shape[0]

        frame_scores = expand_segment_scores_to_frames(anomaly_scores, T_original)
        all_frame_scores.append(frame_scores)

    return all_frame_scores


def compute_auc(all_frame_scores, gt_file: str) -> float:
    """Compute frame-level AUC against ground truth array.

    Uses RTFM's own trim convention (gt[:len(pred)]) as a final safety net
    for any residual off-by-a-few rounding from the linspace boundary split,
    but the bulk of the alignment is now handled correctly by
    expand_segment_scores_to_frames rather than a blind repeat-by-16.
    """
    gt = np.load(gt_file)
    pred = np.concatenate(all_frame_scores)
    gt_aligned = gt[:len(pred)]
    if len(gt_aligned) != len(pred):
        print_log(
            f"WARNING: length mismatch after trim — gt={len(gt_aligned)}, "
            f"pred={len(pred)}. Check video ordering matches gt-ucf.npy's expected order."
        )
        min_len = min(len(gt_aligned), len(pred))
        gt_aligned = gt_aligned[:min_len]
        pred = pred[:min_len]
    return roc_auc_score(gt_aligned, pred)


def evaluate(config_path: str, dataset_config_path: str, checkpoint_path: str):
    cfg = load_yaml(config_path)
    ds_cfg = load_yaml(dataset_config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_log(f"Using device: {device}")

    _, test_loader, test_ds = build_dataloaders(
        train_list=ds_cfg["paths"]["train_list"],
        test_list=ds_cfg["paths"]["test_list"],
        n_segments=cfg["model"]["n_segments"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        gt_file=ds_cfg["paths"]["gt_file"],
        mean_crop=cfg["model"]["mean_crop"],
    )

    scorer, gate, head = load_models_from_checkpoint(checkpoint_path, cfg["model"], device)

    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    logger = CSVLogger(results_dir, prefix="budget_sweep")

    n_segments = cfg["model"]["n_segments"]
    feat_dim = cfg["model"]["feat_dim"]
    example_input = torch.randn(1, n_segments, feat_dim).to(device)

    results = []
    for budget in cfg["budget_levels"]:
        print_log(f"--- Evaluating budget={budget} ---")

        all_frame_scores = run_inference_at_budget(
            scorer, gate, head, test_ds, budget, device
        )
        auc = compute_auc(all_frame_scores, ds_cfg["paths"]["gt_file"])

        pipeline = BATGPipeline(scorer, gate, head, budget=budget)
        flops_result = count_flops_fvcore(pipeline, example_input)
        fps_result = measure_fps(pipeline, example_input, n_runs=50, device=str(device))

        peak_mem_mb = None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            _ = pipeline(example_input)
            peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6

        row = {
            "budget": budget,
            "auc": auc,
            "gflops": flops_result["flops_G"],
            "params_M": flops_result["params_M"],
            "fps": fps_result["fps"],
            "latency_ms_mean": fps_result["latency_ms_mean"],
            "latency_ms_std": fps_result["latency_ms_std"],
            "peak_mem_mb": peak_mem_mb,
        }
        results.append(row)
        logger.log(**row)
        print_log(
            f"Budget {budget}: AUC={auc:.4f}  GFLOPs={row['gflops']:.4f}  "
            f"FPS={row['fps']:.1f}  Latency={row['latency_ms_mean']:.2f}ms  "
            f"PeakMem={peak_mem_mb}"
        )

    print_log("Budget sweep complete.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/batg_base.yaml")
    parser.add_argument("--dataset", type=str, default="configs/dataset_ucf.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    evaluate(args.config, args.dataset, args.checkpoint)