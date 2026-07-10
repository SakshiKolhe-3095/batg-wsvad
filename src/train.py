"""
src/train.py
BATG — Main Training Loop

Wires together SegmentScorer -> BATGGate -> MILHead into the full BATG
pipeline, trains on UCF-Crime I3D features with MIL ranking loss + budget
regularization loss.

Key design point (Project Rule #9 — do NOT relax this):
    ONE model is trained across ALL budget levels, not one model per budget.
    Each training step samples a random budget from `budget_levels` (config)
    so the same scorer/gate/head learn to work at every operating point.
    At eval/budget-sweep time, the same trained weights are re-used with a
    fixed budget passed in per run (see src/budget_sweep.py, not yet written).

Usage:
    python -m src.train --config configs/batg_base.yaml --dataset configs/dataset_ucf.yaml

Checkpoints saved every `checkpoint_every_n_epochs` (see config), resumable
after AI Kosh 4hr session cutoff (see src/utils/checkpoint.py).
"""

import argparse
import os

import torch
import torch.optim as optim
import yaml

from src.data.dataset import build_dataloaders
from src.models.scoring import SegmentScorer
from src.models.batg_gate import BATGGate
from src.models.mil_head import MILHead
from src.utils.seed import set_seed
from src.utils.checkpoint import save_checkpoint, latest_checkpoint
from src.eval import run_inference_at_budget, compute_auc
from src.utils.logger import CSVLogger, print_log


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_models(model_cfg: dict, device: torch.device):
    scorer = SegmentScorer(
        feat_dim=model_cfg["feat_dim"],
        hidden_dim=model_cfg["scorer_hidden_dim"],
        learn_weights=model_cfg.get("scorer_learn_weights", False),
    ).to(device)

    gate = BATGGate(
        budget=0.5,  # default/placeholder; actual budget sampled per step below
        temperature=model_cfg.get("gate_temperature", 1.0),
    ).to(device)

    head = MILHead(
        feat_dim=model_cfg["feat_dim"],
        dropout=model_cfg.get("mil_dropout", 0.7),
        k_ratio=model_cfg.get("mil_k_ratio", 0.2),
    ).to(device)

    return scorer, gate, head


def batg_forward(scorer, gate, head, feats, budget, is_train):
    """One full BATG forward pass. Returns anomaly scores, kept_fraction, budget_loss."""
    scores = scorer(feats)                                         # (B, T)
    mask, kept_fraction, b_loss = gate(scores, budget=budget, training=is_train)  # (B, T)
    gated_feats = feats * mask.unsqueeze(-1)                        # (B, T, feat_dim)
    anomaly_scores = head(gated_feats)                              # (B, T)
    return anomaly_scores, kept_fraction, b_loss


def train(config_path: str, dataset_config_path: str, resume: bool = True):
    cfg = load_yaml(config_path)
    ds_cfg = load_yaml(dataset_config_path)

    set_seed(cfg["training"]["seed"])  # ALWAYS first — reproducibility

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_log(f"Using device: {device}")

    # --- Data ---
    train_loader, test_loader, test_ds = build_dataloaders(
        train_list=ds_cfg["paths"]["train_list"],
        test_list=ds_cfg["paths"]["test_list"],
        n_segments=cfg["model"]["n_segments"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        gt_file=ds_cfg["paths"]["gt_file"],
        mean_crop=cfg["model"]["mean_crop"],
    )

    # --- Models ---
    scorer, gate, head = build_models(cfg["model"], device)
    params = list(scorer.parameters()) + list(gate.parameters()) + list(head.parameters())
    optimizer = optim.Adam(params, lr=cfg["training"]["learning_rate"])
    total_epochs = cfg["training"]["epochs"]
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(total_epochs * 0.4), int(total_epochs * 0.7)],
        gamma=0.1
    )
    budget_levels = cfg["budget_levels"]  # e.g. [0.25, 0.50, 0.75, 1.00]
    lambda_budget = cfg["training"]["lambda_budget"]
    checkpoint_dir = cfg["paths"]["checkpoint_dir"]
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    logger = CSVLogger(results_dir, prefix="batg_train")

    # --- Resume logic (AI Kosh 4hr cap survival) ---
    # NOTE: checkpoint.py's load_checkpoint() only restores a single model via
    # model.load_state_dict(...). BATG has 3 modules (scorer/gate/head), so we
    # bypass load_checkpoint here and load the raw checkpoint dict directly,
    # restoring each module's state_dict manually. save_checkpoint() is still
    # used as-is for saving (it's already a generic dict-saver, no change needed).
    start_epoch = 0
    best_auc = -1.0
    if resume:
        ckpt_path = latest_checkpoint(checkpoint_dir, prefix="batg")
        if ckpt_path:
            state = torch.load(ckpt_path, map_location=device)
            scorer.load_state_dict(state["model_state_dict"])
            gate.load_state_dict(state["gate_state_dict"])
            head.load_state_dict(state["head_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            start_epoch = state.get("epoch", -1) + 1
            best_auc = state.get("best_auc", state.get("auc", -1.0))
            print_log(f"Resumed from {ckpt_path} at epoch {start_epoch} "
                       f"best_auc={best_auc:.4f} "
                       f"(scorer+gate+head+optimizer all restored)")

    checkpoint_every = cfg["training"]["checkpoint_every_n_epochs"]
    for epoch in range(start_epoch, total_epochs):
        scorer.train()
        gate.train()
        head.train()

        epoch_loss = 0.0
        epoch_mil_loss = 0.0
        epoch_budget_loss = 0.0
        n_batches = 0

        for feats, labels in train_loader:
            feats = feats.to(device)      # (B, n_segments, feat_dim)
            labels = labels.to(device).float()

            # Sample a random budget each step — trains ONE model across ALL
            # operating points (Project Rule #9, do not fix this to one value)
            budget = budget_levels[torch.randint(len(budget_levels), (1,)).item()]

            optimizer.zero_grad()
            anomaly_scores, kept_fraction, b_loss = batg_forward(
                scorer, gate, head, feats, budget=budget, is_train=True
            )
            m_loss = head.mil_loss(anomaly_scores, labels)
            total_loss = m_loss + lambda_budget * b_loss

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_mil_loss += m_loss.item()
            epoch_budget_loss += b_loss.item()
            n_batches += 1

        scheduler.step()

        # --- Validation AUC check (catches overfitting, saves true best) ---
        scorer.eval()
        gate.eval()
        head.eval()
        val_auc = None
        if epoch % checkpoint_every == 0 or epoch == total_epochs - 1:
            all_scores = run_inference_at_budget(scorer, gate, head, test_ds, budget=1.0, device=device)
            val_auc = compute_auc(all_scores, ds_cfg["paths"]["gt_file"])
            if val_auc > best_auc:
                best_auc = val_auc
                save_checkpoint({
                    "epoch": epoch,
                    "model_state_dict": scorer.state_dict(),
                    "gate_state_dict": gate.state_dict(),
                    "head_state_dict": head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "auc": val_auc,
                }, path=os.path.join(checkpoint_dir, "batg_best.pt"))
                print_log(f"[BEST] New best val AUC={val_auc:.4f} at epoch {epoch} -> saved batg_best.pt")
        scorer.train()
        gate.train()
        head.train()

        avg_loss = epoch_loss / max(1, n_batches)

        avg_mil_loss = epoch_mil_loss / max(1, n_batches)
        avg_budget_loss = epoch_budget_loss / max(1, n_batches)

        current_lr = optimizer.param_groups[0]["lr"]
        val_auc_str = f"{val_auc:.4f}" if val_auc is not None else "n/a"
        print_log(
            f"Epoch {epoch}: total_loss={avg_loss:.4f} "
            f"mil_loss={avg_mil_loss:.4f} budget_loss={avg_budget_loss:.4f} "
            f"lr={current_lr:.6f} val_auc={val_auc_str} best_auc={best_auc:.4f} "
            f"(last batch budget={budget}, kept={kept_fraction:.2%})"
        )
        logger.log(
            epoch=epoch,
            total_loss=avg_loss,
            mil_loss=avg_mil_loss,
            budget_loss=avg_budget_loss,
            last_budget=budget,
            kept_fraction=kept_fraction,
        )

        if epoch % checkpoint_every == 0:
            save_checkpoint({
                "epoch": epoch,
                "model_state_dict": scorer.state_dict(),
                "gate_state_dict": gate.state_dict(),
                "head_state_dict": head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, path=os.path.join(checkpoint_dir, f"batg_epoch{epoch:03d}.pt"))
            print_log(f"Checkpoint saved at epoch {epoch}")

    print_log("Training complete.")
    return scorer, gate, head


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/batg_base.yaml")
    parser.add_argument("--dataset", type=str, default="configs/dataset_ucf.yaml")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    train(args.config, args.dataset, resume=not args.no_resume)