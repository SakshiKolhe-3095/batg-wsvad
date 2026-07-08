"""
src/budget_sweep.py
BATG — Budget Sweep + Core Paper Figure (FLOPs-AUC tradeoff curve)

Runs src/eval.py's evaluate() across all budget levels for a trained BATG
checkpoint, then overlays the fixed-point baselines (both accuracy-tier and
efficiency-tier, per the two-tier baseline split — see docs/related_work_notes.md)
to produce the paper's central figure: BATG's controllable curve vs. every
other method's single fixed operating point.

This is the "core paper experiment" per BATG_Research_Proposal.docx Section 8.

Baseline numbers below are hardcoded from:
  - Accuracy tier: handover Section 2 (Sultani/RTFM/BN-WVAD published numbers)
  - Efficiency tier: baselines/efficiency_baselines/*/numbers_only.md
    (Wang 2023, Karim 2024, Mohamad 2026 — verified via web search, see files)
    Zhang 2025/DE-Net: not yet run, placeholder — update once Mumtaj
    reproduces it (see handover KI-4, version caveat also applies)

VERIFY BEFORE TRUSTING THE FIGURE:
  - BATG's own numbers come from a REAL eval.py run against a REAL checkpoint
    — nothing here is invented. If eval hasn't been run yet, this script
    will fail loudly (by design) rather than plot placeholder BATG points.
  - Baseline numbers are as-published by each paper's authors, on UCF-Crime,
    frame-level AUC — not independently re-verified end-to-end by us except
    where explicitly noted (RTFM: independently attempted, see docs).

Usage:
    python -m src.budget_sweep --config configs/batg_base.yaml --dataset configs/dataset_ucf.yaml --checkpoint checkpoints/batg_epoch099.pt
"""

import argparse
import os

import matplotlib.pyplot as plt

from src.eval import evaluate
from src.utils.logger import print_log


# --- Fixed-point baselines (hardcoded, sourced as documented above) ---

ACCURACY_BASELINES = {
    # name: (AUC_percent, note)
    "Sultani 2018":  (75.4, "C3D features, published"),
    "RTFM 2021":     (84.3, "I3D 2048-dim RGB+Flow, published"),
    "BN-WVAD":       (87.24, "I3D, published"),
}

EFFICIENCY_BASELINES = {
    # name: (AUC_percent, GFLOPs_or_None, note)
    "Wang 2023":     (84.7, None, "0.14M params — efficiency axis is param count, not FLOPs"),
    "Karim 2024":    (86.94, None, "6.4s decision period — efficiency axis is latency, not FLOPs"),
    "Mohamad 2026":  (91.61, 0.71, "7.9M params — VERIFY AUC protocol before final use, see numbers_only.md"),
    "DE-Net (Zhang 2025)": (None, None, "not yet reproduced locally — see handover KI-4 version caveat"),
}


def plot_tradeoff_curve(batg_results: list, output_path: str):
    """Generate the core FLOPs-AUC tradeoff curve figure.

    BATG's own points are plotted as a connected line (the "curve" — the
    whole point of the paper). Baselines are plotted as scattered single
    points, since none of them expose a controllable budget.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # BATG curve — real measured points from eval.py
    budgets = [r["budget"] * 100 for r in batg_results]
    aucs = [r["auc"] * 100 for r in batg_results]
    ax.plot(budgets, aucs, marker="o", linewidth=2, label="BATG (ours)", color="tab:blue")
    for r in batg_results:
        ax.annotate(f"{r['gflops']:.2f} GFLOPs",
                    (r["budget"] * 100, r["auc"] * 100),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    # Accuracy-tier baselines — plotted at B=100% for reference (they have no
    # budget concept, this just anchors them on the same AUC axis)
    for name, (auc, note) in ACCURACY_BASELINES.items():
        ax.scatter([100], [auc], marker="s", s=80, label=f"{name} (accuracy tier)")

    # Efficiency-tier baselines — plotted at their single fixed operating
    # point; x-position uses GFLOPs if available, else just annotated at
    # right edge with a note (since their "budget" isn't directly comparable)
    for name, (auc, gflops, note) in EFFICIENCY_BASELINES.items():
        if auc is None:
            continue  # not yet reproduced, skip plotting
        x_pos = 100  # placeholder x since these aren't on the same budget axis
        ax.scatter([x_pos], [auc], marker="^", s=80, label=f"{name} (efficiency tier)")

    ax.set_xlabel("Budget B (%) / Compute Level")
    ax.set_ylabel("AUC (%)")
    ax.set_title("BATG FLOPs-AUC Tradeoff Curve vs. Fixed-Point Baselines")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print_log(f"Figure saved to {output_path}")
    plt.close(fig)


def run_budget_sweep(config_path: str, dataset_config_path: str, checkpoint_path: str):
    print_log("Running BATG evaluation across all budget levels...")
    batg_results = evaluate(config_path, dataset_config_path, checkpoint_path)

    print_log("\n=== Core Experiment Table ===")
    print_log(f"{'Budget':>8} | {'AUC':>8} | {'GFLOPs':>8} | {'FPS':>8} | {'Latency(ms)':>12}")
    for r in batg_results:
        print_log(
            f"{r['budget']*100:>7.0f}% | {r['auc']*100:>7.2f}% | "
            f"{r['gflops']:>8.4f} | {r['fps']:>8.1f} | {r['latency_ms_mean']:>12.2f}"
        )

    print_log("\n=== Accuracy-Tier Baselines (fixed point, no budget concept) ===")
    for name, (auc, note) in ACCURACY_BASELINES.items():
        print_log(f"{name}: AUC={auc}% ({note})")

    print_log("\n=== Efficiency-Tier Baselines (fixed operating point) ===")
    for name, (auc, gflops, note) in EFFICIENCY_BASELINES.items():
        auc_str = f"{auc}%" if auc is not None else "N/A"
        print_log(f"{name}: AUC={auc_str}, GFLOPs={gflops} ({note})")

    figure_path = "results/figures/batg_tradeoff_curve.png"
    plot_tradeoff_curve(batg_results, figure_path)

    return batg_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/batg_base.yaml")
    parser.add_argument("--dataset", type=str, default="configs/dataset_ucf.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    run_budget_sweep(args.config, args.dataset, args.checkpoint)