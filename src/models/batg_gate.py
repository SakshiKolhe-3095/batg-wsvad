"""
src/models/batg_gate.py
BATG — Budget-Aware Temporal Gating Module

Core novelty of the paper. Takes per-segment importance scores from SegmentScorer
and produces a binary keep/drop mask calibrated to a user-specified compute budget B.

Budget B = fraction of segments to keep, e.g.:
  B=0.25 → keep top 25% of segments (cheapest setting)
  B=0.50 → keep top 50%
  B=0.75 → keep top 75%
  B=1.00 → keep all segments (full compute, matches unmodified baseline)

Training uses a soft (differentiable) approximation of the hard mask
so gradients flow through. Inference uses the hard binary mask.

Input:  scores (B, T)     — from SegmentScorer
        budget (float)    — target fraction of segments to keep, in (0, 1]
Output: mask (B, T)       — 1.0=keep, 0.0=drop (hard at inference, soft at train)
        kept_fraction     — actual fraction kept (should ≈ budget)

Budget regularization loss (add to MIL loss during training):
  budget_loss = MSE(mask.mean(dim=1), target_budget)
  This penalizes the model for deviating from the requested compute level.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BATGGate(nn.Module):
    """Budget-Aware Temporal Gating Module.
    
    Args:
        budget:         default budget fraction (can be overridden at forward time)
        temperature:    softmax temperature for soft mask during training
                        lower T = sharper (closer to hard), higher T = smoother
                        start with 1.0, anneal toward 0.1 if needed
        hard_threshold: if True, always use hard binary mask (inference mode)
    """

    def __init__(self,
                 budget: float = 0.5,
                 temperature: float = 1.0,
                 hard_threshold: bool = False) -> None:
        super().__init__()

        assert 0 < budget <= 1.0, "Budget must be in (0, 1]"
        self.budget          = budget
        self.temperature     = temperature
        self.hard_threshold  = hard_threshold

    def _hard_mask(self, scores: torch.Tensor, budget: float) -> torch.Tensor:
        """Binary keep/drop mask: keep top (budget × T) segments by score.
        
        Args:
            scores: (B, T)
            budget: float in (0, 1]
        Returns:
            mask: (B, T) — binary, dtype float
        """
        # TODO (Mumtaj): implement
        # Steps:
        #   1. k = max(1, int(T * budget))   — number of segments to keep
        #   2. find the k-th largest score per video (torch.topk)
        #   3. threshold = kth score value
        #   4. mask = (scores >= threshold).float()
        # Edge case: if budget=1.0, return all-ones mask
        B, T = scores.shape
        if budget >= 1.0:
            return torch.ones_like(scores)
        k = max(1, int(T * budget))
        topk_vals, _ = torch.topk(scores, k, dim=1)
        threshold = topk_vals[:, -1].unsqueeze(1)
        mask = (scores >= threshold).float()
        return mask

    def _soft_mask(self, scores: torch.Tensor, budget: float) -> torch.Tensor:
        """Soft (differentiable) approximation of hard mask for training.
        
        Uses a scaled sigmoid centered on the budget-quantile threshold.
        This allows gradients to flow through the gating decision.
        
        Args:
            scores: (B, T)
            budget: float in (0, 1]
        Returns:
            soft_mask: (B, T) — values in (0, 1), not binary
        """
        # TODO (Mumtaj): implement
        # Approach:
        #   1. Find threshold = quantile of scores at (1 - budget) level
        #      e.g. if budget=0.5, threshold = median score per video
        #      torch.quantile(scores, 1 - budget, dim=1, keepdim=True)
        #   2. soft_mask = torch.sigmoid((scores - threshold) / self.temperature)
        # This gives ~1.0 for segments well above threshold, ~0.0 for well below
        if budget >= 1.0:
            return torch.ones_like(scores)
        threshold = torch.quantile(scores, 1.0 - budget, dim=1, keepdim=True)
        soft_mask = torch.sigmoid((scores - threshold) / self.temperature)
        return soft_mask

    def budget_loss(self, mask: torch.Tensor, target_budget: float) -> torch.Tensor:
        """Budget regularization loss.
        
        Penalizes deviation between actual kept fraction and target budget.
        Add this to MIL loss: total_loss = mil_loss + lambda_budget * budget_loss
        
        Args:
            mask:          (B, T) — soft or hard mask
            target_budget: float — desired fraction of segments to keep
        Returns:
            scalar loss (MSE between actual and target keep-fraction)
        """
        # TODO (Mumtaj): implement
        # actual_fraction = mask.mean(dim=1)        # (B,) — per-video keep fraction
        # target = torch.full_like(actual_fraction, target_budget)
        # return F.mse_loss(actual_fraction, target)
        actual_fraction = mask.mean(dim=1)
        target = torch.full_like(actual_fraction, target_budget)
        return F.mse_loss(actual_fraction, target)

    def forward(self,
                scores: torch.Tensor,
                budget: float = None,
                training: bool = None) -> tuple:
        """Apply gating to segment scores.
        
        Args:
            scores:   (B, T) — importance scores from SegmentScorer
            budget:   override budget for this forward pass (uses self.budget if None)
            training: override train/eval mode (uses self.training if None)
        Returns:
            mask:           (B, T) — keep/drop mask
            kept_fraction:  float — actual fraction kept (≈ budget)
            b_loss:         scalar — budget regularization loss (use during training)
        """
        # TODO (Mumtaj): implement
        # Steps:
        #   1. b = budget if budget is not None else self.budget
        #   2. is_train = training if training is not None else self.training
        #   3. if is_train: mask = self._soft_mask(scores, b)
        #      else:        mask = self._hard_mask(scores, b)
        #   4. b_loss = self.budget_loss(mask, b)
        #   5. kept_fraction = mask.mean().item()
        #   6. return mask, kept_fraction, b_loss
        b = budget if budget is not None else self.budget
        is_train = training if training is not None else self.training

        if is_train and not self.hard_threshold:
            mask = self._soft_mask(scores, b)
        else:
            mask = self._hard_mask(scores, b)

        b_loss = self.budget_loss(mask, b)
        kept_fraction = mask.mean().item()
        return mask, kept_fraction, b_loss


if __name__ == "__main__":
    B, T = 4, 32
    scores = torch.rand(B, T)
    gate = BATGGate(budget=0.5, temperature=1.0)
    print(f"BATGGate created.")
    try:
        mask, kept_frac, b_loss = gate(scores, budget=0.5)
        print(f"mask shape: {mask.shape}, kept: {kept_frac:.2%}, budget_loss: {b_loss:.4f}")
        print("Smoke test passed.")
    except NotImplementedError as e:
        print(f"[expected] {e} — implement TODOs above.")