"""
src/models/scoring.py
BATG — Cheap Segment Importance Scoring Module

Computes a lightweight importance score for each temporal segment
BEFORE the expensive backbone/MIL head runs on it.
Three complementary signals, all computable from raw features cheaply:

  1. Motion energy    — L2 norm of feature vector (proxy for activity level)
  2. Temporal variance — variance across neighboring segments (local change rate)
  3. Confidence score — tiny learned MLP (1024 → 64 → 1)

Combined into a single per-segment importance score via weighted sum.
Weights (alpha, beta, gamma) are either fixed or learned — see TODO below.

Input:  (B, T, 1024)  — batch of B videos, T segments, 1024-dim I3D features
Output: (B, T)        — importance score per segment, higher = more important
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentScorer(nn.Module):
    """Lightweight segment importance scorer.
    
    Args:
        feat_dim:    input feature dimension (default 1024, DeepMIL I3D)
        hidden_dim:  hidden size of confidence MLP (default 64, keep small)
        alpha:       weight for motion energy signal (default 1.0)
        beta:        weight for temporal variance signal (default 1.0)
        gamma:       weight for learned confidence signal (default 1.0)
        learn_weights: if True, alpha/beta/gamma are learnable parameters
                       if False, they are fixed scalars (simpler, fewer params)
    """

    def __init__(self,
                 feat_dim: int = 1024,
                 hidden_dim: int = 64,
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 gamma: float = 1.0,
                 learn_weights: bool = False) -> None:
        super().__init__()

        self.feat_dim = feat_dim

        # --- Signal 3: tiny learned confidence MLP ---
        # TODO (Mumtaj): verify hidden_dim=64 is sufficient, can try 128 if underfitting
        self.confidence_mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),   # output in [0, 1]
        )

        # --- Combination weights ---
        if learn_weights:
            # TODO (Mumtaj): learnable weights — initialize to 1.0, let training adjust
            self.alpha = nn.Parameter(torch.tensor(alpha))
            self.beta  = nn.Parameter(torch.tensor(beta))
            self.gamma = nn.Parameter(torch.tensor(gamma))
        else:
            # fixed scalars — simpler, use this first, switch to learnable if results weak
            self.register_buffer("alpha", torch.tensor(alpha))
            self.register_buffer("beta",  torch.tensor(beta))
            self.register_buffer("gamma", torch.tensor(gamma))

    def _motion_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Signal 1: L2 norm of each segment's feature vector.
        
        High norm = high activity = more likely informative segment.
        
        Args:
            x: (B, T, feat_dim)
        Returns:
            energy: (B, T)  — L2 norm per segment
        """
        # TODO (Mumtaj): implement
        # Hint: torch.norm(x, dim=-1) or x.norm(p=2, dim=-1)
        return torch.norm(x, p=2, dim=-1)

    def _temporal_variance(self, x: torch.Tensor) -> torch.Tensor:
        """Signal 2: local temporal variance — how much this segment differs from neighbors.
        
        Computed as variance of the segment relative to its local window.
        Simple version: variance across the T dimension per feature, then mean over features.
        
        Args:
            x: (B, T, feat_dim)
        Returns:
            variance: (B, T)
        """
        # TODO (Mumtaj): implement
        # Simple approach: compute rolling variance with window=3 or 5
        # Simplest approach (start here): just use x.var(dim=1, keepdim=True).expand_as(x[...,0])
        # That gives global variance per video — not truly local, but easy and often works
        # Better approach: for each t, compute var of x[:, max(0,t-2):t+3, :] 
        # Start simple, upgrade if AUC is weak.
        B, T, D = x.shape
        x_pad = F.pad(x.transpose(1, 2), (2, 2), mode="replicate").transpose(1, 2)
        windows = x_pad.unfold(1, 5, 1)
        var = windows.var(dim=-1, unbiased=False)
        return var.mean(dim=-1)

    def _confidence(self, x: torch.Tensor) -> torch.Tensor:
        """Signal 3: learned confidence score from tiny MLP.
        
        Args:
            x: (B, T, feat_dim)
        Returns:
            conf: (B, T)  — confidence score in [0, 1]
        """
        # TODO (Mumtaj): implement
        # Hint: pass x through self.confidence_mlp, squeeze last dim
        # self.confidence_mlp expects (*, feat_dim) input, works on any batch shape
        return self.confidence_mlp(x).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-segment importance score.
        
        Args:
            x: (B, T, 1024) — batch of video features
        Returns:
            scores: (B, T)  — importance score per segment (higher = keep)
        """
        # TODO (Mumtaj): implement forward after the 3 signals are done
        # Steps:
        #   1. energy   = self._motion_energy(x)        # (B, T)
        #   2. variance = self._temporal_variance(x)    # (B, T)
        #   3. conf     = self._confidence(x)           # (B, T)
        #   4. Normalize each signal to [0,1] range per video
        #      (so no single signal dominates due to scale)
        #      e.g. energy_norm = (energy - energy.min(-1, keepdim=True).values) /
        #                         (energy.max(-1, keepdim=True).values - energy.min(-1, keepdim=True).values + 1e-8)
        #   5. scores = alpha * energy_norm + beta * variance_norm + gamma * conf
        #   6. return scores   # (B, T)
        energy   = self._motion_energy(x)
        variance = self._temporal_variance(x)
        conf     = self._confidence(x)

        def _norm(t):
            mn = t.min(dim=-1, keepdim=True).values
            mx = t.max(dim=-1, keepdim=True).values
            return (t - mn) / (mx - mn + 1e-8)

        scores = self.alpha * _norm(energy) + self.beta * _norm(variance) + self.gamma * conf
        return scores


if __name__ == "__main__":
    # Smoke test — will fail until TODOs are implemented, that's expected
    B, T, D = 4, 32, 1024
    x = torch.randn(B, T, D)
    scorer = SegmentScorer(feat_dim=D, hidden_dim=64)
    print(f"SegmentScorer created. Params: {sum(p.numel() for p in scorer.parameters()):,}")
    try:
        scores = scorer(x)
        print(f"Output shape: {scores.shape}  (expected ({B}, {T}))")
        assert scores.shape == (B, T)
        print("Smoke test passed.")
    except NotImplementedError as e:
        print(f"[expected] {e} — implement TODOs above.")