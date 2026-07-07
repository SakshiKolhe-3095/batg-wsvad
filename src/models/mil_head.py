"""
src/models/mil_head.py
BATG — MIL Anomaly Scoring Head

Adapted from RTFM's model.py (already patched by Mumtaj for 1024-dim features).
Takes gated features (segments that passed BATG's keep/drop decision) and
produces an anomaly score per segment.

During training (MIL):
  - Input contains B videos (mix of abnormal + normal, loaded in pairs)
  - Loss: top-k MIL ranking loss (same as RTFM)
  - Top-k segments from abnormal bag should score higher than top-k from normal bag

During inference:
  - Input: one video's kept segments
  - Output: anomaly score per kept segment → used for frame-level AUC

Input:  (B, T_kept, 1024)  — only kept segments after gating
Output: (B, T_kept)        — anomaly score per segment, in [0, 1]

IMPORTANT: T_kept varies per video per budget level.
The head must handle variable-length input — use per-segment scoring (no global pooling
before scoring, only after for the MIL loss).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MILHead(nn.Module):
    """MIL anomaly scoring head, adapted from RTFM model.py.
    
    Args:
        feat_dim:   input feature dimension (default 1024)
        dropout:    dropout rate (default 0.7, same as RTFM)
        k_ratio:    top-k ratio for MIL loss (default 0.2 = top 20% of segments)
    """

    def __init__(self,
                 feat_dim: int = 1024,
                 dropout: float = 0.7,
                 k_ratio: float = 0.2) -> None:
        super().__init__()

        self.k_ratio = k_ratio

        # TODO (Mumtaj): adapt from RTFM's model.py FC layers
        # RTFM uses: Linear(2048→512) but we use 1024-dim features
        # So: Linear(1024→512) → ReLU → Dropout → Linear(512→128) → ReLU → Dropout → Linear(128→1) → Sigmoid
        # This matches the patched version you already made work in RTFM
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score each kept segment.
        
        Args:
            x: (B, T_kept, 1024) — gated features
        Returns:
            scores: (B, T_kept) — anomaly score per segment in [0, 1]
        """
        # TODO (Mumtaj): implement — should be straightforward given RTFM familiarity
        # Hint: self.fc expects (..., feat_dim) input, works on any leading dims
        # scores = self.fc(x).squeeze(-1)   # (B, T_kept)
        # return scores
        scores = self.fc(x).squeeze(-1)
        return scores

    def mil_loss(self,
                 scores: torch.Tensor,
                 labels: torch.Tensor,
                 mask: torch.Tensor = None) -> torch.Tensor:
        """Top-k MIL ranking loss (same formulation as RTFM).
        
        For each abnormal video, top-k segment scores should be high.
        For each normal video, top-k segment scores should be low.
        
        Args:
            scores: (B, T_kept) — anomaly scores from forward()
            labels: (B,)        — video-level labels (1=anomaly, 0=normal)
            mask:   (B, T)      — optional BATG gate mask (to align T dimensions if needed)
        Returns:
            loss: scalar MIL loss
        """
        # TODO (Mumtaj): adapt MIL loss from RTFM's train.py
        # Steps:
        #   1. k = max(1, int(scores.shape[1] * self.k_ratio))
        #   2. top_scores = scores.topk(k, dim=1).values.mean(dim=1)   # (B,) mean of top-k per video
        #   3. abnormal_scores = top_scores[labels == 1]
        #   4. normal_scores   = top_scores[labels == 0]
        #   5. loss = torch.clamp(1 - abnormal_scores.mean() + normal_scores.mean(), min=0)
        #      (ranking loss: anomaly videos should score > normal by margin 1)
        #   6. add smoothness loss on scores (penalize abrupt score changes)
        #      smooth_loss = torch.mean((scores[:, 1:] - scores[:, :-1])**2)
        #   7. return loss + 0.01 * smooth_loss  (0.01 = weight from RTFM paper)
        k = max(1, int(scores.shape[1] * self.k_ratio))
        top_scores = scores.topk(k, dim=1).values.mean(dim=1)

        abnormal_scores = top_scores[labels == 1]
        normal_scores   = top_scores[labels == 0]

        if abnormal_scores.numel() == 0 or normal_scores.numel() == 0:
            loss = torch.tensor(0.0, device=scores.device, requires_grad=True)
        else:
            loss = torch.clamp(1 - abnormal_scores.mean() + normal_scores.mean(), min=0)

        smooth_loss = torch.mean((scores[:, 1:] - scores[:, :-1]) ** 2)
        return loss + 0.01 * smooth_loss


if __name__ == "__main__":
    B, T_kept, D = 4, 16, 1024   # T_kept = 16 if budget=0.5 and T=32
    x = torch.randn(B, T_kept, D)
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])  # 2 abnormal, 2 normal

    head = MILHead(feat_dim=D, dropout=0.7, k_ratio=0.2)
    print(f"MILHead created. Params: {sum(p.numel() for p in head.parameters()):,}")
    try:
        scores = head(x)
        print(f"scores shape: {scores.shape}  (expected ({B}, {T_kept}))")
        loss = head.mil_loss(scores, labels)
        print(f"MIL loss: {loss.item():.4f}")
        print("Smoke test passed.")
    except NotImplementedError as e:
        print(f"[expected] {e} — implement TODOs above.")