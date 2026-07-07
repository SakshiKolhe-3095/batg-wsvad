# Wang 2023 — Numbers Only (no code available)

**Paper:** Yang Wang, Jiaogen Zhou, Jihong Guan, "A Lightweight Video Anomaly
Detection Model with Weak Supervision and Adaptive Instance Selection"
(Light-WVAD), arXiv:2310.05330 (v1: 9 Oct 2023, v2: 5 Jul 2024)

**Code status:** No public code repository found as of 2026-07-05. Confirmed
no public code — numbers taken directly from paper.

## Method summary
- Adaptive Instance Selection (AIS): dynamically chooses number of top-K
  normal/abnormal instances used for MIL loss, based on current training
  status of the model (vs. RTFM's fixed K=3)
- Multi-level Temporal correlation Attention (MTA) module — lightweight,
  Conv1D-based, focuses model attention on important instances in time dim
- Hourglass-shaped Fully Connected (HFC) layer — 2048-64-128 structure
  (vs. conventional FC's 2048-128-64), ~half the parameters
- Feature extractor: I3D, 32 clips per video (same convention as RTFM/BATG)
- Novel "antagonistic loss" replacing standard sparsity loss

## Reported numbers — UCF-Crime (frame-level AUC)
| Metric | Value |
|---|---|
| AUC (UCF-Crime) | **84.7%** — highest among lightweight WS-VAD methods, 3rd overall among general models |
| AUC (ShanghaiTech) | 95.9% |
| AUC (XD-Violence) | 77.3% |
| Params | **0.14M** — <1% of RTFM's parameter count |

## Relevance to BATG
- Efficiency mechanism = adaptive instance *selection for training loss*,
  not inference-time compute gating. Model still runs full backbone on all
  32 segments at inference — the efficiency gain here is parameter count
  (model size), not FLOPs/compute reduction at inference.
- Reports ONE fixed operating point (0.14M params, 84.7% AUC) — no
  user-controllable budget curve. Confirms the gap BATG targets.
- Fair efficiency-tier comparison point: BATG should be compared at matched
  FLOPs/params where possible, noting Wang's efficiency axis (param count)
  differs from BATG's (inference-time compute budget).