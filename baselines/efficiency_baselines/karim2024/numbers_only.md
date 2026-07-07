# Karim 2024 — Numbers Only (no code available)

**Paper:** Hamza Karim, Keval Doshi, Yasin Yilmaz, "Real-Time Weakly
Supervised Video Anomaly Detection," WACV 2024, pp. 6848-6856.
DOI: 10.1109/WACV57701.2024.00670

**Code status:** No public code repository found as of 2026-07-05. Confirmed
no public code — numbers taken directly from paper.

## Method summary
- Targets real-time / near-real-time anomaly decisions (paper defines
  "real-time" as decision period < 30 sec)
- Shows existing SOTA WS-VAD models' performance is proportional to number
  of frames processed at once — shrinking the decision window degrades
  their AUC significantly
- Proposes a method that maintains strong AUC even at short decision periods

## Reported numbers — UCF-Crime
| Metric | Value |
|---|---|
| AUC (their method) | **86.94%** |
| Decision period (their method) | **6.4 seconds** |
| AUC (best competing method) | at most 85.92% |
| Decision period (best competing method) | 273 seconds |

## Relevance to BATG
- Efficiency axis here = **decision latency / temporal window size**, not
  FLOPs or compute-per-segment. Fixed at one operating point: their model
  runs at 6.4s decision period, full stop — no user-controllable dial.
- Different efficiency dimension than BATG (BATG controls FLOPs via
  segment keep/drop fraction; Karim controls decision-window length).
  Still valid efficiency-tier baseline: both represent "one fixed
  efficient operating point" vs. BATG's tunable curve.
- Note for paper: when citing, be precise that Karim's "real-time" claim
  is about decision latency, not raw compute (FLOPs) reduction — avoid
  conflating the two efficiency axes in write-up.