# Mohamad 2026 — Numbers Only (no code available)

**Paper:** Muhammad Luqman Arif Mohamad, Mohd Amiruddin Abd Rahman,
Nurisya Mohd Shah, Arun Kumar Sangaiah, "Efficient NASNetMobile-Enhanced
Vision Transformer for Weakly Supervised Video Anomaly Detection"
(NASNetMobile-EViT), Jan 2026.

**Code status:** No public code repository found as of 2026-07-05. Confirmed
no public code — numbers taken directly from paper / secondary citations.

## Method summary
- Frame motion selector module using Gaussian Mixture Model — ranks and
  samples frames rich in motion cues before downstream processing (cheap
  pre-filter, conceptually closest of the 3 numbers-only baselines to
  BATG's segment-scoring idea)
- NASNetMobile (pre-trained, low-parameter) for fine-grained local spatial
  feature extraction — replaces heavier 3D CNN / I3D backbones
- Enhanced Vision Transformer (EViT) with RMS normalization + Query-Adaptive
  Pooling extended from 2D images to temporal token maps, for long-range
  temporal relations
- Evaluated on 4-5 benchmarks depending on paper version: UCF-Crime,
  XD-Violence, ActivityNet-VAD, NREF (and ShanghaiTech/Avenue/Ped2 in some
  citing sources — verify exact benchmark set against original paper before
  final citation)

## Reported numbers — UCF-Crime
| Metric | Value |
|---|---|
| AUC (UCF-Crime) | **91.61%** |
| AP (XD-Violence) | 91.65% |
| Accuracy (ActivityNet-VAD) | 30.36% |
| Accuracy (NREF) | 77.00% |
| Params | **7.9M** |
| GFLOPs | **0.71** |
| Improvement over best I3D method (DFMBN) | +4.31% UCF-Crime, +6.54% XD-Violence |

## Relevance to BATG
- Closest of the three numbers-only baselines to BATG conceptually: the
  Gaussian-Mixture-Model frame motion selector is a pre-backbone cheap
  filtering step, similar in spirit to BATG's segment scorer — but
  selection is fixed at design/training time, not exposed as a
  user-controllable budget parameter at inference.
- Strong efficiency profile (7.9M params, 0.71 GFLOPs) at a single fixed
  operating point — no reported curve across multiple compute budgets.
- Best efficiency-tier comparison point for the "cheap pre-filter" framing
  in the paper's related-work section; use this baseline when specifically
  arguing BATG's segment-scoring approach differs by exposing a tunable
  budget rather than a fixed motion-based selection threshold.
- CAVEAT: reported AUC (91.61%) is notably higher than RTFM (84.30%) and
  even BN-WVAD (87.24%) — verify this is genuinely comparable protocol
  (same train/test split, frame-level AUC definition) before using in
  direct side-by-side table; some VAD papers report AUC_A (abnormal-only)
  or different aggregation which can inflate numbers relative to standard
  frame-level AUC_O used by RTFM/Sultani/BN-WVAD.