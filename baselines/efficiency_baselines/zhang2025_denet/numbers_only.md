\# Zhang 2025 DE-Net — Numbers Only (repo has no usable code)



\*\*Paper (this repo's actual version):\*\* Chen Zhang, Guorong Li, Yuankai Qi, Hanhua Ye,

Laiyun Qing, Ming-Hsuan Yang, Qingming Huang, "Dynamic Erasing Network Based on

Multi-Scale Temporal Features for Weakly Supervised Video Anomaly Detection"

(DE-Net, MSTM module), arXiv:2312.01764 (submitted 4 Dec 2023)



\*\*Separate 2025 journal version exists:\*\* Same first author, different module —

"Dynamic Erasing Network With Adaptive Temporal Modeling for Weakly Supervised

Video Anomaly Detection" (ATM module), IEEE TNNLS, DOI 10.1109/TNNLS.2025.3553556

(online 8 Apr 2025). This is likely the "Zhang 2025" version originally intended

for this baseline slot — NOT the one cloned into this repo.



\*\*Code status:\*\* Cloned repo (github.com/ArielZc/DE-Net, commit 09de5825)

contains only a README and demo video — no model/training code, despite the

paper stating "Code will be made available at this https URL." No code was

ever actually published for either version, as of 2026-07-11. Numbers taken

directly from the 2023 arXiv paper (the version actually available/cited).



\## Method summary

\- Multi-Scale Temporal Modeling (MSTM): parallel Conv1D branches at different

&#x20; strides/kernel sizes, each followed by a transformer encoder, fused via FC

\- Dynamic Erasing (DE): after first pass, zeros out high-scoring segments in

&#x20; incomplete-looking abnormal videos, re-runs MSTM+classifier to force

&#x20; discovery of subtler anomalous segments

\- Feature extractor: Kinetics-pretrained I3D, 16-frame clips, T=64 segments

\- Classifier: 3-layer FC (512-32-1), dropout 0.6, sigmoid output



\## Reported numbers — UCF-Crime (frame-level AUC)

| Metric | Value |

|---|---|

| AUC (UCF-Crime, I3D features) | \*\*86.33%\*\* |

| AUC (TAD dataset) | 93.10% |

| AP (XD-Violence, I3D only) | 81.66% |

| AP (XD-Violence, I3D+VGGish) | 83.13% |



\## Relevance to BATG

\- Efficiency mechanism = adaptive-\*scale\* selection at training/architecture

&#x20; level (which temporal resolution to model), not inference-time compute

&#x20; gating. All segments still processed by the full backbone at inference —

&#x20; no user-controllable budget, no FLOPs reduction mechanism.

\- Reports ONE fixed operating point (86.33% AUC, T=64 fixed) — confirms the

&#x20; gap BATG targets: no budget-controllable curve exists here either.

\- No FLOPs/params reported in the paper for direct efficiency-axis comparison

&#x20; with BATG; "efficiency" in this paper refers to representational efficiency

&#x20; (handling variable-duration anomalies), not compute efficiency.

