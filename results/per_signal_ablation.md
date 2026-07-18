\# Per-Signal Ablation



Tests whether each of BATG's three scoring signals (motion energy, temporal

variance, learned confidence) is individually load-bearing, by disabling one

signal at a time (via `disabled\_signals` param in SegmentScorer) and

retraining with otherwise identical hyperparameters (margin=0.5, k\_ratio=0.5,

smoothness=0.05, lambda\_budget=3.0, 100 epochs, seed 42).



| Configuration | 25% | 50% | 75% |

|---|---|---|---|

| Full model (all 3 signals) | 55.50% | 57.90% | 61.30% |

| No energy | 54.96% | 58.81% | 61.35% |

| No variance | 55.50% | 58.37% | 61.33% |

| No confidence | 55.24% | 58.27% | 61.16% |



\*\*Finding:\*\* All four configurations are statistically indistinguishable

(maximum spread within any single budget level ≤0.91pt). Reported as an

honest null result — two possible readings, both stated: (1) no single

signal is individually load-bearing given current feature quality/signal

redundancy, or (2) the early-plateau property shared across all BATG

configs (see main training logs — best checkpoint consistently lands near

epoch 0 regardless of configuration) limits how much any single ablation

can reveal about signal-specific importance.

