\# Non-Learned Floor Baseline



Motion-energy-only scoring (L2 norm of I3D features, min-max normalized per

video, no trained scorer/gate/MIL-head) used directly as the anomaly score,

evaluated through the same frame-alignment pipeline as BATG.



| Method | AUC |

|---|---|

| Random chance | 50.00% |

| Non-learned floor (motion-energy-only) | 55.58% |

| BATG (learned, sampled-budget, 100%) | 62.43% (mean, 3 seeds) |

| Published SOTA (BN-WVAD) | 87.24% |



\*\*Finding:\*\* BATG's learned model sits \~7 points above a trivial non-learned

floor and \~12.4 points above random chance — the learned components are

adding meaningful signal beyond what raw motion energy alone captures, even

though the absolute number remains well below published SOTA on substituted

features.

