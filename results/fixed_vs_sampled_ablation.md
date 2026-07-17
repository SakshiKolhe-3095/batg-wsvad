\# Fixed-Budget vs. Sampled-Budget Training Ablation



Tests whether BATG's core design choice (one model trained across randomly-sampled

budgets) sacrifices accuracy compared to training four separate models, each

fixed to one budget level.



| Budget | Sampled model (mean, 3 seeds) | Fixed-budget model | Difference |

|---|---|---|---|

| 25%  | 55.31% | 55.90% | +0.59 |

| 50%  | 57.93% | 58.40% | +0.47 |

| 75%  | 61.08% | 61.30% | +0.22 |

| 100% | 62.43% | 63.10% | +0.67 |



\*\*Finding:\*\* Fixed-budget models perform marginally better at every level (as

expected for specialized models), but the gap is small (<0.7 points) and does

not grow with budget. This supports BATG's design: a single model trained

across sampled budgets achieves accuracy within \~0.5pt of four separately

trained dedicated models, at a fraction of the training/deployment cost

(one model vs four).

