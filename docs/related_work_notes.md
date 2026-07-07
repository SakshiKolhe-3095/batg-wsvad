# 2026-06-22

Completed:
- Created repo
- Created venv
- Cloned RTFM

Issues:
- Feature download link unavailable

Next:
- Run baseline inference

# 3-Signal Justification Paragraph

*(For related-work or methodology section — justifies motion energy + temporal
variance + confidence score as the chosen signal set)*

BATG's segment scorer combines three signals chosen specifically because each is
computable directly from already-extracted features, before any backbone
computation is spent, and because together they capture complementary notions
of segment importance. Motion energy—the L2 norm of a segment's feature
vector—serves as a cheap proxy for activity intensity: segments with abrupt
motion or visual change tend to produce higher-magnitude feature responses,
and anomalous events in surveillance footage (fights, accidents, running) are
disproportionately motion-heavy. Temporal variance captures a different
property: how much a segment differs from its local neighbors, flagging
transitions and discontinuities that a single-frame or static-activity measure
would miss—useful for anomalies that manifest as a *change* in scene state
rather than raw motion alone (e.g., an object appearing, a crowd dispersing).
The learned confidence score, a lightweight MLP trained jointly with the rest
of the pipeline, complements the two hand-crafted signals by letting the model
pick up dataset-specific or task-specific importance cues that motion energy
and temporal variance do not directly encode. No single signal is sufficient
on its own: motion energy alone would over-select high-activity normal scenes
(e.g., crowded but benign pedestrian traffic); temporal variance alone would
over-select any abrupt but irrelevant visual change (e.g., camera exposure
shift); confidence alone, learned under only video-level weak labels, would be
undersupervised early in training without the other two signals as a prior.
Combined, the three signals are cheap enough to compute before the expensive
backbone runs, yet expressive enough to produce a keep/drop decision that
correlates with true segment informativeness under weak supervision.

# Two-Tier Baseline Split Defense Paragraph

*(For related-work or experimental-setup section — justifies why accuracy
baselines and efficiency baselines are kept as two separate comparison groups
rather than one merged table)*

This work evaluates BATG against two distinct groups of baselines because the
paper makes two distinct claims that require two distinct proofs. The first
claim—that BATG does not sacrifice accuracy relative to standard WS-VAD
models—is evaluated against accuracy-oriented baselines (Sultani et al., 2018;
RTFM, 2021; BN-WVAD) that make no attempt at compute efficiency and instead
optimize purely for detection accuracy at full compute. These serve as an
accuracy ceiling: if BATG at its full budget (B=100%) matches this ceiling, the
first claim is supported. The second claim—that no existing method exposes a
user-controllable compute–accuracy tradeoff—is evaluated against efficiency
oriented baselines (Wang et al., 2023; Karim et al., 2024; Mohamad et al.,
2026; Zhang et al., 2025 / DE-Net) that do incorporate an efficiency mechanism
but, critically, each commits to a single fixed operating point at design or
training time. Merging these two groups into one table would obscure both
comparisons: accuracy baselines would look artificially inefficient (they were
never designed to be efficient, so comparing their FLOPs is not meaningful),
while efficiency baselines would look artificially inaccurate if judged purely
on their one reported point without acknowledging that that point was
optimized for a specific compute budget, not for the compute-agnostic accuracy
these accuracy-tier models pursue. Keeping the tiers separate lets each
comparison answer the question it is actually suited to answer, and keeps the
paper's central contribution—the FLOPs–AUC curve itself—positioned correctly
relative to both a fixed accuracy ceiling and a set of fixed efficiency
points.