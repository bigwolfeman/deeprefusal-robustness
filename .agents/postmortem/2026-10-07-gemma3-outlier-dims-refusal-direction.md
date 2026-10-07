# Postmortem: massive-activation dimensions hijack the Gemma 3 refusal direction

Date: 2026-10-07

## Executive summary

After the padding fix, no E0 candidate on `gemma-3-1b-it` passed Arditi's filters (induce > 0 and KL < 0.1). Ablating good candidates gave a first-token KL of 20 to 45 nats on harmless prompts, which breaks the model. Root cause: Gemma 3 has massive-activation residual dimensions (dimension 1038 most of all), with large values at every position, not only BOS. The harmful-minus-harmless mean difference is dominated by them (|cos| up to 0.89 with that one axis). Ablating the direction removes most of a feature the model needs everywhere. Zeroing those dimensions in every candidate gives 23 passing candidates. The selected direction passes the E0 gate (harmful refusal 82% → 0% when ablated; harmless refusal 0% → 90% when added; KL 0.06). The durable lesson: inspect the coordinates of a mean-difference direction before trusting it, on any model family with outlier dimensions.

## Timeline

- 01:25 CDT: with correct baselines, KL is still 20+ nats for every candidate that induces refusal.
- 01:27: excluding BOS from ablation does not help. The top coordinate of each failing direction is 1038 (0.68 to 0.89 of the unit vector).
- 01:29: zeroing outlier dimensions (mean |h| > 10× the layer median, in at least 3 layers) gives 23 passing candidates. Best: layer 15, last template position, bypass −25.2, induce +10.2, KL 0.060.
- 01:31: E0 gate passes. Spot-checked generations are coherent compliance under ablation and coherent refusal under addition.

## What broke

Arditi selection on Gemma 3 1B. The ECS-189G reproduction's last-layer Gemma 3 direction also failed its own add and ablate checks. That failure is consistent with this cause, but it was not verified on that artifact.

## Root cause

Gemma 3 residual norms reach about 50,000 at the BOS position and about 10,000 elsewhere by layer 13. A few dimensions carry most of that magnitude. Their mean differs slightly between harmful and harmless prompts, and that difference dominates the raw mean difference in L2 norm.

## Why safety nets missed it

Arditi's KL filter caught it, which is what the filter is for. The fix needed a diagnosis, not a looser filter.

## Guardrails added

- `drlab.directions.outlier_dims` with `configs/direction.yaml` `outliers.{ratio: 10, min_layers: 3}`. The zeroed dims are saved in `direction.pt` and reused by every later extraction (quick eval, path test).
- The criterion is coarse. It zeroes 98 of 1152 dims, many flagged only in layers 1 to 5. A tighter criterion (deep layers only) is open. It did not block the gate.
- E0 amendment recorded in the [eval protocol](../../lab/experiments/algorithms/2026-10-07-refusal-eval-protocol.md).
