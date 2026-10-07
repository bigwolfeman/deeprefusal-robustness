# Agent Note: Adaptive-subspace DeepRefusal (E3)

Status: proposed

## Problem

DeepRefusal trains against one fixed refusal direction `r̂`, found offline before training. In the official code (`get_direction_ablation_hooks` in `src/main.py`), each hooked module (layer input, attention output, MLP output) is ablated in a forward pass with probability `ablation_prob` (default 0.05). Inside an ablated module, each token is ablated with the same probability. `args.py` sets `only_response=False` by default, so prompt and padding positions are eligible too. (With `only_response=True` the mask would still be wrong: sequences are left-padded to 1024, but `response_start_idx` counts from the unpadded start.) So about p² = 0.25% of (module, position) cells are ablated per forward pass. Weight abliteration removes the direction at every layer and every position at once. Two public attacks break the release:

- abliterix scales the weight delta toward the base model, then abliterates.
- APS steers with per-layer probes.

Our threat model excludes the base model (see [constitution](../../../../docs/constitution.md#threat-model)), so abliterix's exact recipe is out of scope. APS is not, and multi-direction weight edits may not be either ([E1](../../../../lab/experiments/planned/2026-10-07-e1-abliteration-without-base.md) tests this). If [E2](../../../../lab/experiments/planned/2026-10-07-e2-refusal-rebuild-geometry.md) confirms redundancy, DeepRefusal teaches the model to carry refusal in a few extra directions. An attacker who removes those directions too wins. A defense that trains against one direction leaves the next few directions open.

## Proposal

Extend DeepRefusal training so the ablation target follows the model:

1. **Growing ablation subspace.** Start with `S_0 = {r̂}`. Every N steps, refit the refusal directions on the current model under ablation of `S_t` (E2's `r'_l` procedure). Add the top-m new directions per layer to `S_t`, up to a cap `k_max`. Ablation hooks then project out the whole of `S_t`, with DeepRefusal's per-layer and per-token Bernoulli sampling.
2. **Attacker-distribution steps.** With probability q per step, ablate `S_t` at every layer and every position at once. This is what weight abliteration does at inference. With the official sampling (about 0.25% of cells), training almost never sees that case.
3. **Capability anchor.** Add a KL term on benign prompts against the starting model, so the subspace growth cannot buy robustness with capability.
4. **Capacity ablation.** Compare LoRA rank 16 (the release), LoRA rank 64, and full fine-tuning. A rank-16 delta may lack the room to make refusal non-linear.

**Gating.** Build this after E1 and E2 have verdicts.
- E2 supports redundancy and E1 breaks DR with subspace removal: build as written. Target class C.
- E1 shows DR holds against class C: keep the design, but make the refit step an APS-style probe attacker. Target class B.
- E2 supports rebuild: revisit this note before building. The failure is then elsewhere.

**Model.** 1B class. The choice between Gemma-3-1B-it (continuity with the ECS-189G work) and Llama-3.2-1B-Instruct (same family as the official 8B release) is open.

## Alternatives considered

- **Robust Self-Attention (Mu and Wagner, 2021), proposed by Gratitude.** It replaces attention aggregation with outlier masking, with no parameters. Rejected for this project: with open weights the attacker also has the code and loads standard attention. It also targets token-outlier attacks such as GCG, which DeepRefusal already resists (2% ASR in the paper).
- **SEAM alone.** It makes harmful and benign gradients conflict, so it targets class D (fine-tuning). It does nothing against activation steering or weight edits. Kept for E4, after this defense exists.
- **TAR (tamper-resistant safeguards).** Meta-learning against fine-tuning attacks. Also class D, and expensive. A candidate for E4.
- **Full fine-tuning to hide the LoRA structure.** This hides the rank cliff that abliterix used, but under our threat model the attacker has no base to diff against anyway. It is obscurity, not defense. It stays only as the capacity ablation in step 4.
- **Dynamic single direction.** The DeepRefusal authors report that re-estimating one direction during training was unstable. We keep `r̂` fixed and only add directions, which should be more stable. This is a risk, not a certainty.

## Acceptance criteria

All on the 1B model, with E0 measurements.

1. **Reproduction first.** DR-1B, trained with the official code and an E0-gated direction, has ASR_J1 ≤ 10% under all-layer `r̂` ablation, while the base model has ≥ 60%.
2. **Robustness.** Under the strongest attack without the base model in the target class, run adaptively (the attacker knows the method and refits beyond `S_t`), the best ASR_J1 at KL_1 ≤ 0.1 is at least 30 points lower than DR-1B's.
3. **Capability.** MMLU and ARC-C within 2 points of DR-1B. XSTest over-refusal within 5 points.
4. **No regression.** Prefill and template-attack ASR no worse than DR-1B, within confidence intervals.
5. **Reporting.** Full attacker-cost curves (ASR against KL_1 and against subspace rank) for base, DR, and adaptive DR.

## Risks

- Training may be unstable as `S_t` changes, as the authors saw with a dynamic direction.
- Capability may fall as `S_t` grows. Removing many directions also removes useful features.
- There may be a floor. Refusal text is a linear readout of the final residual stream, so some last-layer direction always exists. Removing it may cost coherence for the attacker, which counts as a success, or the defense may not reach it, which is a real negative result.
- Over-refusal may rise.
- An adaptive attacker who refits past `k_max` may still win. Then the result is a cost curve, not robustness, and the paper must say so.
- Compute: Kaggle T4 has no bf16. A 1B LoRA run took Gratitude 8 to 10 h on a 5070 Ti. Measure one run before planning the sweep.
