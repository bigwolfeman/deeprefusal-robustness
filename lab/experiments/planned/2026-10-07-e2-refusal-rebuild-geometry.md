# Experiment: E2, does DeepRefusal rebuild the refusal direction or route around it? (Figure 1 test)

Status: planned

Date proposed: 2026-10-07
Algorithm: [refusal eval protocol (E0)](../algorithms/2026-10-07-refusal-eval-protocol.md)
Owner: Wolfe + agent session (Claude Opus 5.5)

## Question

Figure 1d of the DeepRefusal paper is labeled a conceptual diagram. It shows the model entering a "jailbreak state" (harmful, refusal suppressed) and then moving back into a "refusal state". The paper does not measure this. Two mechanisms could explain DeepRefusal's resistance to refusal-direction ablation:

- **Rebuild** (Figure 1d taken literally): after `r̂` is removed at a layer, later layers write `r̂` back.
- **Redundancy**: the weight delta writes refusal information into other directions, so removing `r̂` leaves refusal readable elsewhere. Nothing is rebuilt.

They predict different geometry. Which one does the official release show? Redundancy would also explain why multi-direction attacks (APS, abliterix after attenuation) succeed.

## Hypothesis

DeepRefusal-8B's resistance to `r̂` ablation comes from redundancy, not rebuild. Under ablation of `r̂` at all layers, harmful and harmless prompts stay linearly separable along a small set of directions that lie mostly in the output span of the DeepRefusal weight delta. Removing those directions as well removes refusal. After a single-layer ablation, `r̂` does not recover in later layers more than it does in the base model.

## Predictions

Definitions (detail in Method): `ΔW = W_DR − W_base`. `U_l` is the span of the top-16 left singular vectors of `ΔW` for `o_proj` and `down_proj` at layers ≤ l. `r'_l` is the harmful-minus-harmless mean difference at layer l, at post-instruction positions, under all-layer `r̂` ablation. `sep_l = ‖r'_l‖ / ‖r_l‖`, where `r_l` is the same difference without ablation. `R(l → l+4)` is the recovery ratio: the `r̂` projection at layer l+4 after ablating at layer l only, divided by the unablated projection.

If the hypothesis is true, we will observe:

- M0: `ΔW` shows a singular-value drop at rank 16 in `o_proj`, `v_proj`, and `gate_proj`, as abliterix reported.
- M1: averaged over ablation layers l ∈ {4, 6, …, 26}, DR's `R(l → l+4)` exceeds the base model's by less than 0.2.
- M2: in layers 8 to 24, DR's `sep_l` ≥ 0.5, while the base model's `sep_l` ≤ 0.3.
- M3: the fraction of `‖r'_l‖²` inside `U_l` is at least 4 times the random baseline `dim(U_l)/4096`, in at least half of layers 8 to 24.
- M4 (causal): with hooks on DR that ablate `r̂` plus the top-4 per-layer principal directions of the residual difference, keyword and J1 refusal on 100 HarmBench prompts drops by at least 50 points. Ablating `r̂` plus 4 random directions per layer drops it by at most 10 points.
- M5 (Figure 1 map): under all-layer `r̂` ablation and under a 20-token prefill, fewer than 10% of DR's harmful-prompt trajectories show a dip-and-return. A dip-and-return means the refusal-probe score falls below the decision threshold for at least 3 consecutive response tokens and then rises above it.

If the hypothesis is false (rebuild holds), we will observe:

- M1: DR's `R(l → l+4)` exceeds the base model's by at least 0.3 on average.
- M5: at least 30% of attacked DR trajectories show a dip-and-return.
- M2 or M3 fails: DR's separation under all-layer ablation is not much larger than the base model's, or it does not concentrate in `U_l`.

If M1 and M2–M4 both hold, both mechanisms are present. We report that as a split verdict: the hypothesis is falsified as stated ("not rebuild"), and redundancy is supported.

What would make this run inconclusive (protocol failure, not a hypothesis test):

- M0 shows no rank cliff. The NousResearch mirror then may not be the exact base, and `ΔW` analyses mean nothing.
- The base-model `r̂` fails the E0 validation gate.
- DR under all-layer `r̂` ablation does not refuse (ASR_J1 ≥ 50%). The paper's 0.4% claim would be false on this release. That is a finding, but there is then no robustness for this experiment to explain. File it, and redirect to E1.
- The refusal probe for M5 has less than 90% held-out accuracy on unattacked DR activations.

## Method

**Threat-model line.** Mechanism study, not an attack. Using `W_base` here is allowed because the defender analyzes their own model. No result from this file is reported as an attack.

**Models.** Same as E1: `NousResearch/Meta-Llama-3-8B-Instruct` and `skysys00/Meta-Llama-3-8B-Instruct-DeepRefusal`. Also, for M5 only: `wangzhang/Llama-3-8B-Instruct-DeepRefusal-Broken` (abliterix) and `FTK11558/Meta-Llama-3-8B-Instruct-DeepRefusal-Broken-APS`. They show where a broken model sits on the map. We only read them; we do not redistribute them.

**Directions.**
- `r̂`: E0 direction from the base model, gated.
- Harmfulness direction `ĥ`: following Zhao et al. 2025 ("LLMs Encode Harmfulness and Refusal Separately"), the harmful-minus-harmless mean difference at the last token of the user instruction, at the same layer as `r̂`. Report `cos(ĥ, r̂)`.

**M0.** SVD of `ΔW` for every linear layer. Report the spectra and the rank where 99% of the Frobenius energy is reached.

**M1, rebuild test.** For each l ∈ {4, 6, …, 26}, ablate `r̂` at the residual output of layer l only (all positions). Measure the `r̂` projection at the last post-instruction token at layers l+1 to L, on 128 HarmBench prompts. Same for the base model.

**M2, M3, separation.** Ablate `r̂` at every layer on DR and on base, using the DeepRefusal paper's sites (attention output, MLP output, residual). Compute `r'_l`, `sep_l`, and the fraction of `r'_l` in `U_l`, on E0 fit data.

**M4, causal check.** Activation hooks (class B, not weight edits, so it complements E1). Per layer, take the top-4 principal components of the harmful-minus-harmless differences under `r̂` ablation, and ablate them together with `r̂`. The control uses 4 random orthonormal directions per layer, 5 random seeds. Generate on 100 HarmBench prompts, judge with keyword and J1.

**M5, Figure 1 map.** 2D coordinates per response token: x is the projection on `ĥ`, and y is the score of a logistic "will refuse" probe. The probe is trained on unattacked DR residual activations (full space, layer of `r̂`, harmful versus harmless, 80/20 split). Conditions: {base, DR} × {no attack, 20-token prefill from the AdvBench target strings, all-layer `r̂` ablation}, plus the two broken checkpoints with no attack. 100 HarmBench prompts, first 64 response tokens, teacher-forced on each model's own greedy output. Plot the mean trajectories and count dip-and-return events.

**Compute.** Forward passes only. 8B bf16 on the 5090 or the DGX Spark. Estimate (not measured): about 4 GPU-hours, plus J1 judging for M4.

**Seeds.** Data subsampling seed 0. Random-direction control: seeds 0 to 4.

**Logging.** Hydra config, wandb project `deeprefusal-robustness`, group `e2`. Plots and metrics go to `lab/experiments/results/2026-10-07-e2-refusal-rebuild-geometry/`. Generations go to `ignored/experiment-artifacts/`.

## Related

- [E1 abliteration without base](2026-10-07-e1-abliteration-without-base.md): the attack-side counterpart.
- [Adaptive DeepRefusal proposal](../../../.agents/notes/proposed/feature/2026-10-07-adaptive-deeprefusal.md): its design depends on which mechanism holds.
- DeepRefusal paper, Figure 1 and Algorithm 1: arXiv 2509.15202.
