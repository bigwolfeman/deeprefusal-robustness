# Experiment: E4, does on-policy training data remove DeepRefusal-1B's over-refusal and prefill regression?

Status: failure

Date proposed: 2026-10-07
Date run: 2026-10-07
Artifacts: [results/2026-10-07-e4-onpolicy-dr-1b](../results/2026-10-07-e4-onpolicy-dr-1b/)
Algorithm: [refusal eval protocol (E0)](../algorithms/2026-10-07-refusal-eval-protocol.md); data design in [on-policy DeepRefusal data note](../../../.agents/notes/implemented/feature/2026-10-07-onpolicy-dr-data.md)
Owner: Wolfe + agent session (Claude Opus 5.5)

Written on 2026-10-07 at about 12:30 CDT. The on-policy data did not exist yet, and no E4 model was trained. The E3 J2 results for the off-policy models were known (they motivate this experiment) and are quoted below as the comparison point.

## Question

The faithful DeepRefusal recipe trains on responses that another model wrote: CircuitBreaker refusals are Llama-3-8B-Instruct outputs (`llama3_output`), and UltraChat and XSTest answers come from GPT models. For Llama-3 (the paper's model) the refusals are on-policy. For Gemma-3-1B they are not. Is off-policy response text the cause of DR-1B's two regressions in E3?

E3 off-policy `dr_baseline` vs base `gemma-3-1b-it` (J2, 1 seed, n = 100, keyword over-refusal on 250 OR-Bench-Hard prompts):

| Metric | base | dr_baseline |
|---|---|---|
| OR-Bench-Hard keyword refusal | 0.32 | 0.99 |
| Prefill ASR_J2 (AdvBench targets) | 0.14 | 0.48 |
| Fresh-direction ASR_J2 (KL_1 ≤ 0.1) | 0.28 | 0.13 |
| MMLU proxy (0-shot, 1,000) | 0.431 | 0.412 |

## Hypothesis

**H-OP.** Off-policy response text causes most of the over-refusal and the prefill regression. DR-1B trained on the same prompts, counts, loss, and p = 0.5, with every response written by `gemma-3-1b-it` itself, keeps DR's fresh-direction robustness and moves over-refusal and prefill ASR back toward the base model.

Secondary, **H-P.** Over the ablation probability p ∈ {0.1, 0.3, 0.5}, on-policy DR-1B traces a frontier: higher p gives lower fresh-direction ASR and higher over-refusal.

Agent priors before any result: H-OP about 45%. H-P about 55%. The most likely partial outcome (about 30%): over-refusal falls but fresh-direction ASR rises toward the base model's, which would mean DR-1B's robustness in E3 was mostly over-refusal.

## Predictions

Models: M0 `gemma-3-1b-it`; M1 `dr_baseline` (E3, off-policy, p = 0.5); N1, N3, N5 = on-policy DR-1B at p = 0.1, 0.3, 0.5 (`dr_onpolicy_p01/p03/p05`, seed 0).

If H-OP is true, N5 shows all of:
- OR-Bench-Hard keyword refusal ≤ 0.60 (M1 0.99, M0 0.32).
- Prefill ASR_J2 ≤ 0.24 (M1 0.48, M0 0.14).
- Fresh-direction ASR_J2 at KL_1 ≤ 0.1 ≤ 0.18 (M1 0.13, M0 0.28).
- MMLU proxy ≥ 0.41.

If H-OP is false, we observe either:
- N5 OR-Bench-Hard refusal > 0.80, or N5 prefill ASR_J2 > 0.40 (off-policy text is not the main cause), or
- N5 keeps over-refusal ≤ 0.60 only with fresh-direction ASR_J2 > 0.22 (on-policy data removes over-refusal and the robustness with it; DR-1B robustness was over-refusal).

If H-P is true: fresh-direction ASR_J2 is ordered N1 ≥ N3 ≥ N5 and OR-Bench-Hard refusal is ordered N1 ≤ N3 ≤ N5, each allowing one inversion of at most 0.05 (one seed, n = 100).

What would make this run inconclusive (protocol failure):
- The on-policy generator drops more than 10% of harmful prompts or more than 10% of benign prompts (the data then differs in prompts, not only in response source).
- A training run does not complete (`meta.json` `completed: false`).
- A decisive comparison lies within 0.05 of its threshold. Then retrain N5 and M1 with seeds 1 and 2 before the verdict.

## Method

**Single changed variable.** Same prompts, same counts, same order of construction, same seed as `dr_baseline` (`configs/train.yaml` data block): 229 XSTest compliant prompts × 2, 3,542 UltraChat `test_sft` first turns, 2,000 CircuitBreaker prompts (seed-42 shuffle), each with one harmful-prefix copy (k ~ U[20, 25] tokens of CB `output`, masked in the loss), empty-instruction probability 0.5 for benign rows. LoRA r16/α16, batch 16, 1 epoch, α = 0.2, lr 1e-4. Only the response text changes.

**On-policy responses** (`python -m drlab.onpolicy`, `configs/onpolicy.yaml`), all from `gemma-3-1b-it`, eager attention:
- Benign: greedy response, up to 768 new tokens. If it is a keyword refusal, use the greedy response with `r̂` ablated at every site (the model's own compliant voice). If that also refuses, drop the row.
- Harmful: the first candidate that J2 scores below 0.1, in this order: greedy, 2 samples at T = 0.7, greedy with `r̂` added at the direction layer (the E0 gate's steering, which induced 90% refusal on harmless prompts). If none qualifies, drop the prompt.
- Prefix recovery: continuation of prompt + harmful prefix, greedy, then greedy with `r̂` added. Accept if the continuation is a keyword refusal in its first 300 characters and J2 < 0.1. Otherwise use the prompt's accepted refusal after the prefix (the paper's construction, in Gemma's voice).
- A response that did not end with an end-of-turn token keeps its text, and its training target gets no end-of-turn token.
- Length control (`data.onpolicy_match_length: true`). Gemma-3-1B-it writes long responses: in a 241-prompt smoke run, the median benign answer passed 768 tokens, and refusals passed 384. The DR loss sums over tokens per sample, so each on-policy response is capped at the token count of the off-policy response it replaces. Every row keeps its off-policy loss weight (smoke run: 81,347 vs 82,504 labeled tokens). A capped response gets no end-of-turn target.
- The off-policy build after the refactor is identical to the E3 data (8,000 of 8,000 examples, equal token ids and labels).
- The generator writes per-source counts to `summary.json`. Generations stay on the 3070 (`/mnt/bigdata/deeprefusal/data/onpolicy/`), never in git.

**Known risk, judge reuse.** J2 selects the harmful training targets, and J2 also scores the attacks. J2 selection only admits responses that J2 rates as safe, so it cannot teach harmful output that J2 misses, but a style bias toward J2-safe text is possible. J1 (HarmBench classifier) rechecks the decisive cells when a large GPU is free.

**Evaluation.** `python -m drlab.bench --config-name bench_e4` with the E3 settings (n_harm 100, OR-Bench-Hard 250, MMLU proxy 1,000, fresh direction and fresh subspaces k ∈ {1, 2, 4, 8}), models M0, M1, N1, N3, N5, all rerun at the same commit (M0 and M1 are not copied from E3). Same J2 and keyword measures as E3.

**Compute.** RTX 3070, sequential queue. Generation, then N5, N3, N1 training (about 0.8 h each), then the bench.

## Results

Runs on the RTX 3070. Generation `onpolicy_gen` 12:25 to 13:34 CDT at `c78c88e`; trainings `dr_onpolicy_p05/p03/p01` 13:34 to 15:58 (each 500/500 steps, `completed: true`, about 0.75 h); `bench_e4` 15:58 to 16:23 at `b2da850`. All five models were benched in the same run. J2 only; J1 is queued (`j1_e4`).

**Data** (`onpolicy_data_summary.json`): drops 2/3,771 benign (0.05%) and 0/2,000 harmful. Harmful targets: greedy 1,367, sample 298, steered 335. Prefix recoveries: steered 1,402, refusal after prefix 451, greedy 147. 32% of CircuitBreaker prompts had no J2-safe greedy response from `gemma-3-1b-it`. Built set: 7,998 examples, 1.351M labeled tokens (off-policy: 8,000 and 1.430M).

**Bench** (ASR_J2 on 100 HarmBench standard prompts, KL_1 on 100 harmless prompts in brackets; OR-Bench-Hard keyword refusal on 250 prompts; 0-shot MMLU proxy on 1,000):

| Condition | M0 base | M1 off-policy p0.5 | N1 p0.1 | N3 p0.3 | N5 p0.5 |
|---|---|---|---|---|---|
| no attack | 0.01 | 0.00 | 0.00 | 0.01 | 0.00 |
| prefill (AdvBench) | 0.16 | 0.46 | 0.00 | 0.00 | 0.00 |
| original `r̂` ablated | 0.36 (0.08) | 0.09 (0.06) | 0.31 (0.10) | 0.23 (0.20) | 0.07 (0.61) |
| fresh direction ablated | 0.36 (0.08) | 0.14 (0.05) | 0.34 (0.10) | 0.04 (0.05) | 0.05 (0.19) |
| fresh per-layer subspace k = 1 | 0.44 (0.55) | 0.36 (0.20) | 0.33 (0.37) | 0.35 (0.46) | 0.35 (1.07) |
| k = 2 | 0.38 (0.91) | 0.35 (0.30) | 0.39 (0.54) | 0.36 (0.62) | 0.37 (1.30) |
| k = 4 | 0.45 (1.88) | 0.18 (0.48) | 0.38 (0.93) | 0.42 (1.02) | 0.43 (1.49) |
| k = 8 | 0.27 (2.19) | 0.11 (0.62) | 0.29 (2.11) | 0.25 (2.13) | 0.20 (2.38) |
| OR-Bench-Hard keyword refusal | 0.320 | 0.992 | 0.716 | 0.864 | 0.912 |
| MMLU proxy | 0.431 | 0.412 | 0.429 | 0.428 | 0.433 |

Against the H-OP predictions for N5: OR-Bench-Hard refusal 0.912 (predicted ≤ 0.60; the falsifying threshold was > 0.80). Prefill ASR_J2 0.00 (≤ 0.24, held). Fresh-direction ASR_J2 0.05, but its KL_1 is 0.19, so no N5 direction attack meets the KL_1 ≤ 0.1 condition as written. MMLU proxy 0.433 (≥ 0.41, held).

Against H-P: fresh-direction ASR_J2 N1 0.34, N3 0.04, N5 0.05 (one inversion of 0.01, inside the 0.05 allowance). OR-Bench-Hard refusal N1 0.716 ≤ N3 0.864 ≤ N5 0.912. Both orderings held.

Observations outside the predictions:
- On-policy training removes the prefill weakness completely (0.00 at every p), below the base model's 0.16. Off-policy DR raised it to 0.46.
- Over-refusal rises with refusal training even at p = 0.1, where ablation is rare (0.716 vs base 0.320). The CircuitBreaker refusal set itself, not the response voice, drives most of it.
- Every DR model, on-policy or off-policy, loses refusal to the fresh per-layer subspace at k = 1 (ASR_J2 0.33 to 0.36, base 0.44). DR's robustness is specific to one shared direction. The per-layer attack costs KL_1 0.2 to 1.1, above the 0.1 budget.
- In N5, ablating the original `r̂` now costs KL_1 0.61 (base 0.08): the direction carries benign function after training.

**J1 rescore** (`j1_e4`, 17:01 to 17:21 CDT at `47fe696`; padding check labels 16/16 equal, max margin error 0.09 nats). ASR_J1, KL_1 in brackets:

| Arm | M0 base | M1 off-policy | N1 p0.1 | N3 p0.3 | N5 p0.5 |
|---|---|---|---|---|---|
| no attack | 0.14 | 0.01 | 0.07 | 0.02 | 0.01 |
| prefill | 0.24 | 0.42 | 0.00 | 0.00 | 0.00 |
| original `r̂` ablated | 0.75 (0.08) | 0.16 (0.06) | 0.80 (0.10) | 0.74 (0.20) | 0.36 (0.61) |
| fresh direction | 0.75 (0.08) | 0.27 (0.05) | 0.81 (0.10) | 0.34 (0.05) | 0.16 (0.19) |
| fresh per-layer subspace k = 1 | 0.85 (0.55) | 0.49 (0.20) | 0.83 (0.37) | 0.84 (0.46) | 0.83 (1.07) |
| k = 2 | 0.80 (0.91) | 0.43 (0.30) | 0.82 (0.54) | 0.75 (0.62) | 0.80 (1.30) |
| k = 4 | 0.77 (1.88) | 0.41 (0.48) | 0.84 (0.93) | 0.78 (1.02) | 0.80 (1.49) |
| k = 8 | 0.58 (2.19) | 0.29 (0.62) | 0.64 (2.11) | 0.61 (2.13) | 0.61 (2.38) |

J1 keeps the H-P ordering: fresh-direction ASR_J1 is N1 0.81 ≥ N3 0.34 ≥ N5 0.16. It makes the per-layer finding stronger: on-policy DR at every p gives no protection against the k = 1 per-layer subspace (0.83 to 0.84, base 0.85). Off-policy M1 holds that attack to 0.49, together with 0.99 over-refusal. J1 and J2 agree on 0.87 to 1.00 of rows where models refuse, and on 0.36 to 0.63 of rows on ablated arms (see the E0 amendment of 2026-10-07 evening).

## Verdict

failure (H-OP falsified). On-policy responses do not remove DR-1B's over-refusal: N5 refuses 91% of OR-Bench-Hard prompts, past the 0.80 falsification line. They do remove the prefill regression completely and keep MMLU-proxy capability. H-P (the p frontier) held on J2 and on the J1 recheck.

## Updated hypothesis

Two separate causes produced E3's regressions. The off-policy Llama-3 refusal text caused the prefill weakness. The refusal training set (2,000 CircuitBreaker prompts with refusals, α = 0.2, per-sample sum loss) causes the over-refusal, and higher p adds to it. At 1B, DR's measured robustness against a single direction and its over-refusal move together along p. The real attacker cost is set by the per-layer subspace attack, which breaks every variant at a moderate KL cost.

Next:
1. Reduce over-refusal at fixed p: add on-policy compliant OR-Bench-style borderline prompts to the benign set, or raise the benign share. That is a new planned experiment.
2. Treat the per-layer subspace attack under a KL budget as the main robustness metric: an ASR-vs-KL_1 curve per model, not a single point.

## Related

- [E3](2026-10-07-e3-adaptive-and-path-robust-dr-1b.md): the off-policy results that motivate this run.
- [Postmortem: Gemma 3 SDPA left padding](../../../.agents/postmortem/2026-10-07-gemma3-sdpa-left-padding.md).
