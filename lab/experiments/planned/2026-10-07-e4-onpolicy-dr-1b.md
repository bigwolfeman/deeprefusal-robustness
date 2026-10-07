# Experiment: E4, does on-policy training data remove DeepRefusal-1B's over-refusal and prefill regression?

Status: planned

Date proposed: 2026-10-07
Algorithm: [refusal eval protocol (E0)](../algorithms/2026-10-07-refusal-eval-protocol.md); data design in [on-policy DeepRefusal data note](../../../.agents/notes/proposed/feature/2026-10-07-onpolicy-dr-data.md)
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

## Related

- [E3](2026-10-07-e3-adaptive-and-path-robust-dr-1b.md): the off-policy results that motivate this run.
- [Postmortem: Gemma 3 SDPA left padding](../../../.agents/postmortem/2026-10-07-gemma3-sdpa-left-padding.md).
