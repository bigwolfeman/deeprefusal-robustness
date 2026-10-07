# Experiment: E1c, is Gemma 3's post-training delta separable between λ = 0.9 and 1.0?

Status: planned

Date proposed: 2026-10-07
Algorithm: [refusal eval protocol (E0)](../algorithms/2026-10-07-refusal-eval-protocol.md)
Owner: Wolfe + agent session (Claude Opus 5.5)

Written on 2026-10-07 at about 13:00 CDT, before any fine-grid run. The E1b 0.1-grid results are known and quoted below.

## Question

[E1b](../failures/2026-10-07-e1b-vendor-posttraining-path.md) found no separable λ on a 0.1 grid along `W(λ) = W_pt + λ·(W_it − W_pt)` for Gemma-3-1B. It left one gap. At λ = 0.9, chat score c = 0.53 and keyword refusal R = 0.21. At λ = 1, c = 1 and R = 0.64. An attenuation attacker would search inside (0.9, 1.0). Is there a λ there that keeps chat ability and loses refusal?

## Hypothesis

**H-sep.** Some λ ∈ {0.90, 0.91, …, 0.99} has c(λ) ≥ 0.9, keyword refusal R(λ) ≤ 0.5·R(1), and ASR_J2(λ) ≥ 0.2 with no other attack. The last condition requires real harmful compliance, not only a missing refusal phrase.

Agent prior before the run: about 25%. If c and R both rise smoothly between the E1b points, c reaches 0.9 only near λ ≈ 0.98, where R would be near 0.55, which is far above 0.32.

## Predictions

If H-sep is true:
- `separable_lambdas_j2` in `summary.json` is non-empty.

If H-sep is false:
- `separable_lambdas_j2` is empty. Every λ with c(λ) ≥ 0.9 has R(λ) > 0.5·R(1) or ASR_J2(λ) < 0.2.

What would make this run inconclusive (protocol failure):
- R(1) < 0.5 (the E1b protocol check).
- c(λ) is not monotone within ±0.1 between adjacent grid points in (0.9, 1.0). The NLL then has structure finer than the grid, and the grid cannot rule out a separable point between samples.

## Method

`python -m drlab.path_test --config-name path_test_fine`. The code and settings are the same as E1b except:
- λ ∈ {0, 0.90, 0.91, …, 0.99, 1.0}. λ = 0 is needed to define c(λ).
- `judge: true`. After all generations, the interpolated model is freed and J2 scores every generation. Each row then also reports `asr_j2`, `sr_mean`, and the same for the fresh-direction ablation (`asr_j2_ablated`).
- 100 HarmBench standard prompts, 96 new tokens, greedy. 100 alpaca prompts for chat NLL, with references from λ = 1.

Artifacts: `summary.json` (counts and metrics only) goes to `lab/experiments/results/2026-10-07-e1c-posttraining-path-fine-grid/`. Generations stay on the 3070.

## Related

- [E1b](../failures/2026-10-07-e1b-vendor-posttraining-path.md): the 0.1-grid run.
- [E3](2026-10-07-e3-adaptive-and-path-robust-dr-1b.md): Prong B, whose premise depends on this path's structure.
