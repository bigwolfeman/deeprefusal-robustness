# Postmortem: Gemma 3 with SDPA attention corrupts left-padded rows (transformers 5.19)

Date: 2026-10-07

## Executive summary

The first E0 direction run on `gemma-3-1b-it` reported a mean harmful refusal score of −17.1 for a model that refuses about 82% of harmful prompts. Root cause: with `attn_implementation="sdpa"` in transformers 5.19.0 (torch 2.11, RTX 3070), every left-padded row in a Gemma 3 batch returns the same wrong next-token distribution. Only the unpadded (longest) row is correct. Explicit `position_ids` do not fix it. Eager attention does. It escaped because the first sanity batch happened to have little padding and looked fine. The durable lesson: check batched against single-row outputs on the exact padding pattern a function uses, every time the stack changes.

## Timeline

- 01:16 CDT: first E0 run fails the Arditi filters. The table shows harmful score −17 and first-token KL of about 20 nats for most candidates.
- 01:20: a debug batch (the first 32 prompts) gives correct scores at batch sizes 1 and 32. That batch had little padding.
- 01:22: the validation batch gives exactly −17.5 for 31 of 32 rows. Batch size 1 on the same prompts is correct.
- 01:23: eager attention gives correct scores, with or without explicit `position_ids`. SDPA is wrong in both cases.

## What broke

`drlab.directions.last_logprobs` and `mean_eoi_acts` with left padding. Any baseline, KL, or bypass score from a padded batch was wrong.

## Root cause

An interaction of Gemma 3's mask construction (sliding-window and global layers) with the SDPA path for left padding in transformers 5.19. We did not trace it further into transformers. The behavior is reproducible with `/mnt/bigdata/deeprefusal/tmp/dbg_dir4.py` on the 3070.

Right padding (used in training) is unaffected. Its batched-versus-single per-position KL (max 0.071) is at the bf16 noise floor of eager attention (max 0.107).

## Why safety nets missed it

Nothing compared batched outputs against single-row outputs on the padding pattern actually used. The first spot check used a batch with too little padding to show the bug.

## Guardrails added

- `drlab.arch.load_model` defaults to eager attention. `encode_prompts` returns explicit `position_ids`.
- `drlab.evalq.quick_eval` switches the trained model to eager before any left-padded generation.
- `drlab.train.refit_subspace` (Prong A) runs on the SDPA training model, so it uses batch size 1 (no padding). This was found while writing this postmortem, before the `prong_a` job started. No Prong A refit ran with padded batches.
