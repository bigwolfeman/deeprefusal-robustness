# Agent Note: On-policy responses for DeepRefusal training data

Status: implemented

## Problem

The DeepRefusal recipe (paper section 5.1, official `train_dataset.py`) takes its training responses from other models. CircuitBreaker refusals are Llama-3-8B-Instruct outputs (`llama3_output`). UltraChat answers and XSTest completions come from GPT models. For the paper's Llama-3 the refusals are on-policy. For Gemma-3-1B none of the responses are.

In E3 the faithful recipe on Gemma-3-1B kept refusal under direction ablation, but it moved the model far from its own behavior. OR-Bench-Hard keyword refusal went from 0.32 to 0.99. Prefill ASR_J2 went from 0.14 to 0.48: Gemma's own hedging was replaced by Llama-style refusals, so when the refusal does not fire, the model complies plainly. A defense that changes a model's voice this much cannot be compared fairly with the undefended model, and part of its measured robustness may be over-refusal.

## Decision

`data.responses: onpolicy` keeps the recipe's prompts, prefixes, counts, and random draws, and every response comes from the model being defended (`src/drlab/onpolicy.py`, `configs/onpolicy.yaml`). `data.responses: offpolicy` remains the default (paper recipe).

- `drlab.data.select_sources` owns the prompt selection and the RNG draw order for both builds. The off-policy build stays byte-identical to the E3 `dr_baseline` data (verified: 8,000 of 8,000 examples have equal token ids and labels).
- Benign rows: the model's greedy answer. A keyword refusal is redone with `r̂` ablated at every site, which gives the model's own compliant voice. If that also refuses, the row is dropped.
- Harmful rows: the first candidate that J2 scores below 0.1, in this order: greedy, 2 samples at T = 0.7, greedy with `r̂` added at the direction layer.
- Prefix rows: the model's continuation after the harmful prefix. A continuation must be an explicit keyword refusal and J2-safe. Otherwise the prompt's accepted refusal follows the prefix, as in the paper.
- A response cut at its token limit gets no end-of-turn target.
- `data.onpolicy_match_length: true` caps each response at the token count of the off-policy response it replaces. Gemma-3-1B-it writes 2 to 3 times longer answers, and the DR loss sums over tokens per sample, so uncapped data would also change each row's loss weight.
- The generator runs in four phases (model, J2, model, J2), so the 8 GB 3070 never holds both.

## Alternatives considered

- **Keep the off-policy recipe.** It is faithful to the paper, but on Gemma it changes two variables at once (the defense and the response distribution). E3 cannot separate them.
- **A system prompt that asks the model to refuse.** It is cheaper than steering, but Gemma 3 has no system role (the text goes into the user turn), so the responses come from a different prompt distribution.
- **Keyword-only acceptance for harmful rows.** No judge is needed, but it rejects Gemma's safe redirections ("please contact a crisis line") and accepts disclaimers followed by compliance. J2 measures harm, which is the property we want.
- **vLLM for generation.** It is about 10× faster, but it needs a second venv (its torch pin conflicts with torch 2.11), and steering needs HF hooks anyway. The HF path is about 1.5 h on the 3070. vLLM becomes worth it for the Prong B distillation set (about 20k responses).

## Consequences

- E4 ([failures/2026-10-07-e4-onpolicy-dr-1b.md](../../../../lab/experiments/failures/2026-10-07-e4-onpolicy-dr-1b.md)): generation dropped 0.05% of benign and 0% of harmful prompts. On-policy DR-1B has no prefill weakness (ASR_J2 0.00 vs 0.46 off-policy) and keeps the MMLU proxy, but over-refusal stays high (OR-Bench-Hard 0.91 at p = 0.5). Over-refusal comes from the refusal training set, not the response voice.
- Generation takes about 70 min on the 3070 (HF eager). The Prong B distillation set (about 20k responses) needs vLLM.

Known limits:

- **Judge reuse.** J2 selects the harmful targets and also scores the attacks. Selection only admits J2-safe text, but a style bias toward what J2 calls safe is possible. J1 rechecks the E4 cells (`python -m drlab.rejudge`).
- **Steered text is not fully on-policy.** Rows from `steered` and `ablated` come from an edited model. `summary.json` counts each source, so their share is visible.
- **Self-distillation weakens the benign loss.** The benign target is the model's own greedy output, so the benign loss mostly holds the model in place. That is the intent, but it also removes any capability gain the GPT answers gave.
