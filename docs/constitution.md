# Project Constitution

**Status**: Draft. Written by the agent on 2026-10-07 from discussion with Wolfe. It needs review by Wolfe and Gratitude [LABS]. Set to `Ratified` when both agree.

**Version**: 0.1.0
**Ratified**: unfilled
**Last Amended**: 2026-10-07

This file is the long-term track for the project. Standing orders and file layout live in [AGENTS.md](../AGENTS.md). This file says why the project exists and what it must not drift into.

## Vision

Open-weight language models ship with safety training that an attacker can remove cheaply. DeepRefusal (Xie et al., EMNLP 2025 Findings, arXiv 2509.15202) claims to make refusal robust. It trains the model while it randomly ablates the refusal direction across layers and tokens. The authors released `skysys00/Meta-Llama-3-8B-Instruct-DeepRefusal`. Public attacks broke that release. abliterix scales the LoRA delta toward the base model and then abliterates. APS steers activations with per-layer probes.

This project does three things. First, it measures what DeepRefusal actually does inside the model, and it tests whether the paper's Figure 1 ("the model re-enters a refusal state from a jailbreak state") is real. Second, it measures how robust DeepRefusal is under a threat model that can be defended. Third, it extends DeepRefusal so that refusal has no cheap weight-space handle, and it reports the result as an attacker-cost curve, not as a single pass or fail.

The end product is an arXiv paper, public code that reproduces every number from a Hydra config, and public defended checkpoints. "Done" means a reader can rerun every attack against every defense and get our curves.

## Problem

People who release open-weight models cannot make safety training stick. If refusal lives in one linear direction, anyone with the weights removes it in minutes. DeepRefusal answered single-direction attacks, but the paper did not test attacks that remove several directions, attacks that steer with per-layer probes, or attacks that edit weights. Nobody has published a mechanistic account of what DeepRefusal's training changes, so nobody knows why it fails where it fails.

## Threat model

This section is the core of the project. Every claim must name the attack class and the attacker knowledge it assumes.

**The attacker has** the defended weights, the architecture code, unlimited inference, white-box gradients, public datasets, and a stated compute budget.

**The attacker does not have** `W_base`, the checkpoint before the defense was trained. Reason: if `W_base` is public, `W_base + λ·(W_def − W_base)` at λ = 0 returns the undefended model. No post-hoc fine-tune can defend against that attacker, because the attacker already owns an undefended model. abliterix's attenuation step is this case. We use it only as a positive control.

**Attack classes:**

| Class | Examples | Priority |
|---|---|---|
| A. Prompt level | templates, GCG, prefill | baseline coverage |
| B. Activation level | direction ablation hooks, APS probe steering | primary |
| C. Weight edits without training | single-direction and iterative subspace abliteration | primary |
| D. Fine-tuning | harmful LoRA or full fine-tune | secondary (SEAM arm) |

**Fine-tuning is the floor.** With enough harmful data and compute, the attacker wins. All class D claims are about attacker cost, not impossibility.

## Users and operators

- Researchers who release or evaluate open-weight safety methods. They use our attacks, curves, and code.
- Wolfe and Gratitude [LABS] operate the experiments.
- Harm if we are wrong: a false robustness claim makes people trust a defense that fails. That is the failure this project exists to catch. We must not repeat it.

## Scope

**In scope**

- Mechanistic analysis of DeepRefusal on the official 8B release: the Figure 1 test, the rebuild-versus-redundancy question, and the structure of the weight delta.
- Attack evaluation of DeepRefusal under the threat model above, classes A to C, with adaptive attacks.
- A correct small-model DeepRefusal reproduction (1B class), with a validated refusal direction and real judges.
- An extended defense that targets the measured failure (see [adaptive DeepRefusal note](../.agents/notes/proposed/feature/2026-10-07-adaptive-deeprefusal.md)).
- A SEAM arm for class D, after classes B and C have results.

**Out of scope (non-goals)**

- Defending against an attacker who has `W_base`. It is impossible by construction (see Threat model).
- Defenses that the attacker can delete in code, such as Robust Self-Attention or inference-time filters. Open weights include the code, so a parameter-free module is not a defense.
- Closed-model or API-side defenses.
- Publishing uncensored ("broken") checkpoints. We publish defenses and aggregate metrics only.
- Claims based on keyword refusal matching.

## Success criteria

1. E1 has a verdict: does weight-space abliteration without `W_base` break DeepRefusal-8B? ([planned](../lab/experiments/planned/2026-10-07-e1-abliteration-without-base.md))
2. E2 has a verdict: is Figure 1d real, and does DeepRefusal rebuild `r̂` or move refusal to new directions? ([planned](../lab/experiments/planned/2026-10-07-e2-refusal-rebuild-geometry.md))
3. A defense moves the attacker-cost curve for the class that broke DeepRefusal. The exact margin is in the [adaptive DeepRefusal note](../.agents/notes/proposed/feature/2026-10-07-adaptive-deeprefusal.md).
4. Every reported number comes from a committed Hydra config and a wandb run that logs the full config.
5. The paper is on arXiv.

## Constraints

- **Hardware.** RTX 5090 (32 GB), shared with other projects. The UPS trips if the GPU is at full load while the CPU is also loaded, so keep the GPU near 80% when CPU work runs at the same time. DGX Spark (128 GB unified memory) via `ssh dgx-spark`. Kaggle: T4×2 or P100, about 30 GPU-hours per week per account, no bf16 on T4.
- **Tooling.** Python with a `uv` venv. Hydra for every run config. wandb for every run, logging the full config, including constructor defaults. bf16 where the hardware supports it.
- **Licenses.** Llama 3 Community License for Llama checkpoints. heretic and abliterix are AGPL-3.0. We run them as external tools from `ignored/refs/`. We do not vendor or import their code into `src/`, because that would put this repository under AGPL.
- **Public repository.** The repo is public. Harmful model generations never get committed.

## Principles

### Principle 1: Threat model first

**The rule**: Every result names its attack class (A to D) and states whether the attacker used `W_base`.

**Why**: The DeepRefusal paper and abliterix argued past each other because neither stated the attacker's knowledge.

**How to apply**: Experiment files have a threat-model line in Method. Tables have a column for attacker knowledge.

### Principle 2: Adaptive attacks or no claim

**The rule**: A defense is robust only against attacks that were tuned against that defense.

**Why**: The DeepRefusal authors reported heretic as failing. A tool adapted to their release broke it. Fixed attack pipelines produce false robustness claims.

**How to apply**: For each defense, run at least one attack that uses knowledge of the defense, such as its training directions or its subspace. Report the strongest attack, not the average.

### Principle 3: Validate a direction before using it

**The rule**: A refusal direction is used only after it passes both Arditi tests. Adding it induces refusal on harmless prompts. Ablating it removes refusal on harmful prompts.

**Why**: The ECS-189G reproduction trained DeepRefusal against a direction that failed both tests: 0/50 refusal when added at α = 2, and no steady drop in refusal when ablated.

**How to apply**: The [eval protocol](../lab/experiments/algorithms/2026-10-07-refusal-eval-protocol.md) defines the tests and thresholds. A run that skips them is a protocol failure.

### Principle 4: Real judges, reported with agreement

**The rule**: Attack success comes from classifier judges, not keyword matching. Report two judges and their agreement.

**Why**: Keyword judges miss compliant answers and count gibberish. abliterix found its own detector inflated refusal counts by about 33%.

**How to apply**: See the [eval protocol](../lab/experiments/algorithms/2026-10-07-refusal-eval-protocol.md). Keyword matching is a diagnostic only.

### Principle 5: Curves, not points

**The rule**: Report attack success as a function of attacker budget (KL, rank, steps), next to capability metrics.

**Why**: One attack setting hides whether the defense raised the cost or only moved the optimum.

**How to apply**: Each attack arm sweeps its budget. Each plot shows ASR against KL with MMLU change.

### Principle 6: Released checkpoints before our own

**The rule**: Mechanistic and attack claims about DeepRefusal are first measured on the authors' release.

**Why**: A failure on our own reproduction could come from our training, not from the method.

**How to apply**: E1 and E2 use the official 8B release. Small-model training starts after them.

### Principle 7: Capability metrics must be off the floor

**The rule**: Use only capability benchmarks where the model scores clearly above chance.

**Why**: GPQA Diamond is near or below chance for 1B models, so it cannot detect damage.

**How to apply**: Use MMLU, GSM8K, ARC-Challenge, and benign-prompt KL. Check the undefended score against chance before adding a benchmark.

### Principle 8: No harmful text in git

**The rule**: Raw generations for harmful prompts live in `ignored/experiment-artifacts/`. `lab/experiments/results/` holds aggregate metrics and pointer files.

**Why**: The repository is public.

**How to apply**: Result writers save judge scores and IDs, not completion text. Qualitative examples in the paper are redacted.

## Architecture sketch

Nothing is built yet. The intended layout:

- `src/` holds one Python package with these parts: direction extraction and validation, activation hooks (ablate, add, steer), judges, evals (ASR, KL, capability), our own attack implementations (subspace abliteration), and training (DeepRefusal reproduction, adaptive variant).
- `configs/` holds Hydra configs. Every run reads one.
- External tools (official DeepRefusal, heretic, abliterix, andyrdt/refusal_direction) live in `ignored/refs/` and run as subprocesses or separate venvs.
- `lab/experiments/` holds plans and verdicts. The shared measurement rules live in the [eval protocol](../lab/experiments/algorithms/2026-10-07-refusal-eval-protocol.md).

Plan and dependencies:

```
E0 eval protocol (directions, judges, KL, capability)   ← every arm depends on it
├── E1 abliteration without W_base on DR-8B             (class C, no training)
├── E2 rebuild geometry / Figure 1 test on DR-8B        (mechanism, no training)
└── E3 adaptive DeepRefusal at 1B                       (training; design depends on E1+E2)
    └── E4 SEAM arm, class D fine-tuning                (after E3 has a defense)
```

## When this constitution applies

- All new work that could change what the project is.
- Features, refactors, and promotions into production.

**Override**: only with explicit user authorization and a one-line rationale recorded in the change. If the vision changes, amend this file with the user.

## Governance

Amend with Wolfe (and Gratitude for research direction) when the vision or a principle is wrong. Bump:

- MAJOR: the project is a different thing.
- MINOR: new principle or new in-scope capability.
- PATCH: clarification.

Record substantial amendments in an Agent Note under `.agents/notes/`.
