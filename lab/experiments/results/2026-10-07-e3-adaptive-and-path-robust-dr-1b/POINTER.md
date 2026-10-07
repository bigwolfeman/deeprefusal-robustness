# E3 bench artifacts (preliminary, J2 only)

- `*_metrics.json`: per-model condition metrics from `python -m drlab.bench` at commit `9c953a6` (3070, 2026-10-07 10:54 to 11:31 CDT). No completion text.
- Generations and per-item scores (they contain harmful text) stay on the 3070 at `/mnt/bigdata/deeprefusal/bench/e3/artifacts/`.
- Checkpoints: HF `bigwolfe/deeprefusal-robustness-ckpts/<run>/final`.
- The E3 verdict waits for J1 (HarmBench classifier) and lm-eval-harness capability, as the plan requires.
