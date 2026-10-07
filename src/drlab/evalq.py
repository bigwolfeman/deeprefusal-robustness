"""End-of-run quick evaluation. Keyword refusal only: a morning signal, not a verdict
(constitution Principle 4). Full benchmarks use the E0 protocol with judges.

Generations go to `<run_dir>/eval_artifacts/` (local only, never uploaded; Principle 8).
Metrics go to `<ckpt_dir>/quick_eval.json` (uploaded with the checkpoint).
"""

from __future__ import annotations

import random
from pathlib import Path

import torch

from drlab import prompts as P
from drlab.directions import find_refusal_tokens, select_direction
from drlab.hooks import Ablator
from drlab.textgen import generate, is_refusal, refusal_rate
from drlab.utils import save_json, save_jsonl


def _sample(xs: list, n: int, seed: int) -> list:
    xs = list(xs)
    random.Random(seed).shuffle(xs)
    return xs[:n]


@torch.no_grad()
def quick_eval(model, tok, tpl, ablator: Ablator, dpack: dict, cfg, ckpt_dir: Path) -> dict:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    base.set_attn_implementation("eager")  # SDPA corrupts left-padded Gemma 3 rows (see arch.load_model)
    model.eval()
    n, mnt = cfg.eval.n, cfg.eval.max_new_tokens
    hb = _sample(P.harmbench_standard(), n, 0)
    adv = _sample(P.advbench(), n, 0)
    harmless = _sample(P.harmless_alpaca("test"), n, 0)
    trained_bases = [b.clone() for b in ablator.bases]
    rows, res = [], {}

    def run(name: str, prompts: list[str], prefills=None):
        outs = generate(model, tok, tpl, prompts, mnt, prefills=prefills)
        res[f"refusal/{name}"] = refusal_rate(outs)
        rows.extend(dict(cond=name, prompt=p, out=o, refusal=is_refusal(o)) for p, o in zip(prompts, outs))
        print(f"[quick_eval] {name}: refusal={res[f'refusal/{name}']:.3f}", flush=True)

    ablator.off()
    run("harmbench_none", hb)
    run("harmless_none", harmless)
    run("advbench_prefill", [a for a, _ in adv], prefills=[t for _, t in adv])

    ablator.set_shared_basis(dpack["direction"].float())
    ablator.full()
    run("harmbench_ablate_orig_dir", hb)
    ablator.off()

    if max(b.shape[0] for b in trained_bases) > 1:
        ablator.set_layer_bases(trained_bases)
        ablator.full()
        run("harmbench_ablate_trained_subspace", hb)
        ablator.off()

    # Attacker's view: re-extract a direction on the trained model (same E0 recipe, smaller sets).
    fit_h, val_h, _ = P.split_fit_val_test(P.harmful_behaviors("train"), 128, 32, 0, 0)
    fit_b, val_b, _ = P.split_fit_val_test(P.harmless_alpaca("train"), 128, 32, 0, 0)
    try:
        ref_ids = find_refusal_tokens(model, tok, tpl, fit_h, fit_b)
    except RuntimeError as e:  # a model that never refuses has no refusal tokens; record, don't hide
        res["fresh_direction_error"] = str(e)
        ref_ids = None
    if ref_ids is not None:
        sel = select_direction(model, tok, tpl, ablator, fit_h, fit_b, val_h, val_b, ref_ids,
                               zero_dims=dpack.get("zero_dims"))
        if sel["best"] is None:
            res["fresh_direction_error"] = "no candidate passed induce>0 and KL<0.1"
        else:
            res["fresh_direction_layer"] = sel["best"]["layer"]
            res["fresh_direction_kl"] = sel["best"]["kl"]
            res["fresh_cos_with_orig"] = torch.nn.functional.cosine_similarity(
                sel["direction"].float().cpu(), dpack["direction"].float().cpu(), dim=0).item()
            ablator.set_shared_basis(sel["direction"].float())
            ablator.full()
            run("harmbench_ablate_fresh_dir", hb)
            ablator.off()

    ablator.set_layer_bases(trained_bases)
    save_json(res, Path(ckpt_dir) / "quick_eval.json")
    save_jsonl(rows, Path(ckpt_dir).parent / "eval_artifacts" / "quick_eval_generations.jsonl")
    return res
