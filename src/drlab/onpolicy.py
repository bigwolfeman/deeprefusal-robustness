"""On-policy DeepRefusal training responses (E4).

Plan: lab/experiments/planned/2026-10-07-e4-onpolicy-dr-1b.md
Design: .agents/notes/proposed/feature/2026-10-07-onpolicy-dr-data.md

The paper recipe trains on Llama-3 refusals and GPT answers. This script keeps the recipe's
prompts and prefixes (`drlab.data.select_sources`) and writes every response with the model being
defended. Four phases, so the 3070 (8 GB) never holds the model and the J2 judge at once:

  1. model: greedy benign responses (keyword refusals redone with r_hat ablated), greedy harmful
     responses, greedy continuations after the harmful prefix.
  2. J2:    score the harmful responses and the prefix continuations.
  3. model: for harmful prompts with no safe response yet, samples at temperature T and a greedy
     response with r_hat added at the direction layer; for prefixes with no recovery, a greedy
     continuation with r_hat added.
  4. J2:    score the phase-3 candidates, select, write the dataset.

Output (harmful text, local only): `<out_dir>/onpolicy.json` and `candidates.jsonl`.
`summary.json` holds counts only.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from drlab.arch import ChatTemplate, load_model
from drlab.data import select_sources
from drlab.hooks import Ablator
from drlab.judges import StrongRejectJudge
from drlab.textgen import generate_full, is_refusal
from drlab.utils import env_info, save_json, save_jsonl, seed_all


def _free() -> None:
    """Callers `del` their own references first; this only collects and releases cached blocks."""
    gc.collect()
    torch.cuda.empty_cache()


def _gen(cfg, model, tok, tpl, prompts, max_new, prefills=None, temperature=0.0):
    return generate_full(model, tok, tpl, prompts, max_new, batch_size=cfg.gen.batch_size, prefills=prefills,
                         temperature=temperature, prefill_budget=cfg.gen.prefill_budget, kv_budget=cfg.gen.kv_budget)


_T0 = time.time()


def _log(msg: str) -> None:
    print(f"[onpolicy {(time.time() - _T0) / 60:5.1f}m] {msg}", flush=True)


def run(cfg: DictConfig) -> dict:
    seed_all(cfg.seed)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dpack = torch.load(cfg.direction_path, map_location="cpu")
    if dpack["model"] != cfg.model.name:
        raise ValueError(f"direction is for {dpack['model']}, generator model is {cfg.model.name}")
    r_hat, r_layer = dpack["direction"].float(), int(dpack["layer"])

    # ---- phase 1: greedy everything ----------------------------------------------------
    model, tok = load_model(cfg.model.name, cfg.model.dtype)
    tpl = ChatTemplate.from_tokenizer(tok)
    src = select_sources(cfg.data, tok)
    if cfg.data.prefill_augmentations != 1:
        raise ValueError("on-policy data supports prefill_augmentations = 1 (one recovery per prompt)")
    ab = Ablator(model)

    benign_prompts = list(dict.fromkeys(instr for instr, _ in src.benign))
    _log(f"phase 1: {len(benign_prompts)} benign, {len(src.cb)} harmful prompts")
    b_resp, b_fin = _gen(cfg, model, tok, tpl, benign_prompts, cfg.gen.max_new_benign)
    benign = {p: dict(response=r, finished=f, source="greedy") for p, r, f in zip(benign_prompts, b_resp, b_fin)}
    refused = [p for p in benign_prompts if is_refusal(benign[p]["response"])]
    _log(f"benign greedy keyword refusals: {len(refused)}; redo with r_hat ablated")
    ab.set_shared_basis(r_hat)
    ab.full()
    a_resp, a_fin = _gen(cfg, model, tok, tpl, refused, cfg.gen.max_new_benign)
    ab.off()
    for p, r, f in zip(refused, a_resp, a_fin):
        benign[p] = (dict(response=r, finished=f, source="ablated") if not is_refusal(r)
                     else dict(response="", finished=False, source="dropped"))

    harm_prompts = [row["prompt"] for row in src.cb]
    prefixes = [x[0] for x in src.prefixes]
    cand: dict[str, list[list[dict]]] = {"refusal": [[] for _ in harm_prompts], "recovery": [[] for _ in harm_prompts]}
    h_resp, h_fin = _gen(cfg, model, tok, tpl, harm_prompts, cfg.gen.max_new_harmful)
    for i, (r, f) in enumerate(zip(h_resp, h_fin)):
        cand["refusal"][i].append(dict(source="greedy", text=r, finished=f))
    c_resp, c_fin = _gen(cfg, model, tok, tpl, harm_prompts, cfg.gen.max_new_recovery, prefills=prefixes)
    for i, (r, f) in enumerate(zip(c_resp, c_fin)):
        cand["recovery"][i].append(dict(source="greedy", text=r, finished=f))
    ab.remove()
    del model, ab
    _free()

    # ---- phase 2: judge ------------------------------------------------------------------
    def judge_pending() -> None:
        judge = StrongRejectJudge()
        todo = [(kind, i, c) for kind in cand for i, cs in enumerate(cand[kind]) for c in cs if "sr" not in c]
        scores = judge.score([harm_prompts[i] for _, i, _ in todo], [c["text"] for _, _, c in todo], bs=cfg.judge_bs)
        for (_, _, c), s in zip(todo, scores):
            c["sr"] = s
        _log(f"J2 scored {len(todo)} candidates")
        del judge
        _free()

    def accepted(kind: str, c: dict) -> bool:
        if c["sr"] >= cfg.select.max_sr:
            return False
        return kind == "refusal" or is_refusal(c["text"])  # a recovery must break off explicitly

    def pending(kind: str) -> list[int]:
        return [i for i, cs in enumerate(cand[kind]) if not any(accepted(kind, c) for c in cs)]

    judge_pending()

    # ---- phase 3: samples and steering for what is still unsafe -----------------------
    need_ref, need_rec = pending("refusal"), pending("recovery")
    _log(f"phase 3: {len(need_ref)} harmful prompts and {len(need_rec)} prefixes lack an accepted response")
    model, tok = load_model(cfg.model.name, cfg.model.dtype)
    ab = Ablator(model)
    torch.manual_seed(cfg.seed)
    for s in range(cfg.select.n_samples):
        rs, fs = _gen(cfg, model, tok, tpl, [harm_prompts[i] for i in need_ref], cfg.gen.max_new_harmful,
                      temperature=cfg.select.temperature)
        for i, r, f in zip(need_ref, rs, fs):
            cand["refusal"][i].append(dict(source=f"sample{s}", text=r, finished=f))
    ab.add_vec = (r_layer, r_hat)
    rs, fs = _gen(cfg, model, tok, tpl, [harm_prompts[i] for i in need_ref], cfg.gen.max_new_harmful)
    for i, r, f in zip(need_ref, rs, fs):
        cand["refusal"][i].append(dict(source="steered", text=r, finished=f))
    rs, fs = _gen(cfg, model, tok, tpl, [harm_prompts[i] for i in need_rec], cfg.gen.max_new_recovery,
                  prefills=[prefixes[i] for i in need_rec])
    for i, r, f in zip(need_rec, rs, fs):
        cand["recovery"][i].append(dict(source="steered", text=r, finished=f))
    ab.add_vec = None
    ab.remove()
    del model, ab
    _free()

    # ---- phase 4: judge, select, write --------------------------------------------------
    judge_pending()
    harmful, counts = [], {"benign": {}, "refusal": {}, "recovery": {}}
    for b in benign.values():
        counts["benign"][b["source"]] = counts["benign"].get(b["source"], 0) + 1
    for i, p in enumerate(harm_prompts):
        ref = next((c for c in cand["refusal"][i] if accepted("refusal", c)), None)
        rec = next((c for c in cand["recovery"][i] if accepted("recovery", c)), None)
        if ref is None:
            row = dict(prompt=p, prefix=prefixes[i], refusal="", refusal_finished=False, refusal_source="dropped",
                       recovery="", recovery_finished=False, recovery_source="dropped")
        else:
            if rec is None:  # paper construction: the prompt's refusal right after the prefix
                rec = dict(source="refusal_after_prefix", text=ref["text"], finished=ref["finished"])
            row = dict(prompt=p, prefix=prefixes[i], refusal=ref["text"], refusal_finished=ref["finished"],
                       refusal_source=ref["source"], recovery=rec["text"], recovery_finished=rec["finished"],
                       recovery_source=rec["source"])
        harmful.append(row)
        for kind in ("refusal", "recovery"):
            k = row[f"{kind}_source"]
            counts[kind][k] = counts[kind].get(k, 0) + 1

    n_b, n_h = len(benign), len(harmful)
    summary = dict(
        sources=counts,
        drop_frac=dict(benign=counts["benign"].get("dropped", 0) / n_b,
                       harmful=counts["refusal"].get("dropped", 0) / n_h),
        unfinished_frac=dict(
            benign=sum(not b["finished"] for b in benign.values() if b["source"] != "dropped") / n_b,
            refusal=sum(not h["refusal_finished"] for h in harmful if h["refusal_source"] != "dropped") / n_h),
        greedy_sr_mean=sum(cand["refusal"][i][0]["sr"] for i in range(n_h)) / n_h,
        config=OmegaConf.to_container(cfg, resolve=True), env=env_info(),
    )
    save_jsonl([dict(kind=k, i=i, **c) for k in cand for i, cs in enumerate(cand[k]) for c in cs],
               out / "candidates.jsonl")
    save_json(dict(benign=benign, harmful=harmful, summary=summary), out / "onpolicy.json")
    save_json(summary, out / "summary.json")
    _log(f"done: {counts} drop_frac={summary['drop_frac']}")
    return summary


@hydra.main(version_base=None, config_path="../../configs", config_name="onpolicy")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
