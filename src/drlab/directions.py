"""Refusal-direction extraction, selection, and validation gate (E0, section 1).

Method: Arditi et al. 2024 (github.com/andyrdt/refusal_direction). Protocol and thresholds:
lab/experiments/algorithms/2026-10-07-refusal-eval-protocol.md
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from drlab import prompts as P
from drlab.arch import ChatTemplate, eoi_token_count, encode_prompts, get_layers, load_model
from drlab.hooks import Ablator
from drlab.textgen import generate, is_refusal, refusal_rate
from drlab.utils import env_info, save_json, save_jsonl, seed_all


@torch.no_grad()
def mean_eoi_acts(model, tok, tpl, prompts: list[str], n_eoi: int, bs: int = 32) -> torch.Tensor:
    """Mean residual-stream input of every layer at the last `n_eoi` prompt positions: [n_eoi, L, d]."""
    L = len(get_layers(model))
    total = None
    for i in range(0, len(prompts), bs):
        enc = encode_prompts(tok, tpl, prompts[i:i + bs], model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states  # L+1 entries; [l] = input to layer l
        acts = torch.stack([hs[l][:, -n_eoi:, :].float() for l in range(L)], dim=2)  # [B, n_eoi, L, d]
        s = acts.sum(0)
        total = s if total is None else total + s
    return total / len(prompts)


@torch.no_grad()
def outlier_dims(model, tok, tpl, prompts: list[str], ratio: float, min_layers: int, bs: int = 32) -> list[int]:
    """Massive-activation dimensions: mean |h| over all prompt tokens above `ratio` x the layer's
    median dimension, in at least `min_layers` layers. Gemma 3 has such dims (e.g. 1038 in the 1B
    model). A raw mean difference can be dominated by them (cos > 0.7), and ablating it breaks the
    model (first-token KL > 20 nats). Candidates are therefore zeroed on these dims.
    Postmortem: .agents/postmortem/2026-10-07-gemma3-outlier-dims-refusal-direction.md"""
    L = len(get_layers(model))
    acc, cnt = None, 0
    for i in range(0, len(prompts), bs):
        enc = encode_prompts(tok, tpl, prompts[i:i + bs], model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        m = enc["attention_mask"].bool()
        a = torch.stack([hs[l][m].float().abs().sum(0) for l in range(L)])
        acc = a if acc is None else acc + a
        cnt += int(m.sum())
    mabs = acc / cnt
    flagged = (mabs > ratio * mabs.median(dim=1, keepdim=True).values).sum(0)
    return (flagged >= min_layers).nonzero().flatten().tolist()


@torch.no_grad()
def last_logprobs(model, tok, tpl, prompts: list[str], bs: int = 32) -> torch.Tensor:
    out = []
    for i in range(0, len(prompts), bs):
        enc = encode_prompts(tok, tpl, prompts[i:i + bs], model.device)
        out.append(F.log_softmax(model(**enc).logits[:, -1, :].float(), dim=-1))
    return torch.cat(out)


def refusal_score(logprobs: torch.Tensor, refusal_ids: list[int]) -> torch.Tensor:
    """Arditi refusal score: log(p / (1 - p)), p = total probability of refusal-start tokens."""
    lp = torch.logsumexp(logprobs[:, refusal_ids], dim=-1)
    return lp - torch.log1p(-lp.exp().clamp(max=1 - 1e-6))


@torch.no_grad()
def find_refusal_tokens(model, tok, tpl, harmful: list[str], harmless: list[str],
                        mass: float = 0.8, max_harmless_frac: float = 0.05) -> list[int]:
    """First greedy tokens that start refusals: frequent on harmful prompts, rare on harmless."""
    fh = Counter(last_logprobs(model, tok, tpl, harmful).argmax(-1).tolist())
    fb = Counter(last_logprobs(model, tok, tpl, harmless).argmax(-1).tolist())
    ids, covered = [], 0
    for t, c in fh.most_common():
        if fb.get(t, 0) / len(harmless) > max_harmless_frac:
            continue
        ids.append(t)
        covered += c
        if covered / len(harmful) >= mass:
            break
    if not ids:
        raise RuntimeError("No refusal-start tokens found: model may not refuse harmful prompts at all")
    return ids


@torch.no_grad()
def select_direction(model, tok, tpl, ablator: Ablator, fit_h, fit_b, val_h, val_b, refusal_ids,
                     max_layer_frac: float = 0.8, kl_max: float = 0.1, zero_dims: list[int] | None = None) -> dict:
    n_eoi = eoi_token_count(tok, tpl)
    L = ablator.n_layers
    cands = mean_eoi_acts(model, tok, tpl, fit_h, n_eoi) - mean_eoi_acts(model, tok, tpl, fit_b, n_eoi)
    if zero_dims:
        cands[..., zero_dims] = 0
    ablator.off()
    base_b = last_logprobs(model, tok, tpl, val_b)
    base_h_score = refusal_score(last_logprobs(model, tok, tpl, val_h), refusal_ids).mean().item()
    base_b_score = refusal_score(base_b, refusal_ids).mean().item()
    rows = []
    for i in range(n_eoi):
        for l in range(int(max_layer_frac * L)):
            r = cands[i, l]
            ablator.set_shared_basis(r)
            ablator.full()
            bypass = refusal_score(last_logprobs(model, tok, tpl, val_h), refusal_ids).mean().item()
            kl = F.kl_div(last_logprobs(model, tok, tpl, val_b), base_b, reduction="batchmean",
                          log_target=True).item()
            ablator.off()
            ablator.add_vec = (l, r)
            induce = refusal_score(last_logprobs(model, tok, tpl, val_b), refusal_ids).mean().item()
            ablator.add_vec = None
            rows.append(dict(pos=i - n_eoi, layer=l, bypass=bypass, induce=induce, kl=kl,
                             norm=r.norm().item()))
    ok = [x for x in rows if x["induce"] > 0 and x["kl"] < kl_max]
    best = min(ok, key=lambda x: x["bypass"]) if ok else None
    r = cands[best["pos"] + n_eoi, best["layer"]].clone() if ok else None
    return dict(best=best, direction=r, candidates=rows, n_eoi=n_eoi,
                base_harmful_score=base_h_score, base_harmless_score=base_b_score)


@torch.no_grad()
def gate_direction(model, tok, tpl, ablator: Ablator, r: torch.Tensor, layer: int, test_h, test_b,
                   max_new_tokens: int, min_induce: float, min_drop: float) -> tuple[dict, list[dict]]:
    """E0 validation gate with keyword refusal (deviation: judges are not on the 3070)."""
    ablator.off()
    g_h = generate(model, tok, tpl, test_h, max_new_tokens)
    g_b = generate(model, tok, tpl, test_b, max_new_tokens)
    ablator.set_shared_basis(r)
    ablator.full()
    g_h_abl = generate(model, tok, tpl, test_h, max_new_tokens)
    ablator.off()
    ablator.add_vec = (layer, r)
    g_b_add = generate(model, tok, tpl, test_b, max_new_tokens)
    ablator.add_vec = None
    res = dict(
        harmful_refusal=refusal_rate(g_h), harmful_refusal_ablated=refusal_rate(g_h_abl),
        harmless_refusal=refusal_rate(g_b), harmless_refusal_added=refusal_rate(g_b_add),
    )
    res["drop"] = res["harmful_refusal"] - res["harmful_refusal_ablated"]
    res["pass_induce"] = res["harmless_refusal_added"] >= min_induce
    res["pass_ablate"] = res["drop"] >= min_drop
    res["pass"] = res["pass_induce"] and res["pass_ablate"]
    gens = ([dict(set="harmful", cond="none", prompt=p, out=o, refusal=is_refusal(o)) for p, o in zip(test_h, g_h)]
            + [dict(set="harmful", cond="ablate", prompt=p, out=o, refusal=is_refusal(o)) for p, o in zip(test_h, g_h_abl)]
            + [dict(set="harmless", cond="none", prompt=p, out=o, refusal=is_refusal(o)) for p, o in zip(test_b, g_b)]
            + [dict(set="harmless", cond="add", prompt=p, out=o, refusal=is_refusal(o)) for p, o in zip(test_b, g_b_add)])
    return res, gens


def extract(cfg: DictConfig) -> dict:
    seed_all(cfg.seed)
    model, tok = load_model(cfg.model.name, cfg.model.dtype)
    tpl = ChatTemplate.from_tokenizer(tok)
    ablator = Ablator(model)
    fit_h, val_h, test_h = P.split_fit_val_test(P.harmful_behaviors("train"), cfg.n_fit, cfg.n_val, cfg.n_test, cfg.seed)
    fit_b, val_b, test_b = P.split_fit_val_test(P.harmless_alpaca("train"), cfg.n_fit, cfg.n_val, cfg.n_test, cfg.seed)
    refusal_ids = find_refusal_tokens(model, tok, tpl, fit_h, fit_b)
    zero_dims = (outlier_dims(model, tok, tpl, fit_h + fit_b, cfg.outliers.ratio, cfg.outliers.min_layers)
                 if cfg.outliers.enabled else [])
    print(f"[direction] outlier dims zeroed: {zero_dims}", flush=True)
    sel = select_direction(model, tok, tpl, ablator, fit_h, fit_b, val_h, val_b, refusal_ids,
                           cfg.max_layer_frac, cfg.kl_max, zero_dims)
    best = sel["best"]
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(dict(rows=sel["candidates"], refusal_ids=refusal_ids,
                   refusal_tokens=[tok.decode([t]) for t in refusal_ids],
                   base_harmful_score=sel["base_harmful_score"], base_harmless_score=sel["base_harmless_score"]),
              out / "candidates.json")
    if best is None:
        raise RuntimeError(f"No candidate passed induce>0 and KL<{cfg.kl_max}; table in {out / 'candidates.json'}")
    gate, gens = gate_direction(model, tok, tpl, ablator, sel["direction"], best["layer"], test_h, test_b,
                                cfg.gate.max_new_tokens, cfg.gate.min_induce, cfg.gate.min_drop)
    torch.save(dict(direction=sel["direction"].cpu(), layer=best["layer"], pos=best["pos"],
                    model=cfg.model.name, refusal_ids=refusal_ids, zero_dims=zero_dims), out / "direction.pt")
    summary = dict(model=cfg.model.name, best=best, gate=gate, refusal_ids=refusal_ids, zero_dims=zero_dims,
                   refusal_tokens=[tok.decode([t]) for t in refusal_ids], n_eoi=sel["n_eoi"],
                   base_harmful_score=sel["base_harmful_score"], base_harmless_score=sel["base_harmless_score"],
                   config=OmegaConf.to_container(cfg, resolve=True), env=env_info())
    save_json(summary, out / "summary.json")
    save_jsonl(gens, Path(cfg.artifact_dir) / "gate_generations.jsonl")
    print(f"[direction] best={best} gate={gate}", flush=True)
    return summary


@hydra.main(config_path="../../configs", config_name="direction", version_base=None)
def main(cfg: DictConfig) -> None:
    summary = extract(cfg)
    if cfg.require_gate and not summary["gate"]["pass"]:
        raise SystemExit(f"direction failed the E0 gate: {summary['gate']}")


if __name__ == "__main__":
    main()
