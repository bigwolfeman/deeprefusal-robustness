"""DeepRefusal training with optional Prong A (adaptive subspace) and Prong B (path-robust loss).

Design: .agents/notes/proposed/feature/2026-10-07-adaptive-deeprefusal.md
Faithful baseline: paper Algorithm 1 + official src/main.py (sum-over-tokens per-sample loss,
loss = (1 - a) * benign + a * malicious with a = 0.2, Bernoulli site/position ablation of r_hat).

Prong A: every `refit_every` steps, refit per-layer refusal directions on the current model with
the current subspace ablated everywhere, and add the part orthogonal to the subspace. r_hat stays
in every layer's basis (the paper's Table 3 shows replacing it fails). With probability
`full_prob` a step ablates the whole subspace at every site and position (what weight
abliteration does).

Prong B: LoRA from the pretrained model, so the LoRA delta is the whole post-training delta.
On a path step, every LoRA layer's scaling is multiplied by lambda_l ~ U[lambda_min, 1]
(W_pt + lambda_l * delta_l) and the refusal loss on harmful rows is added at that point.
"""

from __future__ import annotations

import math
import random
import re
import shutil
import time
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from drlab import prompts as P
from drlab.arch import ChatTemplate, eoi_token_count
from drlab.data import build_dr_dataset, collate
from drlab.directions import mean_eoi_acts
from drlab.hooks import Ablator
from drlab.utils import env_info, hf_upload, save_json, seed_all


# ---------------------------------------------------------------------------- loss
def chunked_token_nll(hidden: torch.Tensor, lm_head: torch.nn.Module, labels: torch.Tensor,
                      chunk: int, softcap: float | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token NLL on labeled positions only, never materializing [B, T, V] logits.
    Returns (nll[N], row_index[N])."""
    h = hidden[:, :-1]
    y = labels[:, 1:]
    m = y != -100
    rows = torch.arange(y.shape[0], device=y.device)[:, None].expand_as(y)[m]
    hs, ys = h[m], y[m]

    def piece(hc, yc):
        logits = lm_head(hc).float()
        if softcap:
            logits = torch.tanh(logits / softcap) * softcap
        return F.cross_entropy(logits, yc, reduction="none")

    out = [checkpoint(piece, hs[i:i + chunk], ys[i:i + chunk], use_reentrant=False)
           for i in range(0, hs.shape[0], chunk)]
    return torch.cat(out), rows


def dr_loss(decoder, lm_head, batch: dict, alpha: float, chunk: int, softcap, rows_mask=None) -> tuple[torch.Tensor, dict]:
    """Official DeepRefusal loss: per-sample SUM of token NLL; (1-a) mean benign + a mean malicious."""
    ids, attn, labels, benign = batch["input_ids"], batch["attention_mask"], batch["labels"], batch["is_benign"]
    if rows_mask is not None:
        ids, attn, labels, benign = ids[rows_mask], attn[rows_mask], labels[rows_mask], benign[rows_mask]
    hidden = decoder(input_ids=ids, attention_mask=attn).last_hidden_state
    nll, rows = chunked_token_nll(hidden, lm_head, labels, chunk, softcap)
    per = torch.zeros(ids.shape[0], device=nll.device).index_add_(0, rows, nll)
    zero = torch.zeros((), device=nll.device)
    b_loss = per[benign].mean() if benign.any() else zero
    m_loss = per[~benign].mean() if (~benign).any() else zero
    loss = (1 - alpha) * b_loss + alpha * m_loss
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite loss: benign={b_loss.item()} malicious={m_loss.item()}")
    ntok = (labels[:, 1:] != -100).sum().item()
    return loss, dict(benign=b_loss.item(), malicious=m_loss.item(), tokens=ntok,
                      mean_tok_nll=(nll.mean().item() if nll.numel() else 0.0))


# ---------------------------------------------------------------------------- LoRA path scaling
def lora_layer_map(model) -> list[tuple[LoraLayer, int]]:
    out = []
    for name, mod in model.named_modules():
        if isinstance(mod, LoraLayer):
            m = re.search(r"layers\.(\d+)\.", name)
            if m is None:
                raise ValueError(f"LoRA module without layer index: {name}")
            out.append((mod, int(m.group(1))))
    return out


class PathScaler:
    """Temporarily scale each layer's LoRA delta: W = W_0 + lambda_l * B A * s."""

    def __init__(self, model):
        self.mods = lora_layer_map(model)
        self.base = [dict(m.scaling) for m, _ in self.mods]

    def set(self, lambdas: list[float]) -> None:
        for (m, li), base in zip(self.mods, self.base):
            for k, v in base.items():
                m.scaling[k] = v * lambdas[li]

    def reset(self) -> None:
        for (m, _), base in zip(self.mods, self.base):
            m.scaling.update(base)


# ---------------------------------------------------------------------------- Prong A refit
@torch.no_grad()
def refit_subspace(model, tok, tpl, ablator: Ablator, fit_h, fit_b, k_max: int, min_ratio: float) -> dict:
    was_training = model.training
    model.eval()
    n_eoi = eoi_token_count(tok, tpl)
    ablator.off()
    u = mean_eoi_acts(model, tok, tpl, fit_h, n_eoi)[-1] - mean_eoi_acts(model, tok, tpl, fit_b, n_eoi)[-1]
    ablator.full()
    a = mean_eoi_acts(model, tok, tpl, fit_h, n_eoi)[-1] - mean_eoi_acts(model, tok, tpl, fit_b, n_eoi)[-1]
    ablator.off()
    new, added, ratios = [], 0, []
    for l, Q in enumerate(ablator.bases):
        r = a[l]
        r_perp = r - (r @ Q.T.to(r.device)) @ Q.to(r.device)
        ratio = (r_perp.norm() / u[l].norm().clamp_min(1e-6)).item()
        ratios.append(ratio)
        if ratio >= min_ratio and Q.shape[0] < k_max:
            Q = torch.cat([Q.to(r.device), (r_perp / r_perp.norm())[None]], 0)
            added += 1
        new.append(Q)
    ablator.set_layer_bases(new)
    if was_training:
        model.train()
    return dict(added=added, mean_ratio=sum(ratios) / len(ratios), max_ratio=max(ratios),
                mean_k=sum(ablator.basis_dims()) / ablator.n_layers)


# ---------------------------------------------------------------------------- main
def lr_at(step: int, total: int, cfg) -> float:
    warm = max(1, int(cfg.warmup_frac * total))
    if step < warm:
        return cfg.lr * (step + 1) / warm
    if cfg.schedule == "constant":
        return cfg.lr
    prog = (step - warm) / max(1, total - warm)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * prog))


def save_ckpt(model, ablator: Ablator, out: Path, meta: dict) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    model.save_pretrained(out)
    torch.save(dict(bases=[b.cpu() for b in ablator.bases]), out / "ablation_bases.pt")
    save_json(meta, out / "meta.json")


def train(cfg: DictConfig) -> dict:
    seed_all(cfg.seed)
    t0 = time.time()
    run_dir = Path(cfg.root) / "ckpts" / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg.model.tokenizer or cfg.model.name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tpl = ChatTemplate.from_tokenizer(tok)
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, dtype=getattr(torch, cfg.model.dtype),
                                                 attn_implementation="sdpa").cuda()
    text_cfg = getattr(model.config, "text_config", None) or model.config
    softcap = getattr(text_cfg, "final_logit_softcapping", None)
    if cfg.train.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lcfg = LoraConfig(r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
                      target_modules=list(cfg.lora.targets), task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    base = model.get_base_model()
    decoder, lm_head = base.model, base.lm_head
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    ablator = Ablator(model)
    dpack = torch.load(cfg.direction_path, map_location="cpu")
    ablator.set_shared_basis(dpack["direction"].float())

    examples, dstats = build_dr_dataset(cfg.data, tok, tpl)
    g = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(examples, batch_size=cfg.train.batch_size, shuffle=True, generator=g,
                        collate_fn=lambda b: collate(b, tok.pad_token_id), drop_last=False)
    total = len(loader) * cfg.train.epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay, betas=(0.9, 0.999))
    scaler = PathScaler(model) if cfg.path.enabled else None
    fit_h = fit_b = None
    if cfg.dr.adaptive.enabled:
        fit_h, _, _ = P.split_fit_val_test(P.harmful_behaviors("train"), cfg.dr.adaptive.n_fit, 0, 0, cfg.seed + 1)
        fit_b, _, _ = P.split_fit_val_test(P.harmless_alpaca("train"), cfg.dr.adaptive.n_fit, 0, 0, cfg.seed + 1)

    full_cfg = OmegaConf.to_container(cfg, resolve=True)
    derived = dict(data_stats=dstats, total_steps=total, n_trainable=n_trainable, softcap=softcap,
                   direction_layer=dpack.get("layer"), direction_pos=dpack.get("pos"),
                   direction_model=dpack.get("model"), model_config=text_cfg.to_dict(),
                   lora_config=lcfg.to_dict(), chat_template_parts=tpl.__dict__, env=env_info(),
                   deviations=["chat template via tokenizer (single BOS)", "right padding, dynamic length",
                               "masks drawn once per step (checkpoint-safe)"])
    wandb.init(project=cfg.wandb.project, group=cfg.wandb.group, name=cfg.run_name, mode=cfg.wandb.mode,
               config=dict(cfg=full_cfg, derived=derived), dir=str(Path(cfg.root) / "wandb"))
    save_json(dict(cfg=full_cfg, derived=derived), run_dir / "run_config.json")
    print(f"[train] {cfg.run_name}: {dstats} steps={total} trainable={n_trainable}", flush=True)

    rng = random.Random(cfg.seed)
    model.train()
    step, stopped_early = 0, False
    for epoch in range(cfg.train.epochs):
        for batch in loader:
            if (time.time() - t0) / 3600 > cfg.train.time_limit_h:
                stopped_early = True
                break
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            for gp in opt.param_groups:
                gp["lr"] = lr_at(step, total, cfg.train)
            B, T = batch["input_ids"].shape
            ts = time.time()

            full_step = cfg.dr.enabled and rng.random() < cfg.dr.full_prob
            if not cfg.dr.enabled:
                ablator.off()
            elif full_step:
                ablator.full()
            else:
                ablator.sample_masks(B, T, cfg.dr.p, cfg.dr.p, batch["input_ids"].device,
                                     batch["response_start"] if cfg.dr.only_response else None)
            cells = ablator.cell_fraction()
            loss, parts = dr_loss(decoder, lm_head, batch, cfg.train.loss_alpha, cfg.train.ce_chunk, softcap)
            loss.backward()

            path_log = {}
            if scaler is not None and rng.random() < cfg.path.prob and (~batch["is_benign"]).any():
                L = ablator.n_layers
                lam = ([rng.uniform(cfg.path.lambda_min, 1.0) for _ in range(L)] if cfg.path.per_layer
                       else [rng.uniform(cfg.path.lambda_min, 1.0)] * L)
                scaler.set(lam)
                harm = ~batch["is_benign"]
                if cfg.dr.enabled:
                    ablator.sample_masks(int(harm.sum()), T, cfg.dr.p, cfg.dr.p, batch["input_ids"].device)
                p_loss, p_parts = dr_loss(decoder, lm_head, batch, 1.0, cfg.train.ce_chunk, softcap, rows_mask=harm)
                (cfg.path.beta * p_loss).backward()  # backward before reset: recompute must see the same scaling
                scaler.reset()
                path_log = {"path/loss": p_parts["malicious"], "path/lambda_mean": sum(lam) / L}
            ablator.off()

            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip).item()
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            log = {"loss": loss.item(), "loss/benign": parts["benign"], "loss/malicious": parts["malicious"],
                   "tok_nll": parts["mean_tok_nll"], "lr": opt.param_groups[0]["lr"], "grad_norm": gnorm,
                   "ablated_cell_frac": cells, "full_step": int(full_step), "seq_len": T,
                   "tokens_per_s": parts["tokens"] / max(time.time() - ts, 1e-6),
                   "mem_gb": torch.cuda.max_memory_allocated() / 2**30, **path_log}
            if cfg.dr.adaptive.enabled and step % cfg.dr.adaptive.refit_every == 0:
                r = refit_subspace(model, tok, tpl, ablator, fit_h, fit_b, cfg.dr.adaptive.k_max, cfg.dr.adaptive.min_ratio)
                log.update({f"refit/{k}": v for k, v in r.items()})
            wandb.log(log, step=step)
            if step % 10 == 0:
                print(f"[train] step {step}/{total} loss={loss.item():.2f} b={parts['benign']:.1f} "
                      f"m={parts['malicious']:.1f} gn={gnorm:.2f} T={T} mem={log['mem_gb']:.1f}G "
                      f"tok/s={log['tokens_per_s']:.0f} cells={cells:.3f}", flush=True)
            if step % cfg.train.save_every == 0:
                save_ckpt(model, ablator, run_dir / "latest", dict(step=step, partial=True))
                if cfg.hf.upload:
                    hf_upload(run_dir / "latest", cfg.hf.repo, f"{cfg.run_name}/latest")
        if stopped_early:
            break

    meta = dict(step=step, total_steps=total, completed=not stopped_early, hours=(time.time() - t0) / 3600,
                basis_dims=ablator.basis_dims())
    save_ckpt(model, ablator, run_dir / "final", meta)
    if cfg.hf.upload:
        hf_upload(run_dir / "final", cfg.hf.repo, f"{cfg.run_name}/final")
    print(f"[train] done {meta}", flush=True)
    return dict(model=model, tok=tok, tpl=tpl, ablator=ablator, run_dir=run_dir, meta=meta, dpack=dpack)


@hydra.main(config_path="../../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    out = train(cfg)
    if cfg.eval.enabled:
        from drlab.evalq import quick_eval

        res = quick_eval(out["model"], out["tok"], out["tpl"], out["ablator"], out["dpack"], cfg,
                         out["run_dir"] / "final")
        wandb.log({f"quick_eval/{k}": v for k, v in res.items() if isinstance(v, (int, float))})
        if cfg.hf.upload:
            hf_upload(out["run_dir"] / "final", cfg.hf.repo, f"{cfg.run_name}/final")
    wandb.finish()


if __name__ == "__main__":
    main()
