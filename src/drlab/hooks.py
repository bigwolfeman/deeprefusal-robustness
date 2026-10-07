"""Subspace ablation and direction addition hooks.

One `Ablator` owns three sites per decoder layer, in this order: residual input (`res`),
attention write (`attn`), MLP write (`mlp`). See `arch.py` for which modules are the writes.

Ablation removes the projection onto a per-layer orthonormal basis Q_l (k_l x d):
    x <- x - mask * (x Q_l^T) Q_l
DeepRefusal (paper Eq. 9, official `get_direction_ablation_hooks`) ablates each site in a
forward pass with probability p, then each position with probability p. Masks are drawn
once per step by `sample_masks` and stored, so gradient-checkpoint recomputation sees the
same masks as the original forward pass. Drawing inside the hook would not.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from drlab.arch import get_layers, get_writers

SITES_PER_LAYER = 3  # res, attn, mlp


class Ablator:
    def __init__(self, model: nn.Module):
        self.model = model
        self.layers = get_layers(model)
        self.n_layers = len(self.layers)
        self.n_sites = self.n_layers * SITES_PER_LAYER
        self.bases: list[Optional[torch.Tensor]] = [None] * self.n_layers  # fp32 (k_l, d)
        # Per-site mask: None = untouched, True = all positions, Tensor[B, T] bool = masked.
        self.masks: list = [None] * self.n_sites
        self.add_vec: Optional[tuple[int, torch.Tensor]] = None  # (layer, vector) added at layer input
        self.handles: list = []
        self._register()

    # ---- basis management -------------------------------------------------
    def set_shared_basis(self, Q: torch.Tensor) -> None:
        Q = _orthonormal_rows(Q.float())
        self.bases = [Q.clone() for _ in range(self.n_layers)]

    def set_layer_bases(self, bases: list[torch.Tensor]) -> None:
        if len(bases) != self.n_layers:
            raise ValueError(f"expected {self.n_layers} bases, got {len(bases)}")
        self.bases = [_orthonormal_rows(b.float()) for b in bases]

    def basis_dims(self) -> list[int]:
        return [0 if b is None else b.shape[0] for b in self.bases]

    # ---- mask control -----------------------------------------------------
    def off(self) -> None:
        self.masks = [None] * self.n_sites

    def full(self) -> None:
        self.masks = [True] * self.n_sites

    def sample_masks(
        self,
        batch: int,
        seq: int,
        p_site: float,
        p_pos: float,
        device: torch.device,
        response_start: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
    ) -> None:
        """DeepRefusal sampling. `response_start` (B,) limits ablation to positions >= start
        (correct for right padding). `valid` (B, T) excludes padding."""
        site_on = torch.rand(self.n_sites) < p_site
        pos_ok = None
        if response_start is not None:
            pos_ok = torch.arange(seq, device=device)[None, :] >= response_start[:, None].to(device)
        if valid is not None:
            pos_ok = valid.bool() if pos_ok is None else pos_ok & valid.bool()
        masks: list = []
        for s in range(self.n_sites):
            if not site_on[s]:
                masks.append(None)
                continue
            m = torch.rand(batch, seq, device=device) < p_pos
            if pos_ok is not None:
                m &= pos_ok
            masks.append(m)
        self.masks = masks

    def cell_fraction(self) -> float:
        """Fraction of (site, position) cells ablated in the current masks."""
        tot, on = 0.0, 0.0
        for m in self.masks:
            if m is None:
                tot += 1
            elif m is True:
                tot += 1
                on += 1
            else:
                tot += 1
                on += m.float().mean().item()
        return on / max(tot, 1)

    # ---- hooks --------------------------------------------------------------
    def _apply(self, x: torch.Tensor, site: int) -> torch.Tensor:
        m = self.masks[site]
        layer = site // SITES_PER_LAYER
        if m is not None and self.bases[layer] is not None:
            Q = self.bases[layer].to(x.device)
            xf = x.float()
            proj = (xf @ Q.T) @ Q
            if m is True:
                xf = xf - proj
            else:
                if m.shape != x.shape[:2]:
                    raise RuntimeError(f"mask shape {tuple(m.shape)} != activation {tuple(x.shape[:2])}")
                xf = xf - m[..., None].to(xf.dtype) * proj
            x = xf.to(x.dtype)
        if site % SITES_PER_LAYER == 0 and self.add_vec is not None and self.add_vec[0] == layer:
            x = x + self.add_vec[1].to(device=x.device, dtype=x.dtype)
        return x

    def _register(self) -> None:
        for li, layer in enumerate(self.layers):
            res_site = li * SITES_PER_LAYER

            def pre_hook(mod, args, kwargs, _s=res_site):
                if args:
                    return (self._apply(args[0], _s),) + tuple(args[1:]), kwargs
                kwargs["hidden_states"] = self._apply(kwargs["hidden_states"], _s)
                return args, kwargs

            self.handles.append(layer.register_forward_pre_hook(pre_hook, with_kwargs=True))
            attn_w, mlp_w = get_writers(self.model, layer)
            for off, mod in ((1, attn_w), (2, mlp_w)):

                def post_hook(mod_, inp, out, _s=res_site + off):
                    if isinstance(out, tuple):
                        return (self._apply(out[0], _s),) + tuple(out[1:])
                    return self._apply(out, _s)

                self.handles.append(mod.register_forward_hook(post_hook))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


def _orthonormal_rows(Q: torch.Tensor) -> torch.Tensor:
    if Q.dim() == 1:
        Q = Q[None]
    q, _ = torch.linalg.qr(Q.T)  # (d, k)
    return q.T.contiguous()
