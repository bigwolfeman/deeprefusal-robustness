"""DeepRefusal training data (paper section 5.1, official `train_dataset.py`).

Faithful parts: 2,000 CircuitBreaker harmful prompts with refusals, each also used once with a
harmful prefix of k ~ U[20, 25] tokens before the refusal (paper Eq. 11); benign UltraChat
first turns; XSTest compliant completions duplicated twice for over-refusal; benign
instruction dropped with probability 0.5 (official `switch_select = [0, 1]`); labels mask
the prompt and the harmful prefix.

Deliberate deviations (recorded in each run's config):
- Chat formatting comes from the tokenizer's template, so there is one BOS, not two.
- Right padding to the batch maximum instead of left padding to 1024. `response_start` is
  then correct in padded coordinates.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset

from drlab.arch import ChatTemplate


@dataclass
class Example:
    input_ids: list[int]
    labels: list[int]
    is_benign: bool
    response_start: int


def _encode(tok, tpl: ChatTemplate, instruction: str, response: str, prefix: str, max_len: int) -> tuple[list[int], list[int], int]:
    p_ids = tok(tpl.prompt(instruction), add_special_tokens=False).input_ids
    pre_ids = tok(prefix, add_special_tokens=False).input_ids if prefix else []
    r_ids = tok(response + tpl.eor, add_special_tokens=False).input_ids
    ids = (p_ids + pre_ids + r_ids)[:max_len]
    labels = ([-100] * (len(p_ids) + len(pre_ids)) + r_ids)[:max_len]
    return ids, labels, len(p_ids)


def load_xstest_compliant(path: str) -> list[tuple[str, str]]:
    rows = list(csv.DictReader(open(path)))
    return [(r["prompt"], r["completion"]) for r in rows
            if not r["type"].startswith("contrast") and r["final_label"] == "1_full_compliance"]


def load_ultrachat_first_turns(split: str, n: int, seed: int) -> list[tuple[str, str]]:
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split, streaming=True)
    out = []
    for ex in ds:
        m = ex["messages"]
        if len(m) >= 2 and m[0]["role"] == "user" and m[1]["role"] == "assistant":
            out.append((m[0]["content"], m[1]["content"]))
        if len(out) >= n:
            break
    if len(out) < n:
        raise ValueError(f"ultrachat {split}: wanted {n}, got {len(out)}")
    return out


def build_dr_dataset(cfg, tok, tpl: ChatTemplate) -> tuple[list[Example], dict]:
    """cfg: the `data` node of the training config."""
    rng = random.Random(cfg.seed)
    stats: dict = {}

    over = load_xstest_compliant(cfg.xstest_path) * cfg.overrefusal_repeat
    ultra = load_ultrachat_first_turns(cfg.benign_split, max(cfg.benign_total - len(over), 0), cfg.seed)
    benign_pairs = over + ultra
    stats.update(n_overrefusal=len(over), n_ultrachat=len(ultra))

    examples: list[Example] = []
    n_empty = 0
    for instr, resp in benign_pairs:
        if rng.random() < cfg.benign_empty_instruction_prob:
            instr = ""
            n_empty += 1
        ids, labels, start = _encode(tok, tpl, instr, resp, "", cfg.max_len)
        examples.append(Example(ids, labels, True, start))
    stats["n_benign_empty_instruction"] = n_empty

    cb = json.loads(Path(cfg.cb_path).read_text())
    rng.shuffle(cb)
    cb = cb[: cfg.n_harmful_prompts]
    n_prefix = 0
    for row in cb:
        ids, labels, start = _encode(tok, tpl, row["prompt"], row["llama3_output"], "", cfg.max_len)
        examples.append(Example(ids, labels, False, start))
        for _ in range(cfg.prefill_augmentations):
            toks = tok.tokenize(row["output"])
            k = rng.randint(cfg.prefix_min_tokens, cfg.prefix_max_tokens)
            prefix = tok.convert_tokens_to_string(toks[:k])
            ids, labels, start = _encode(tok, tpl, row["prompt"], row["llama3_output"], prefix, cfg.max_len)
            examples.append(Example(ids, labels, False, start))
            n_prefix += 1
    stats.update(n_harmful_prompts=len(cb), n_prefix_aug=n_prefix)

    rng.shuffle(examples)
    stats["n_total"] = len(examples)
    stats["n_labeled_tokens"] = sum(sum(1 for y in e.labels if y != -100) for e in examples)
    stats["mean_len"] = sum(len(e.input_ids) for e in examples) / len(examples)
    return examples, stats


def collate(batch: list[Example], pad_id: int) -> dict[str, torch.Tensor]:
    T = max(len(e.input_ids) for e in batch)
    ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), T), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), T), dtype=torch.long)
    for i, e in enumerate(batch):
        n = len(e.input_ids)
        ids[i, :n] = torch.tensor(e.input_ids)
        labels[i, :n] = torch.tensor(e.labels)
        attn[i, :n] = 1
    return dict(
        input_ids=ids, labels=labels, attention_mask=attn,
        is_benign=torch.tensor([e.is_benign for e in batch]),
        response_start=torch.tensor([e.response_start for e in batch]),
    )
