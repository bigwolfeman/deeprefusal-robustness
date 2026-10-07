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

`data.responses` selects the response text: `offpolicy` (the recipe above: Llama-3 refusals, GPT
answers) or `onpolicy` (same prompts and prefixes, responses written by the defended model, from
`python -m drlab.onpolicy`; .agents/notes/proposed/feature/2026-10-07-onpolicy-dr-data.md).
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
class Sources:
    """Prompts and random draws shared by the off-policy and on-policy builds. The RNG draw order
    (benign coin flips, CircuitBreaker shuffle, prefix lengths) matches commit 3bb80fb, so the
    off-policy data is identical to the E3 `dr_baseline` data."""
    benign: list[tuple[str, str]]       # (instruction, off-policy response), XSTest x repeat + UltraChat
    empty_instruction: list[bool]       # per benign row
    cb: list[dict]                      # CircuitBreaker rows in training order
    prefixes: list[list[str]]           # per cb row, `prefill_augmentations` harmful prefixes
    rng: random.Random                  # continues with the final example shuffle
    stats: dict


@dataclass
class Example:
    input_ids: list[int]
    labels: list[int]
    is_benign: bool
    response_start: int


def _encode(tok, tpl: ChatTemplate, instruction: str, response: str, prefix: str, max_len: int,
            finished: bool = True, cap: int | None = None) -> tuple[list[int], list[int], int]:
    """`finished=False` (a generated response cut at its token limit) omits the end-of-turn
    target, so the model is not taught to stop mid-answer. `cap` truncates the response to that
    many tokens (then it is unfinished)."""
    p_ids = tok(tpl.prompt(instruction), add_special_tokens=False).input_ids
    pre_ids = tok(prefix, add_special_tokens=False).input_ids if prefix else []
    if cap is None:
        r_ids = tok(response + (tpl.eor if finished else ""), add_special_tokens=False).input_ids
    else:
        r_ids = tok(response, add_special_tokens=False).input_ids
        if len(r_ids) > cap:
            r_ids, finished = r_ids[:cap], False
        r_ids = r_ids + (tok(tpl.eor, add_special_tokens=False).input_ids if finished else [])
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


def select_sources(cfg, tok) -> Sources:
    """cfg: the `data` node of the training config."""
    rng = random.Random(cfg.seed)
    over = load_xstest_compliant(cfg.xstest_path) * cfg.overrefusal_repeat
    ultra = load_ultrachat_first_turns(cfg.benign_split, max(cfg.benign_total - len(over), 0), cfg.seed)
    benign = over + ultra
    empty = [rng.random() < cfg.benign_empty_instruction_prob for _ in benign]
    cb = json.loads(Path(cfg.cb_path).read_text())
    rng.shuffle(cb)
    cb = cb[: cfg.n_harmful_prompts]
    prefixes = []
    for row in cb:
        row_prefixes = []
        for _ in range(cfg.prefill_augmentations):
            toks = tok.tokenize(row["output"])
            k = rng.randint(cfg.prefix_min_tokens, cfg.prefix_max_tokens)
            row_prefixes.append(tok.convert_tokens_to_string(toks[:k]))
        prefixes.append(row_prefixes)
    stats = dict(n_overrefusal=len(over), n_ultrachat=len(ultra), n_benign_empty_instruction=sum(empty),
                 n_harmful_prompts=len(cb), n_prefix_aug=sum(len(x) for x in prefixes))
    return Sources(benign, empty, cb, prefixes, rng, stats)


def _offpolicy_rows(src: Sources):
    """(instruction, response, finished, is_benign, prefix, cap) in build order."""
    for (instr, resp), empty in zip(src.benign, src.empty_instruction):
        yield ("" if empty else instr), resp, True, True, "", None
    for row, row_prefixes in zip(src.cb, src.prefixes):
        yield row["prompt"], row["llama3_output"], True, False, "", None
        for prefix in row_prefixes:
            yield row["prompt"], row["llama3_output"], True, False, prefix, None


def _onpolicy_rows(src: Sources, path: str, stats: dict, tok, match_length: bool):
    """Same rows as `_offpolicy_rows`, responses from `drlab.onpolicy`. Dropped prompts are
    skipped and counted. Fails loudly if the file was built from different sources.

    `match_length`: cap each response at the token count of the off-policy response it replaces.
    The DR loss sums over tokens per sample, so this keeps every row's loss weight equal to the
    off-policy build and makes the response source the only changed variable (E4 Method)."""
    def cap(off_resp: str) -> int | None:
        return len(tok(off_resp, add_special_tokens=False).input_ids) if match_length else None

    op = json.loads(Path(path).read_text())
    harm = op["harmful"]
    if [h["prompt"] for h in harm] != [r["prompt"] for r in src.cb]:
        raise ValueError(f"{path}: harmful prompts differ from data config (rebuild with drlab.onpolicy)")
    if [[h["prefix"]] for h in harm] != src.prefixes:
        raise ValueError(f"{path}: harmful prefixes differ from data config (prefill_augmentations must be 1)")
    benign = op["benign"]
    dropped = {"benign": 0, "harmful": 0, "prefix": 0}
    for (instr, off_resp), empty in zip(src.benign, src.empty_instruction):
        b = benign.get(instr)
        if b is None:
            raise ValueError(f"{path}: no on-policy response for benign prompt {instr[:60]!r}")
        if b["source"] == "dropped":
            dropped["benign"] += 1
            continue
        yield ("" if empty else instr), b["response"], b["finished"], True, "", cap(off_resp)
    for h, row in zip(harm, src.cb):
        if h["refusal_source"] == "dropped":
            dropped["harmful"] += 1
            dropped["prefix"] += 1
            continue
        off_cap = cap(row["llama3_output"])
        yield h["prompt"], h["refusal"], h["refusal_finished"], False, "", off_cap
        yield h["prompt"], h["recovery"], h["recovery_finished"], False, h["prefix"], off_cap
    stats["onpolicy_dropped"] = dropped
    stats["onpolicy_sources"] = op["summary"]["sources"]


def build_dr_dataset(cfg, tok, tpl: ChatTemplate) -> tuple[list[Example], dict]:
    """cfg: the `data` node of the training config."""
    src = select_sources(cfg, tok)
    stats = dict(src.stats, responses=cfg.responses)
    if cfg.responses == "offpolicy":
        rows = _offpolicy_rows(src)
    elif cfg.responses == "onpolicy":
        rows = _onpolicy_rows(src, cfg.onpolicy_path, stats, tok, cfg.onpolicy_match_length)
    else:
        raise ValueError(f"data.responses must be offpolicy or onpolicy, got {cfg.responses!r}")
    examples = []
    for instr, resp, finished, is_benign, prefix, cap in rows:
        ids, labels, start = _encode(tok, tpl, instr, resp, prefix, cfg.max_len, finished, cap)
        examples.append(Example(ids, labels, is_benign, start))
    src.rng.shuffle(examples)
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
