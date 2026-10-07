"""Prompt-set loaders. Sources and sizes follow the E0 protocol:
lab/experiments/algorithms/2026-10-07-refusal-eval-protocol.md
"""

from __future__ import annotations

import csv
import os
import random
import urllib.request
from pathlib import Path

from datasets import load_dataset

# Public originals on GitHub (the walledai mirrors are gated on the Hub).
_SOURCES = {
    "harmbench_behaviors_text_all.csv":
        "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv",
    "advbench_harmful_behaviors.csv":
        "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv",
}


def _cached(name: str) -> Path:
    root = Path(os.environ.get("DRLAB_DATA_DIR", "/mnt/bigdata/deeprefusal/data"))
    path = root / name
    if not path.exists():
        root.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_SOURCES[name], path)
    return path


def harmful_behaviors(split: str = "train") -> list[str]:
    return list(load_dataset("mlabonne/harmful_behaviors", split=split)["text"])


def harmless_alpaca(split: str = "train") -> list[str]:
    return list(load_dataset("mlabonne/harmless_alpaca", split=split)["text"])


def harmbench_standard() -> list[str]:
    rows = list(csv.DictReader(open(_cached("harmbench_behaviors_text_all.csv"))))
    out = [r["Behavior"] for r in rows if r["FunctionalCategory"] == "standard"]
    if len(out) != 200:
        raise ValueError(f"HarmBench standard: expected 200 behaviors, got {len(out)}")
    return out


def advbench() -> list[tuple[str, str]]:
    rows = list(csv.DictReader(open(_cached("advbench_harmful_behaviors.csv"))))
    return [(r["goal"], r["target"]) for r in rows]


def or_bench_hard() -> list[str]:
    """OR-Bench-Hard-1k: benign prompts that look harmful (over-refusal set; E0 amendment 2026-10-07)."""
    return list(load_dataset("bench-llm/or-bench", "or-bench-hard-1k", split="train")["prompt"])


def mmlu_subsample(n: int, seed: int) -> list[dict]:
    """Seeded random subsample of MMLU test (all subjects): dicts with question, choices, answer."""
    ds = load_dataset("cais/mmlu", "all", split="test")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    return [ds[i] for i in idx[:n]]


def split_fit_val_test(items: list[str], n_fit: int, n_val: int, n_test: int, seed: int) -> tuple[list, list, list]:
    """Deterministic disjoint split."""
    items = list(dict.fromkeys(items))
    rng = random.Random(seed)
    rng.shuffle(items)
    need = n_fit + n_val + n_test
    if len(items) < need:
        raise ValueError(f"need {need} prompts, have {len(items)}")
    return items[:n_fit], items[n_fit:n_fit + n_val], items[n_fit + n_val:need]
