"""Shared I/O, seeding, Hugging Face upload, and wandb setup."""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def save_jsonl(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def hf_upload(folder: str | Path, repo_id: str, path_in_repo: str, retries: int = 4) -> None:
    """Upload a folder to a Hub repo. Raises after `retries` failures (never silent)."""
    from huggingface_hub import HfApi

    api = HfApi()
    last = None
    for attempt in range(retries):
        try:
            api.upload_folder(folder_path=str(folder), repo_id=repo_id, path_in_repo=path_in_repo,
                              commit_message=f"upload {path_in_repo}")
            print(f"[hf] uploaded {folder} -> {repo_id}/{path_in_repo}", flush=True)
            return
        except Exception as e:  # network errors: retry with backoff, then raise
            last = e
            print(f"[hf] upload attempt {attempt + 1}/{retries} failed: {e!r}", file=sys.stderr, flush=True)
            time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"HF upload failed after {retries} attempts: {last!r}")


def _git_sha() -> str:
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                                       text=True).strip()
    except Exception as e:
        return f"unavailable: {e!r}"


# Captured when the job imports drlab, so a deploy during a long run does not change the record.
_SHA_AT_START = _git_sha()


def env_info() -> dict:
    import transformers, peft  # noqa: E401

    info = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "host": os.uname().nodename,
    }
    info["git_sha"] = _SHA_AT_START
    return info
