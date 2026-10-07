"""Model loading, architecture access, and chat formatting.

Hook sites follow the DeepRefusal paper (Algorithm 1): the residual stream entering each
decoder layer, and the two writes into it (attention and MLP). For Llama the writes are the
outputs of `self_attn` and `mlp`. Gemma 2/3 apply a post-norm before the residual add, so the
write into the residual stream is the output of `post_attention_layernorm` and
`post_feedforward_layernorm`. Hooking `self_attn` on Gemma would ablate before an RMSNorm
with a learned per-channel weight, which does not keep the result orthogonal to the direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

_WRITER_ATTRS = {
    "gemma3": ("post_attention_layernorm", "post_feedforward_layernorm"),
    "gemma3_text": ("post_attention_layernorm", "post_feedforward_layernorm"),
    "gemma2": ("post_attention_layernorm", "post_feedforward_layernorm"),
    "llama": ("self_attn", "mlp"),
    "qwen2": ("self_attn", "mlp"),
    "qwen3": ("self_attn", "mlp"),
    "mistral": ("self_attn", "mlp"),
}


# Eager attention by default: with transformers 5.19, Gemma 3 + SDPA corrupts left-padded rows
# (every padded row returns the same next-token distribution). See
# .agents/postmortem/2026-10-07-gemma3-sdpa-left-padding.md
def load_model(name: str, dtype: str = "bfloat16", device: str = "cuda", attn_impl: str = "eager"):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=getattr(torch, dtype), attn_implementation=attn_impl
    ).to(device)
    model.eval()
    return model, tok


def model_type(model: nn.Module) -> str:
    cfg = getattr(model.config, "text_config", None) or model.config
    return cfg.model_type


def get_layers(model: nn.Module) -> nn.ModuleList:
    """Return the decoder-layer ModuleList, through PEFT and multimodal wrappers."""
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and name.endswith("layers") and len(mod) > 0:
            first = mod[0]
            if hasattr(first, "mlp") and (hasattr(first, "self_attn") or hasattr(first, "attention")):
                return mod
    raise ValueError(f"No decoder layers found in {type(model).__name__}")


def get_writers(model: nn.Module, layer: nn.Module) -> tuple[nn.Module, nn.Module]:
    mt = model_type(model)
    if mt not in _WRITER_ATTRS:
        raise ValueError(f"Unknown residual writer layout for model_type={mt}; add it to _WRITER_ATTRS")
    a, m = _WRITER_ATTRS[mt]
    return getattr(layer, a), getattr(layer, m)


def hidden_size(model: nn.Module) -> int:
    cfg = model.config
    cfg = getattr(cfg, "text_config", cfg)
    return cfg.hidden_size


@dataclass(frozen=True)
class ChatTemplate:
    """Pieces of the chat template, measured by rendering the tokenizer's own template.

    Rendering through `apply_chat_template` avoids the double-BOS bug in the official
    DeepRefusal Gemma template (literal `<bos>` plus tokenizer-added BOS).
    """

    prefix: str  # text before the user instruction
    eoi: str  # end-of-instruction text: after the instruction, up to the response start
    eor: str  # end-of-response text after the assistant content

    @classmethod
    def from_tokenizer(cls, tok: PreTrainedTokenizerBase) -> "ChatTemplate":
        marker, resp = "XQXINSTRUCTIONXQX", "XQXRESPONSEXQX"
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": marker}], add_generation_prompt=True, tokenize=False
        )
        full = tok.apply_chat_template(
            [{"role": "user", "content": marker}, {"role": "assistant", "content": resp}], tokenize=False
        )
        prefix, eoi = prompt.split(marker)
        if not full.startswith(prompt):
            raise ValueError("Chat template: full conversation does not start with the generation prompt")
        eor = full[len(prompt):].split(resp)[1]
        return cls(prefix=prefix, eoi=eoi, eor=eor)

    def prompt(self, instruction: str) -> str:
        return f"{self.prefix}{instruction}{self.eoi}"


def eoi_token_count(tok: PreTrainedTokenizerBase, tpl: ChatTemplate) -> int:
    return len(tok(tpl.eoi, add_special_tokens=False).input_ids)


def encode_prompts(tok, tpl: ChatTemplate, instructions: list[str], device: str = "cuda") -> dict:
    """Left-padded batch of generation prompts with explicit position_ids (pads excluded).
    The template already contains BOS."""
    tok.padding_side = "left"
    texts = [tpl.prompt(x) for x in instructions]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    pos = (enc["attention_mask"].cumsum(1) - 1).clamp(min=0)
    return dict(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], position_ids=pos)
