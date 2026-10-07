"""DeepRefusal analysis, attacks, and defenses. See docs/constitution.md."""

from omegaconf import OmegaConf

if not OmegaConf.has_resolver("sanitize"):
    OmegaConf.register_new_resolver("sanitize", lambda s: str(s).replace("/", "__"))
