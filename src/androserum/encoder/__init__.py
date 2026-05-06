"""DexBERT transformer encoder (migrated from upstream, TF-free).

Public API::

    from androserum.encoder import load_pretrained_encoder
    transformer, tokenizer, cfg = load_pretrained_encoder(device="cuda")
"""

from .loader import (
    DEFAULT_CFG_REL,
    DEFAULT_VOCAB_REL,
    DEFAULT_WEIGHTS_REL,
    load_config,
    load_pretrained_encoder,
    load_tokenizer,
)

__all__ = [
    "DEFAULT_CFG_REL",
    "DEFAULT_VOCAB_REL",
    "DEFAULT_WEIGHTS_REL",
    "load_config",
    "load_pretrained_encoder",
    "load_tokenizer",
]
