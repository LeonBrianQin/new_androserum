"""Unified entry point for loading the pretrained DexBERT encoder + tokenizer.

The pretrained checkpoint was saved as the full ``BertAEModel4Pretrain``
state_dict (transformer + AE + classifier + decoder), but for clustering
we only need the transformer trunk. This module:

1. Reads the encoder config from JSON.
2. Builds a fresh ``models.Transformer``.
3. Loads only the ``transformer.*`` slice of the checkpoint into it.
4. Returns the transformer + tokenizer + config triplet.

Typical use::

    from androserum.encoder import load_pretrained_encoder

    transformer, tokenizer, cfg = load_pretrained_encoder(device="cuda")
    transformer.eval()  # already eval-mode by default

See ``PROJECT_HANDOFF.md`` §7.2 for the full design rationale.
"""

from __future__ import annotations

from pathlib import Path

import torch

from . import models, tokenization

DEFAULT_CFG_REL = "configs/encoder_base.json"
DEFAULT_WEIGHTS_REL = "assets/model_steps_604364.pt"
DEFAULT_VOCAB_REL = "assets/vocab.txt"

_TRANSFORMER_PREFIX = "transformer."


def _project_root() -> Path:
    """Walk up from this file until we find ``pyproject.toml``."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(
        f"could not locate project root (no pyproject.toml found above {here})"
    )


def _resolve(path: str | Path | None, default_rel: str) -> Path:
    """Resolve a user-provided path, falling back to ``<project_root>/<default_rel>``."""
    if path is None:
        return _project_root() / default_rel
    p = Path(path)
    if not p.is_absolute():
        # Interpret relative paths against the project root, not cwd.
        # That way ``load_pretrained_encoder()`` works no matter where the
        # caller's cwd is, as long as the assets live under the project tree.
        candidate = _project_root() / p
        if candidate.exists() or not p.exists():
            return candidate
    return p


def load_config(cfg_path: str | Path | None = None) -> "models.Config":
    """Load ``models.Config`` from a JSON file (defaults to ``configs/encoder_base.json``)."""
    resolved = _resolve(cfg_path, DEFAULT_CFG_REL)
    if not resolved.exists():
        raise FileNotFoundError(f"encoder config not found: {resolved}")
    return models.Config.from_json(str(resolved))


def load_tokenizer(
    vocab_path: str | Path | None = None,
    do_lower_case: bool = True,
) -> "tokenization.FullTokenizer":
    """Build a ``FullTokenizer`` from a vocab.txt (defaults to ``assets/vocab.txt``)."""
    resolved = _resolve(vocab_path, DEFAULT_VOCAB_REL)
    if not resolved.exists():
        raise FileNotFoundError(f"vocab file not found: {resolved}")
    return tokenization.FullTokenizer(
        vocab_file=str(resolved), do_lower_case=do_lower_case
    )


def _extract_transformer_state(full_state: dict) -> dict:
    """Strip the ``transformer.`` prefix and keep only those keys."""
    return {
        k[len(_TRANSFORMER_PREFIX):]: v
        for k, v in full_state.items()
        if k.startswith(_TRANSFORMER_PREFIX)
    }


def load_pretrained_encoder(
    cfg_path: str | Path | None = None,
    weights_path: str | Path | None = None,
    vocab_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple["models.Transformer", "tokenization.FullTokenizer", "models.Config"]:
    """Load the DexBERT pretrained transformer encoder + tokenizer.

    Args:
        cfg_path: Encoder config JSON. Defaults to ``configs/encoder_base.json``.
        weights_path: Pretrained ``.pt`` (full ``BertAEModel4Pretrain`` state_dict).
            Defaults to ``assets/model_steps_604364.pt``.
        vocab_path: Vocab file. Defaults to ``assets/vocab.txt``.
        device: Target device. The checkpoint is always loaded on CPU first
            (to avoid OOM on a fresh GPU) and then moved.
        strict: If True (default), require an exact match between the
            ``transformer.*`` slice of the checkpoint and the freshly
            instantiated ``Transformer`` state_dict. Set to False only when
            you knowingly load a partial checkpoint.

    Returns:
        ``(transformer, tokenizer, cfg)``. ``transformer`` is in eval mode.

    Raises:
        FileNotFoundError: If any of the three files is missing.
        RuntimeError: If cfg/vocab sizes disagree, or the checkpoint contains
            no ``transformer.*`` keys, or ``strict=True`` and keys mismatch.
    """
    cfg = load_config(cfg_path)
    tokenizer = load_tokenizer(vocab_path)

    if cfg.vocab_size != len(tokenizer.vocab):
        raise RuntimeError(
            f"cfg.vocab_size ({cfg.vocab_size}) != tokenizer vocab size "
            f"({len(tokenizer.vocab)}); cfg and vocab files don't match"
        )

    weights = _resolve(weights_path, DEFAULT_WEIGHTS_REL)
    if not weights.exists():
        raise FileNotFoundError(f"weights file not found: {weights}")

    transformer = models.Transformer(cfg)

    # weights_only=False is required because the upstream checkpoint was
    # produced with PyTorch < 2.4 conventions; the file is from a trusted
    # source (DexBERT release), so this is safe in practice.
    full_state = torch.load(
        str(weights), map_location="cpu", weights_only=False
    )
    if not isinstance(full_state, dict):
        raise RuntimeError(
            f"unexpected checkpoint format: expected dict, got {type(full_state).__name__}"
        )

    transformer_state = _extract_transformer_state(full_state)
    if not transformer_state:
        raise RuntimeError(
            f"no 'transformer.*' keys found in {weights}; "
            f"got keys like {list(full_state)[:5]} -- wrong checkpoint?"
        )

    missing, unexpected = transformer.load_state_dict(
        transformer_state, strict=strict
    )
    if (missing or unexpected) and not strict:
        # When strict=True PyTorch raises on its own; this branch only fires
        # when the caller opted into lenient loading.
        raise RuntimeError(
            "transformer state_dict mismatch: "
            f"missing={len(missing)}, unexpected={len(unexpected)}; "
            f"first missing={missing[:5]}, first unexpected={unexpected[:5]}"
        )

    transformer.to(device)
    transformer.eval()
    return transformer, tokenizer, cfg


__all__ = [
    "DEFAULT_CFG_REL",
    "DEFAULT_VOCAB_REL",
    "DEFAULT_WEIGHTS_REL",
    "load_config",
    "load_pretrained_encoder",
    "load_tokenizer",
]
