"""Sanity tests for the migrated DexBERT encoder.

These tests are the engineering safety net for the encoder migration:
they run after every change to ``encoder/{loader,models,tokenization,utils}.py``
or every dependency upgrade, to confirm the pretrained checkpoint still
loads end-to-end and the transformer trunk produces sane outputs.

Requires:
    - ``configs/encoder_base.json``
    - ``assets/vocab.txt``
    - ``assets/model_steps_604364.pt``

The full module is marked ``slow`` because loading the 1.8 GB checkpoint
takes a couple of seconds. Skip with ``pytest -m 'not slow'`` when you
just want to run unit-level tests.
"""

from __future__ import annotations

import pytest
import torch

from androserum.encoder import (
    load_config,
    load_pretrained_encoder,
    load_tokenizer,
)

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Shared fixture: load the encoder once, reuse across every test in this file
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def loaded():
    """Load (transformer, tokenizer, cfg) once per test module."""
    return load_pretrained_encoder(device="cpu")


# ---------------------------------------------------------------------------
# Lightweight tests: don't need the full checkpoint
# ---------------------------------------------------------------------------
def test_load_config_standalone():
    """``configs/encoder_base.json`` parses into a valid Config."""
    cfg = load_config()
    assert cfg.vocab_size == 9537
    assert cfg.dim == 768
    assert cfg.n_layers == 8


def test_load_tokenizer_standalone():
    """``assets/vocab.txt`` loads into a 9,537-token FullTokenizer."""
    tok = load_tokenizer()
    assert len(tok.vocab) == 9537
    for required in ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"):
        assert required in tok.vocab, f"vocab missing reserved token {required!r}"


# ---------------------------------------------------------------------------
# Heavy tests: depend on the loaded fixture (= the .pt is read once)
# ---------------------------------------------------------------------------
def test_config_matches_dexbert_paper(loaded):
    """Numerical config matches the DexBERT-base spec."""
    _, _, cfg = loaded
    assert cfg.vocab_size == 9537
    assert cfg.dim == 768
    # DexBERT is intentionally shallower than BERT-base (8 vs 12 layers).
    assert cfg.n_layers == 8
    assert cfg.n_heads == 8
    assert cfg.max_len == 512


def test_transformer_param_count(loaded):
    """The transformer trunk should be ~55M params (8 layers × 768 dim)."""
    t, _, _ = loaded
    n = sum(p.numel() for p in t.parameters())
    assert 50_000_000 < n < 60_000_000, f"unexpected param count: {n:,}"


def test_transformer_is_in_eval_mode(loaded):
    """``loader`` returns the model in eval mode by default."""
    t, _, _ = loaded
    assert not t.training, "transformer should be in eval mode after load"


def test_forward_pass_shape_and_finiteness(loaded):
    """Forward on a CLS/SEP-padded input yields shape (1, max_len, dim) with finite values."""
    t, tok, cfg = loaded
    cls_id = tok.vocab["[CLS]"]
    sep_id = tok.vocab["[SEP]"]
    pad_id = tok.vocab["[PAD]"]

    ids = [cls_id, sep_id] + [pad_id] * (cfg.max_len - 2)
    inp = torch.tensor([ids], dtype=torch.long)
    seg = torch.zeros_like(inp)
    mask = torch.tensor(
        [[1, 1] + [0] * (cfg.max_len - 2)], dtype=torch.long
    )

    with torch.no_grad():
        h = t(inp, seg, mask)

    assert h.shape == (1, cfg.max_len, cfg.dim)
    assert torch.isfinite(h).all(), "encoder produced NaN/Inf outputs"

    cls_norm = h[0, 0].norm().item()
    # Empirically observed CLS norm with the DexBERT release weights is ~5.7;
    # we allow a wide range so that re-pretrained or fine-tuned weights can
    # still pass, but reject obvious degeneracy (zero vector, huge blow-up).
    assert 0.5 < cls_norm < 100.0, f"CLS norm out of plausible range: {cls_norm:.3f}"


def test_tokenizer_round_trip_no_unk_for_ascii_smali_keywords(loaded):
    """A handful of common smali keywords must be in-vocab (no [UNK])."""
    _, tok, _ = loaded
    unk_id = tok.vocab["[UNK]"]
    # Smali keywords that DexBERT's vocab is guaranteed to cover.
    sample = "invoke virtual direct static interface return move const-string"
    ids = tok.convert_tokens_to_ids(tok.tokenize(tok.convert_to_unicode(sample)))
    n_unk = sum(1 for i in ids if i == unk_id)
    assert n_unk == 0, (
        f"unexpected [UNK] for canonical smali keywords: "
        f"{n_unk}/{len(ids)} mapped to [UNK]"
    )
