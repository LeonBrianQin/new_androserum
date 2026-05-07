"""Frozen DexBERT transformer: method instruction text → pooled CLS vectors."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from tqdm import tqdm

from androserum.encoder import load_pretrained_encoder

__all__ = ["instructions_to_cls_batch", "encode_methods_parquet_file"]


def _pad_batch(
    ids_rows: list[list[int]],
    max_len: int,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build ``input_ids``, ``seg``, ``mask`` (1 = real token, 0 = pad)."""
    b = len(ids_rows)
    inp = torch.full((b, max_len), pad_id, dtype=torch.long)
    seg = torch.zeros((b, max_len), dtype=torch.long)
    mask = torch.zeros((b, max_len), dtype=torch.long)
    for i, row in enumerate(ids_rows):
        n = min(len(row), max_len)
        inp[i, :n] = torch.tensor(row[:n], dtype=torch.long)
        mask[i, :n] = 1
    return inp, seg, mask


def _rows_to_cls(
    texts: Sequence[str],
    *,
    transformer: torch.nn.Module,
    tokenizer,
    cfg,
    device: torch.device,
    batch_size: int,
    progress: bool,
    desc: str,
) -> np.ndarray:
    cls_id = tokenizer.vocab["[CLS]"]
    sep_id = tokenizer.vocab["[SEP]"]
    pad_id = tokenizer.vocab["[PAD]"]
    max_len = cfg.max_len
    n = len(texts)
    out_parts: list[np.ndarray] = []
    n_batch = (n + batch_size - 1) // batch_size
    starts = range(0, n, batch_size)
    if progress and n_batch > 0:
        starts = tqdm(starts, total=n_batch, desc=desc, unit="batch")
    for start in starts:
        chunk = texts[start : start + batch_size]
        buf_ids: list[list[int]] = []
        for text in chunk:
            uni = tokenizer.convert_to_unicode(text or "")
            toks = tokenizer.tokenize(uni)
            cap = max_len - 2
            if len(toks) > cap:
                toks = toks[:cap]
            ids_row = [cls_id] + tokenizer.convert_tokens_to_ids(toks) + [sep_id]
            buf_ids.append(ids_row)
        inp, seg, mask = _pad_batch(buf_ids, max_len, pad_id)
        inp = inp.to(device)
        seg = seg.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            h = transformer(inp, seg, mask)
        cls = h[:, 0, :].detach().float().cpu().numpy()
        out_parts.append(cls)
    if not out_parts:
        return np.zeros((0, cfg.dim), dtype=np.float32)
    return np.concatenate(out_parts, axis=0)


def instructions_to_cls_batch(
    texts: Sequence[str],
    transformer: torch.nn.Module,
    tokenizer,
    cfg,
    device: torch.device,
    batch_size: int = 32,
    *,
    progress: bool = False,
    desc: str = "encode",
) -> np.ndarray:
    """Tokenize instruction blobs, run encoder, return float32 ``(N, dim)`` CLS rows."""
    return _rows_to_cls(
        texts,
        transformer=transformer,
        tokenizer=tokenizer,
        cfg=cfg,
        device=device,
        batch_size=batch_size,
        progress=progress,
        desc=desc,
    )


def encode_methods_parquet_file(
    parquet_path: str,
    out_npz: str,
    *,
    device: str | torch.device = "cuda",
    batch_size: int = 32,
    cfg_path: str | None = None,
    weights_path: str | None = None,
    vocab_path: str | None = None,
    show_progress: bool = True,
) -> None:
    """Read ``data/methods/<sha>.parquet``, write ``full_id`` + ``embedding`` ``.npz``."""
    from pathlib import Path

    from androserum.data.method_parquet import read_methods_parquet

    p = Path(parquet_path)
    if not p.is_file():
        raise FileNotFoundError(str(p))

    if isinstance(device, str) and device == "cuda":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        dev = torch.device(device)
    else:
        dev = device

    transformer, tokenizer, cfg = load_pretrained_encoder(
        cfg_path=cfg_path,
        weights_path=weights_path,
        vocab_path=vocab_path,
        device=dev,
    )

    rows = read_methods_parquet(p)
    sep = "\n"
    texts = [sep.join(r.instructions) for r in rows]
    ids_str = [r.full_id for r in rows]

    emb = instructions_to_cls_batch(
        texts,
        transformer,
        tokenizer,
        cfg,
        dev,
        batch_size=batch_size,
        progress=show_progress,
        desc=f"encode {p.stem[:8]}…",
    )

    out = Path(out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        full_id=np.array(ids_str, dtype=object),
        embedding=emb.astype(np.float32, copy=False),
    )
