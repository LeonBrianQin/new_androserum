#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3 — ``methods/<sha>.parquet`` → ``embeddings/baseline/<sha>.npz`` (frozen DexBERT).

Uses GPU when ``device=cuda`` and PyTorch sees a CUDA device; otherwise falls back to CPU.

Examples::

    python scripts/03_encode_methods.py --device=cuda --batch_size=32
    python scripts/03_encode_methods.py --sha_file=configs/sha_dev.txt
"""

from __future__ import annotations

from pathlib import Path

import fire
from tqdm import tqdm

from androserum.inference.frozen_encode import encode_methods_parquet_file


def _normalize_sha_token(s: str) -> str | None:
    s = s.strip().upper()
    if len(s) != 64:
        return None
    if any(c not in "0123456789ABCDEF" for c in s):
        return None
    return s


def load_sha_list(sha_file: str | None, methods_dir: Path) -> list[str]:
    if sha_file:
        p = Path(sha_file)
        if not p.is_file():
            raise FileNotFoundError(f"sha_file not found: {p}")
        seen: set[str] = set()
        out: list[str] = []
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sha = _normalize_sha_token(line.split(",", 1)[0])
            if sha is None or sha in seen:
                continue
            seen.add(sha)
            out.append(sha)
        return out

    shas: list[str] = []
    for p in sorted(methods_dir.glob("*.parquet")):
        sha = _normalize_sha_token(p.stem)
        if sha is not None:
            shas.append(sha)
    return sorted(set(shas))


def main(
    sha_file: str | None = None,
    methods_dir: str = "data/methods",
    out_dir: str = "data/embeddings/baseline",
    device: str = "cuda",
    batch_size: int = 32,
    skip_existing: bool = True,
    limit: int = 0,
    cfg_path: str | None = None,
    weights_path: str | None = None,
    vocab_path: str | None = None,
) -> None:
    """Encode each methods parquet to a compressed ``.npz`` with ``full_id`` + ``embedding``.

    Single entry-point: zero-arg run uses the defaults (good for IDE
    debugging / breakpoints); CLI invocation goes through ``fire``.
    """
    mdir = Path(methods_dir)
    odir = Path(out_dir)
    odir.mkdir(parents=True, exist_ok=True)

    shas = load_sha_list(sha_file, mdir)
    if limit and limit > 0:
        shas = shas[:limit]

    for sha in tqdm(shas, desc="03_encode_methods", unit="apk"):
        pq = mdir / f"{sha}.parquet"
        if not pq.is_file():
            continue
        target = odir / f"{sha}.npz"
        if skip_existing and target.is_file() and target.stat().st_size > 0:
            continue
        encode_methods_parquet_file(
            str(pq),
            str(target),
            device=device,
            batch_size=batch_size,
            cfg_path=cfg_path,
            weights_path=weights_path,
            vocab_path=vocab_path,
            show_progress=False,
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        fire.Fire(main)
    else:
        main()
