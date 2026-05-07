#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add SuSi labels to ``data/methods/*.parquet`` (``susi_cats``, ``susi_dominant_cat``).

SuSi lists default to the upstream Android 4.2 text files. Override with local paths::

    python scripts/02b_tag_susi.py \\
        --susi_sources=third_party/susi/Ouput_CatSources_v0_9.txt \\
        --susi_sinks=third_party/susi/Ouput_CatSinks_v0_9.txt
"""

from __future__ import annotations

from pathlib import Path

import fire
import requests
from tqdm import tqdm

from androserum.data.method_parquet import read_methods_parquet, write_methods_parquet
from androserum.data.susi_index import build_susi_index
from androserum.data.susi_tagger import tag_method_susi

_DEFAULT_SOURCES = (
    "https://raw.githubusercontent.com/secure-software-engineering/SuSi/develop/"
    "SourceSinkLists/Android%204.2/SourcesSinks/Ouput_CatSources_v0_9.txt"
)
_DEFAULT_SINKS = (
    "https://raw.githubusercontent.com/secure-software-engineering/SuSi/develop/"
    "SourceSinkLists/Android%204.2/SourcesSinks/Ouput_CatSinks_v0_9.txt"
)


def _ensure_susi_file(url: str, cache_path: Path) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.is_file() and cache_path.stat().st_size > 0:
        return cache_path
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    cache_path.write_bytes(r.content)
    return cache_path


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
    susi_sources: str | None = None,
    susi_sinks: str | None = None,
    susi_cache_dir: str = "third_party/susi",
    limit: int = 0,
) -> None:
    """Tag every row in each Parquet with SuSi fields derived from ``api_calls``.

    Single entry-point: defaults make this debuggable from the IDE with
    zero args; ``fire`` covers CLI overrides.
    """
    root = Path(methods_dir)
    cache = Path(susi_cache_dir)
    if susi_sources and susi_sinks:
        src_p, snk_p = Path(susi_sources), Path(susi_sinks)
    else:
        src_p = _ensure_susi_file(_DEFAULT_SOURCES, cache / "Ouput_CatSources_v0_9.txt")
        snk_p = _ensure_susi_file(_DEFAULT_SINKS, cache / "Ouput_CatSinks_v0_9.txt")

    index = build_susi_index([src_p, snk_p])
    if len(index) == 0:
        raise RuntimeError("SuSi index is empty — check list files / network download.")

    shas = load_sha_list(sha_file, root)
    if limit and limit > 0:
        shas = shas[:limit]

    for sha in tqdm(shas, desc="02b_tag_susi", unit="apk"):
        pq = root / f"{sha}.parquet"
        if not pq.is_file():
            continue
        rows = read_methods_parquet(pq)
        tagged = [tag_method_susi(r, index) for r in rows]
        write_methods_parquet(tagged, pq)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        fire.Fire(main)
    else:
        main()
