#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2 — ``processed/<sha>.txt`` → ``methods/<sha>.parquet``.

Examples::

    # 处理 data/processed/ 下所有 64-hex 文件名的 .txt
    python scripts/02_extract_methods.py

    # 只处理列表中的 SHA（一行一个，支持 # 注释）
    python scripts/02_extract_methods.py --sha_file=configs/sha_dev.txt

    # 强制重算已有的 parquet
    python scripts/02_extract_methods.py --skip_existing=false

Requires: ``pip install -e .`` so ``import androserum`` works.
"""

from __future__ import annotations

from pathlib import Path

import fire
from tqdm import tqdm

from androserum.data.method_extractor import extract_methods
from androserum.data.method_parquet import write_methods_parquet


def _normalize_sha_token(s: str) -> str | None:
    s = s.strip().upper()
    if len(s) != 64:
        return None
    if any(c not in "0123456789ABCDEF" for c in s):
        return None
    return s


def load_sha_list(sha_file: str | None, processed_dir: Path) -> list[str]:
    """SHAs to process: from ``sha_file`` or every ``*.txt`` under ``processed_dir``."""
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
            if sha is None:
                continue
            if sha not in seen:
                seen.add(sha)
                out.append(sha)
        return out

    shas: list[str] = []
    for p in sorted(processed_dir.glob("*.txt")):
        sha = _normalize_sha_token(p.stem)
        if sha is not None:
            shas.append(sha)
    return sorted(set(shas))


def main(
    sha_file: str | None = None,
    processed_dir: str = "data/processed",
    methods_dir: str = "data/methods",
    skip_existing: bool = True,
    limit: int = 0,
    encoding: str = "utf-8",
) -> None:
    """Parse instruction txts and write one Parquet per APK SHA.

    Single entry-point so the script can be debugged from an IDE with no
    args (defaults are used) and from the CLI via ``fire`` for argument
    injection. Set a breakpoint here to step through extraction.
    """
    proc_root = Path(processed_dir)
    out_root = Path(methods_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    shas = load_sha_list(sha_file, proc_root)
    if limit and limit > 0:
        shas = shas[:limit]

    if not shas:
        print(f"[WARN] no SHAs to process under {proc_root} (sha_file={sha_file!r})")
        return

    for sha in tqdm(shas, desc="02_extract_methods", unit="apk"):
        out_path = out_root / f"{sha}.parquet"
        if skip_existing and out_path.is_file() and out_path.stat().st_size > 0:
            continue
        txt_path = proc_root / f"{sha}.txt"
        if not txt_path.is_file():
            print(f"[SKIP] missing processed txt: {txt_path}")
            continue
        rows = extract_methods(txt_path, apk_sha=sha, encoding=encoding)
        write_methods_parquet(rows, out_path)


if __name__ == "__main__":
    import sys

    # No CLI args -> IDE / debugger run with all defaults.
    # Any CLI args  -> hand off to fire so you can override flags.
    if len(sys.argv) > 1:
        fire.Fire(main)
    else:
        main()
