#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 1 — APK → smali → ``data/processed/<SHA>.txt`` instruction file.

Thin ``fire`` wrapper around ``androserum.data.apk_processor.process_one_apk``.
Skips APKs whose ``processed/<SHA>.txt`` already exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import fire
from tqdm import tqdm

from androserum.data.apk_processor import process_one_apk


def _normalize_sha_token(s: str) -> str | None:
    s = s.strip().upper()
    if len(s) != 64:
        return None
    if any(c not in "0123456789ABCDEF" for c in s):
        return None
    return s


def load_sha_list(sha_file: str | None, apks_dir: Path) -> list[str]:
    """If ``sha_file`` is given, use it; otherwise list every ``<SHA>.apk`` in ``apks_dir``."""
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
    for p in sorted(apks_dir.glob("*.apk")):
        sha = _normalize_sha_token(p.stem)
        if sha is not None:
            shas.append(sha)
    return sorted(set(shas))


def main(
    sha_file: str | None = "configs/sha_dev_200.txt",
    apks_dir: str = "data/apks",
    processed_dir: str = "data/processed",
    keep_smali: bool = False,
    skip_existing: bool = True,
    limit: int = 0,
) -> None:
    """Disassemble each ``<SHA>.apk`` into a DexBERT-style instruction txt."""
    a_dir = Path(apks_dir)
    p_dir = Path(processed_dir)
    p_dir.mkdir(parents=True, exist_ok=True)

    shas = load_sha_list(sha_file, a_dir)
    if limit and limit > 0:
        shas = shas[:limit]

    ok = fail = skip = miss = 0
    for sha in tqdm(shas, desc="01_disassemble", unit="apk"):
        apk_path = a_dir / f"{sha}.apk"
        txt_path = p_dir / f"{sha}.txt"
        if skip_existing and txt_path.is_file() and txt_path.stat().st_size > 0:
            skip += 1
            continue
        if not apk_path.is_file():
            print(f"[MISS] no apk: {apk_path}")
            miss += 1
            continue
        success = process_one_apk(
            str(apk_path), str(p_dir), keep_smali=keep_smali
        )
        if success:
            ok += 1
        else:
            fail += 1
    print(f"\nDone. OK={ok}, SKIP={skip}, MISS={miss}, FAIL={fail}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        fire.Fire(main)
    else:
        main()
