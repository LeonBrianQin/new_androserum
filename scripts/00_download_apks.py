#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 0 — AndroZoo: ``configs/sha_dev_*.txt`` → ``data/apks/<SHA>.apk``.

Thin ``fire`` wrapper around :mod:`androserum.data.androzoo` so this fits
the unified ``main()`` + IDE-debug pattern used by the other ``scripts/*``.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fire
from tqdm import tqdm

from androserum.data.androzoo import download_one


def _normalize_sha_token(s: str) -> str | None:
    s = s.strip().upper()
    if len(s) != 64:
        return None
    if any(c not in "0123456789ABCDEF" for c in s):
        return None
    return s


def load_sha_list(sha_file: str) -> list[str]:
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


def main(
    sha_file: str = "configs/sha_dev_200.txt",
    out_dir: str = "data/apks",
    apikey: str | None = None,
    workers: int = 4,
    timeout: int = 120,
    retries: int = 3,
    verify_hash: bool = True,
    limit: int = 0,
) -> None:
    """Download every APK in ``sha_file`` from AndroZoo into ``out_dir``."""
    key = (
        (apikey or "").strip()
        or os.environ.get("ANDROZOO_APIKEY", "").strip()
    )
    if not key:
        sys.exit(
            "[ERR] missing AndroZoo apikey: pass --apikey or set ANDROZOO_APIKEY "
            "(e.g. `source .env.local`)."
        )

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    shas = load_sha_list(sha_file)
    if limit and limit > 0:
        shas = shas[:limit]
    print(f"[INFO] downloading {len(shas)} APKs -> {out_root}")

    results: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                download_one,
                sha,
                str(out_root),
                key,
                timeout,
                retries,
                verify_hash,
            ): sha
            for sha in shas
        }
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="00_download",
            unit="apk",
        ):
            results.append(fut.result())

    ok = sum(1 for r in results if r.startswith("[OK]"))
    skip = sum(1 for r in results if r.startswith("[SKIP]"))
    fail = sum(1 for r in results if r.startswith("[FAIL]"))
    print(f"\nDone. OK={ok}, SKIP={skip}, FAIL={fail}")

    if fail:
        log_path = out_root / "download_failures.log"
        log_path.write_text(
            "\n".join(r for r in results if r.startswith("[FAIL]")) + "\n",
            encoding="utf-8",
        )
        print(f"failure log: {log_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        fire.Fire(main)
    else:
        main()
