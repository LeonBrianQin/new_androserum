#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2c — APK/class hierarchy -> override sidecar parquet.

Produces:
    data/overrides/<SHA>.parquet
"""

from __future__ import annotations

from pathlib import Path

import fire
from tqdm import tqdm

from androserum.data.override_index import (
    build_override_records_for_apk,
    write_override_parquet,
)
from androserum.train.dataset import load_sha_list


def main(
    sha_file: str | None = "configs/sha_dev_200.txt",
    apks_dir: str = "data/apks",
    out_dir: str = "data/overrides",
    skip_existing: bool = True,
    limit: int = 0,
) -> None:
    a_dir = Path(apks_dir)
    o_dir = Path(out_dir)
    o_dir.mkdir(parents=True, exist_ok=True)

    shas = load_sha_list(sha_file, a_dir)
    if limit and limit > 0:
        shas = shas[:limit]

    ok = skip = miss = fail = 0
    for sha in tqdm(shas, desc="02c_extract_overrides", unit="apk"):
        apk_path = a_dir / f"{sha}.apk"
        out_path = o_dir / f"{sha}.parquet"
        if skip_existing and out_path.is_file() and out_path.stat().st_size > 0:
            skip += 1
            continue
        if not apk_path.is_file():
            print(f"[MISS] no apk: {apk_path}")
            miss += 1
            continue
        try:
            rows = build_override_records_for_apk(apk_path, sha)
            write_override_parquet(rows, out_path)
            ok += 1
        except Exception as e:
            print(f"[FAIL] {sha}: {type(e).__name__}: {e}")
            fail += 1

    print(f"\nDone. OK={ok}, SKIP={skip}, MISS={miss}, FAIL={fail}")


if __name__ == "__main__":
    fire.Fire(main)
