#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2d — cross-APK library-key sidecars for signal C.

Current provider:
  exact full_id match across >= 2 APKs

Outputs:
  data/library_keys/<SHA>.parquet
"""

from __future__ import annotations

import json

import fire

from androserum.data.library_index import build_exact_full_id_library_sidecars


def main(
    sha_file: str | None = "configs/sha_dev_200.txt",
    methods_dir: str = "data/methods",
    out_dir: str = "data/library_keys",
    limit: int = 0,
    min_apk_support: int = 2,
) -> None:
    summary = build_exact_full_id_library_sidecars(
        methods_dir=methods_dir,
        out_dir=out_dir,
        sha_file=sha_file,
        limit=limit,
        min_apk_support=min_apk_support,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    fire.Fire(main)
