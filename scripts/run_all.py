#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end pipeline runner: Phase 0 → 1 → 2 → 2b → 3.

Calls each phase's ``main()`` in order. Every phase has its own
``skip_existing`` so re-runs are cheap; you can also start mid-pipeline by
flipping the per-phase booleans (e.g. ``--do_download=false`` if APKs are
already on disk).

Default config: dev set of 200 APKs from ``configs/sha_dev_200.txt``.

Examples::

    # the full pipeline on the 200-APK dev set, GPU encoding
    python scripts/run_all.py

    # dry-run / debug: only the first 5 APKs, CPU encoder
    python scripts/run_all.py --limit=5 --device=cpu

    # encode-only (Phases 0/1/2/2b already done)
    python scripts/run_all.py --do_download=false --do_disassemble=false \\
                              --do_extract=false --do_susi=false
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import fire


def _load_phase_main(script_filename: str):
    """Import a ``scripts/<script_filename>`` module and return its ``main``."""
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    mod_name = script_filename.removesuffix(".py")
    mod = import_module(mod_name)
    return mod.main


def main(
    sha_file: str = "configs/sha_dev_200.txt",
    apks_dir: str = "data/apks",
    processed_dir: str = "data/processed",
    methods_dir: str = "data/methods",
    embeddings_dir: str = "data/embeddings/baseline",
    susi_cache_dir: str = "third_party/susi",
    device: str = "cuda",
    batch_size: int = 32,
    workers: int = 4,
    limit: int = 0,
    do_download: bool = True,
    do_disassemble: bool = True,
    do_extract: bool = True,
    do_susi: bool = True,
    do_encode: bool = True,
) -> None:
    """Run the full Phase 0–3 pipeline for every SHA in ``sha_file``."""
    if do_download:
        print("\n[run_all] Phase 0: download APKs")
        _load_phase_main("00_download_apks.py")(
            sha_file=sha_file,
            out_dir=apks_dir,
            workers=workers,
            limit=limit,
        )

    if do_disassemble:
        print("\n[run_all] Phase 1: APK -> smali -> processed/<SHA>.txt")
        _load_phase_main("01_disassemble_apks.py")(
            sha_file=sha_file,
            apks_dir=apks_dir,
            processed_dir=processed_dir,
            limit=limit,
        )

    if do_extract:
        print("\n[run_all] Phase 2: processed/<SHA>.txt -> methods/<SHA>.parquet")
        _load_phase_main("02_extract_methods.py")(
            sha_file=sha_file,
            processed_dir=processed_dir,
            methods_dir=methods_dir,
            limit=limit,
        )

    if do_susi:
        print("\n[run_all] Phase 2b: tag SuSi categories on every parquet")
        _load_phase_main("02b_tag_susi.py")(
            sha_file=sha_file,
            methods_dir=methods_dir,
            susi_cache_dir=susi_cache_dir,
            limit=limit,
        )

    if do_encode:
        print("\n[run_all] Phase 3: frozen-DexBERT encode -> embeddings/<SHA>.npz")
        _load_phase_main("03_encode_methods.py")(
            sha_file=sha_file,
            methods_dir=methods_dir,
            out_dir=embeddings_dir,
            device=device,
            batch_size=batch_size,
            limit=limit,
        )

    print("\n[run_all] all requested phases finished")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        fire.Fire(main)
    else:
        main()
