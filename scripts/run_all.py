#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end pipeline runner: Phase 0 → 1 → 2 → 2b → 3 → 4.

Calls each phase's ``main()`` in order. Every phase has its own
``skip_existing`` so re-runs are cheap; you can also start mid-pipeline by
flipping the per-phase booleans (e.g. ``--do_download=false`` if APKs are
already on disk).

Default config: dev set of 200 APKs from ``configs/sha_dev_200.txt``.

Examples::

    # the full data pipeline on the 200-APK dev set, GPU encoding
    python scripts/run_all.py

    # dry-run / debug: only the first 5 APKs, CPU encoder
    python scripts/run_all.py --limit=5 --device=cpu

    # encode-only (Phases 0/1/2/2b already done)
    python scripts/run_all.py --do_download=false --do_disassemble=false \\
                              --do_extract=false --do_susi=false

    # Phase 4 only, through the unified entry point
    python scripts/run_all.py --do_download=false --do_disassemble=false \\
                              --do_extract=false --do_susi=false --do_encode=false \\
                              --do_train=true --device=cuda
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
    overrides_dir: str = "data/overrides",
    library_keys_dir: str = "data/library_keys",
    embeddings_dir: str = "data/embeddings/baseline",
    phase4_checkpoint_dir: str = "data/checkpoints",
    phase4_embeddings_dir: str = "data/embeddings/finetuned",
    susi_cache_dir: str = "third_party/susi",
    device: str = "cuda",
    batch_size: int = 32,
    workers: int = 4,
    limit: int = 0,
    phase4_device: str | None = None,
    phase4_limit: int = 0,
    phase4_batch_size: int = 8,
    phase4_epochs: int = 3,
    phase4_lr: float = 3e-5,
    phase4_weight_decay: float = 1e-2,
    phase4_temperature: float = 0.07,
    phase4_projection_dim: int = 256,
    phase4_freeze_n_layers: int = 4,
    phase4_freeze_embeddings: bool = True,
    phase4_label_group_size: int = 2,
    phase4_label_fraction: float = 0.5,
    phase4_steps_per_epoch: int = 1000,
    phase4_max_unlabeled_per_apk: int = 256,
    phase4_unlabeled_keep_ratio: float = 0.05,
    phase4_num_workers: int = 0,
    phase4_seed: int = 13,
    phase4_log_every: int = 20,
    phase4_grad_clip_norm: float = 1.0,
    phase4_export_after_train: bool = False,
    phase4_overrides_dir: str | None = "data/overrides",
    phase4_libraries_dir: str | None = "data/library_keys",
    phase4_use_signal_c: bool = False,
    phase4_use_signal_e: bool = False,
    phase4_cfg_path: str | None = None,
    phase4_weights_path: str | None = None,
    phase4_vocab_path: str | None = None,
    do_download: bool = True,
    do_disassemble: bool = True,
    do_extract: bool = True,
    do_susi: bool = True,
    do_overrides: bool = False,
    do_library_keys: bool = False,
    do_encode: bool = True,
    do_train: bool = False,
) -> None:
    """Run the Phase 0–4 pipeline for every SHA in ``sha_file``.

    Phase 4 knobs are prefixed with ``phase4_`` so the Phase 3 encode settings
    and the Phase 4 train settings can coexist in one unified debug entry.
    ``do_train`` defaults to ``False`` to preserve the old Phase 0–3 behavior.
    Current Phase 4 defaults reflect the latest small-scale local sweep:
    ``batch_size=8, lr=3e-5, label_fraction=0.5``.
    """
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

    if do_overrides:
        print("\n[run_all] Phase 2c: APK/class hierarchy -> override sidecar parquet")
        _load_phase_main("02c_extract_overrides.py")(
            sha_file=sha_file,
            apks_dir=apks_dir,
            out_dir=overrides_dir,
            limit=limit,
        )

    if do_library_keys:
        print("\n[run_all] Phase 2d: cross-APK exact full_id library sidecar parquet")
        _load_phase_main("02d_extract_library_keys.py")(
            sha_file=sha_file,
            methods_dir=methods_dir,
            out_dir=library_keys_dir,
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

    if do_train:
        print("\n[run_all] Phase 4: A+B contrastive fine-tuning")
        _load_phase_main("04_train_contrastive.py")(
            sha_file=sha_file,
            methods_dir=methods_dir,
            checkpoint_dir=phase4_checkpoint_dir,
            finetuned_dir=phase4_embeddings_dir,
            device=phase4_device or device,
            batch_size=phase4_batch_size,
            epochs=phase4_epochs,
            lr=phase4_lr,
            weight_decay=phase4_weight_decay,
            temperature=phase4_temperature,
            projection_dim=phase4_projection_dim,
            freeze_n_layers=phase4_freeze_n_layers,
            freeze_embeddings=phase4_freeze_embeddings,
            label_group_size=phase4_label_group_size,
            label_fraction=phase4_label_fraction,
            steps_per_epoch=phase4_steps_per_epoch,
            max_unlabeled_per_apk=phase4_max_unlabeled_per_apk,
            unlabeled_keep_ratio=phase4_unlabeled_keep_ratio,
            overrides_dir=phase4_overrides_dir,
            libraries_dir=phase4_libraries_dir,
            use_signal_c=phase4_use_signal_c,
            use_signal_e=phase4_use_signal_e,
            num_workers=phase4_num_workers,
            limit=phase4_limit if phase4_limit > 0 else limit,
            seed=phase4_seed,
            log_every=phase4_log_every,
            grad_clip_norm=phase4_grad_clip_norm,
            export_after_train=phase4_export_after_train,
            cfg_path=phase4_cfg_path,
            weights_path=phase4_weights_path,
            vocab_path=phase4_vocab_path,
        )

    print("\n[run_all] all requested phases finished")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        fire.Fire(main)
    else:
        main()
