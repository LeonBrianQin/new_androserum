#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 4 — A+B[+E] contrastive fine-tuning on method-level DexBERT inputs.

Default behavior trains on the dev 200 SHA list and writes checkpoints to
``data/checkpoints``. Exporting finetuned per-APK embeddings is optional,
because it can take a while and is often run only after the training loss
looks healthy.

Examples::

    # local MVP run on the full dev-200 set
    python scripts/04_train_contrastive.py --device=cuda --epochs=3

    # tiny debug run
    python scripts/04_train_contrastive.py --limit=3 --steps_per_epoch=20 --epochs=1

    # train then immediately export finetuned CLS embeddings
    python scripts/04_train_contrastive.py --export_after_train=true

    # enable signal E (same override target), assuming data/overrides/ exists
    python scripts/04_train_contrastive.py --use_signal_e=true
"""

from __future__ import annotations

from dataclasses import fields

import fire

from androserum.train import ContrastiveTrainConfig, train_contrastive_ab


def main(**kwargs) -> None:
    """Train the Phase 4 A+B contrastive model from CLI keyword arguments."""
    allowed = {f.name for f in fields(ContrastiveTrainConfig)}
    unknown = sorted(set(kwargs) - allowed)
    if unknown:
        raise TypeError(f"unknown args for ContrastiveTrainConfig: {unknown}")
    cfg = ContrastiveTrainConfig(**kwargs)
    train_contrastive_ab(cfg)


if __name__ == "__main__":
    fire.Fire(main)
