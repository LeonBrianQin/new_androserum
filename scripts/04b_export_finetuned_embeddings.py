#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export finetuned Phase 4 embeddings for an arbitrary SHA split from a checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import fire
import torch

from androserum.train import ContrastiveDexBertModel, export_finetuned_embeddings


@dataclass
class ExportPhase4Config:
    checkpoint_path: str
    sha_file: str
    methods_dir: str
    out_dir: str
    device: str = "cuda"
    batch_size: int = 16
    cfg_path: str | None = None
    weights_path: str | None = None
    vocab_path: str | None = None


def main(**kwargs) -> None:
    cfg = ExportPhase4Config(**kwargs)
    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu")
    projection_dim = int(ckpt.get("projection_dim", 256))

    model, tokenizer, encoder_cfg = ContrastiveDexBertModel.from_pretrained(
        projection_dim=projection_dim,
        cfg_path=cfg.cfg_path,
        weights_path=cfg.weights_path,
        vocab_path=cfg.vocab_path,
        device=cfg.device,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    written = export_finetuned_embeddings(
        model=model,
        tokenizer=tokenizer,
        encoder_cfg=encoder_cfg,
        methods_dir=cfg.methods_dir,
        out_dir=cfg.out_dir,
        batch_size=cfg.batch_size,
        device=cfg.device,
        sha_file=cfg.sha_file,
        limit=0,
    )
    summary = {
        "checkpoint_path": cfg.checkpoint_path,
        "sha_file": cfg.sha_file,
        "methods_dir": cfg.methods_dir,
        "out_dir": cfg.out_dir,
        "written": written,
    }
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.out_dir) / "_export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    fire.Fire(main)

