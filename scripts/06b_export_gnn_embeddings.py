#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export Phase 6 GNN embeddings for an arbitrary SHA split from a checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import fire
import torch

from androserum.gnn import FcgGraphDataset, GnnTrainConfig, export_gnn_embeddings
from androserum.gnn.bgrl import BgrlModel


@dataclass
class ExportPhase6Config:
    checkpoint_path: str
    sha_file: str
    fcg_dir: str
    embeddings_dir: str
    out_dir: str
    device: str = "cuda"


def main(**kwargs) -> None:
    cfg = ExportPhase6Config(**kwargs)
    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu")
    train_cfg = ckpt["train_config"]
    gcfg = GnnTrainConfig(**train_cfg)
    family_to_id = ckpt.get("family_to_id") or None

    dataset = FcgGraphDataset.from_dirs(
        fcg_dir=cfg.fcg_dir,
        embeddings_dir=cfg.embeddings_dir,
        sha_file=cfg.sha_file,
        limit=0,
        graph_mode=gcfg.graph_mode,
        external_prior_mode=gcfg.external_prior_mode,
        add_reverse_edges=gcfg.add_reverse_edges,
        family_to_id=family_to_id,
    )
    if len(dataset) == 0:
        raise RuntimeError("no graphs to export")

    sample0 = dataset.load(0)
    model = BgrlModel(
        input_dim=int(sample0.x.shape[1]),
        hidden_dim=gcfg.hidden_dim,
        output_dim=gcfg.output_dim,
        predictor_hidden_dim=gcfg.predictor_hidden_dim,
        dropout=gcfg.encoder_dropout,
        external_prior_mode=gcfg.external_prior_mode,
        external_family_vocab=len(dataset.family_to_id),
        ema_decay=gcfg.ema_decay,
    ).to(cfg.device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    written = export_gnn_embeddings(
        model=model,
        dataset=dataset,
        out_dir=cfg.out_dir,
        device=cfg.device,
        encoder_name=gcfg.export_encoder,
    )
    summary = {
        "checkpoint_path": cfg.checkpoint_path,
        "sha_file": cfg.sha_file,
        "fcg_dir": cfg.fcg_dir,
        "embeddings_dir": cfg.embeddings_dir,
        "out_dir": cfg.out_dir,
        "written": written,
        "graph_mode": gcfg.graph_mode,
        "external_prior_mode": gcfg.external_prior_mode,
        "external_family_vocab": len(dataset.family_to_id),
    }
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.out_dir) / "_export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    fire.Fire(main)
