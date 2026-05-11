#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 6 — GraphSAGE+BGRL over Phase 5 FCG sidecars.

Supported graph modes:

  * ``internal_only``:
      only aligned internal nodes + internal edges
  * ``relay`` + ``external_prior_mode=none``:
      include boundary relay nodes, zero features + node-type embedding
  * ``relay`` + ``external_prior_mode=global|package``:
      include boundary relay nodes, plus a simple external prior
"""

from __future__ import annotations

from dataclasses import asdict, fields

import fire

from androserum.gnn import GnnTrainConfig, load_gnn_config, train_bgrl_graphsage


def main(**kwargs) -> None:
    cfg_path = kwargs.pop("cfg_path", None)
    if cfg_path:
        cfg = load_gnn_config(cfg_path)
        merged = asdict(cfg)
        merged.update(kwargs)
        cfg = GnnTrainConfig(**merged)
    else:
        allowed = {f.name for f in fields(GnnTrainConfig)}
        unknown = sorted(set(kwargs) - allowed)
        if unknown:
            raise TypeError(f"unknown args for GnnTrainConfig: {unknown}")
        cfg = GnnTrainConfig(**kwargs)

    train_bgrl_graphsage(cfg)


if __name__ == "__main__":
    fire.Fire(main)
