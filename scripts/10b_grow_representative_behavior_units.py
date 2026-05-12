#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grow several representative behavior units for one APK."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import fire

from androserum.behavior.growth import grow_representative_behavior_units


@dataclass
class GrowRepresentativeBUsConfig:
    apk_sha: str
    anchor_json_path: str = ""
    clue_json_path: str = ""
    fcg_dir: str = "data/fcg"
    embedding_npz_path: str = ""
    output_dir: str = "data/reports/behavior_unit_sets"
    max_steps: int = 40
    max_nodes: int = 80
    tau_add: float = 0.01
    tau_quality_delta: float = 0.005
    tau_candidate_sim: float = 0.10
    min_nodes_target: int = 6
    trim_boundary: bool = True
    max_units: int = 5


def main(**kwargs) -> None:
    cfg = GrowRepresentativeBUsConfig(**kwargs)
    if not cfg.anchor_json_path:
        cfg.anchor_json_path = f"data/reports/anchor_discovery/{cfg.apk_sha}.anchors.json"
    if not cfg.clue_json_path:
        cfg.clue_json_path = f"data/reports/behavior_clues/{cfg.apk_sha}.clues.json"
    if not cfg.embedding_npz_path:
        cfg.embedding_npz_path = f"data/gnn_embeddings/gnn_bgrl_relay_package/{cfg.apk_sha}.npz"

    units = grow_representative_behavior_units(
        apk_sha=cfg.apk_sha,
        anchor_json_path=cfg.anchor_json_path,
        clue_json_path=cfg.clue_json_path,
        fcg_dir=cfg.fcg_dir,
        embedding_npz_path=cfg.embedding_npz_path,
        max_steps=cfg.max_steps,
        max_nodes=cfg.max_nodes,
        tau_add=cfg.tau_add,
        tau_quality_delta=cfg.tau_quality_delta,
        tau_candidate_sim=cfg.tau_candidate_sim,
        min_nodes_target=cfg.min_nodes_target,
        trim_boundary=cfg.trim_boundary,
        max_units=cfg.max_units,
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg.apk_sha}.behavior_unit_set.json"
    payload = {
        "config": asdict(cfg),
        "apk_sha": cfg.apk_sha,
        "behavior_units": [asdict(u) for u in units],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[phase1-set] wrote {out_path}")
    for idx, bu in enumerate(units, start=1):
        print(
            json.dumps(
                {
                    "rank": idx,
                    "anchor": bu.anchor_full_id,
                    "anchor_category": bu.anchor_category,
                    "behavior_label": bu.stats.get("behavior_label"),
                    "behavior_label_reason": bu.stats.get("behavior_label_reason"),
                    "n_total_nodes": bu.stats.get("n_total_nodes"),
                    "n_internal_nodes": bu.stats.get("n_internal_nodes"),
                    "n_external_nodes": bu.stats.get("n_external_nodes"),
                    "info_score": bu.stats.get("info_score"),
                    "conductance_proxy": bu.stats.get("conductance_proxy"),
                    "boilerplate_ratio": bu.stats.get("boilerplate_ratio"),
                    "terminated_by": bu.stats.get("terminated_by"),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    fire.Fire(main)
