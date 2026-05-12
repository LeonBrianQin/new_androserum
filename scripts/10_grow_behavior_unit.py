#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 1: grow a behavior unit from one hard anchor."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import fire

from androserum.behavior.growth import grow_behavior_unit


@dataclass
class GrowBehaviorUnitConfig:
    apk_sha: str
    anchor_full_id: str
    anchor_json_path: str = ""
    clue_json_path: str = ""
    fcg_dir: str = "data/fcg"
    embedding_npz_path: str = ""
    output_dir: str = "data/reports/behavior_units"
    max_steps: int = 40
    max_nodes: int = 80
    tau_add: float = 0.01
    tau_quality_delta: float = 0.005
    tau_candidate_sim: float = 0.10
    min_nodes_target: int = 6
    trim_boundary: bool = True


def main(**kwargs) -> None:
    cfg = GrowBehaviorUnitConfig(**kwargs)
    if not cfg.anchor_json_path:
        cfg.anchor_json_path = f"data/reports/anchor_discovery/{cfg.apk_sha}.anchors.json"
    if not cfg.clue_json_path:
        cfg.clue_json_path = f"data/reports/behavior_clues/{cfg.apk_sha}.clues.json"
    if not cfg.embedding_npz_path:
        cfg.embedding_npz_path = f"data/gnn_embeddings/gnn_bgrl_relay_package/{cfg.apk_sha}.npz"

    bu = grow_behavior_unit(
        apk_sha=cfg.apk_sha,
        anchor_json_path=cfg.anchor_json_path,
        clue_json_path=cfg.clue_json_path,
        fcg_dir=cfg.fcg_dir,
        embedding_npz_path=cfg.embedding_npz_path,
        anchor_full_id=cfg.anchor_full_id,
        max_steps=cfg.max_steps,
        max_nodes=cfg.max_nodes,
        tau_add=cfg.tau_add,
        tau_quality_delta=cfg.tau_quality_delta,
        tau_candidate_sim=cfg.tau_candidate_sim,
        min_nodes_target=cfg.min_nodes_target,
        trim_boundary=cfg.trim_boundary,
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.apk_sha}.{cfg.anchor_full_id.replace('/', '_').replace(';','').replace('->','__').replace('(','_').replace(')','').replace(':','_')}"
    out_path = out_dir / f"{stem}.json"
    payload = {
        "config": asdict(cfg),
        "behavior_unit": asdict(bu),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[phase1] wrote {out_path}")
    print(json.dumps(bu.stats, ensure_ascii=False, indent=2))
    for step in bu.steps[:10]:
        print(
            json.dumps(
                {
                    "step_id": step.step_id,
                    "selected_full_id": step.selected_full_id,
                    "selected_node_kind": step.selected_node_kind,
                    "gain": step.gain,
                    "score_components": step.score_components,
                    "stop_reason": step.stop_reason,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    fire.Fire(main)
