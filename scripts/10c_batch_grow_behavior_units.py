#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch grow representative behavior units for many APKs."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

import fire

from androserum.behavior.growth import grow_representative_behavior_units


def _normalize_sha_token(s: str) -> str | None:
    s = s.strip().upper()
    if len(s) != 64:
        return None
    if any(c not in "0123456789ABCDEF" for c in s):
        return None
    return s


def _load_sha_file(sha_file: str) -> list[str]:
    p = Path(sha_file)
    if not p.is_file():
        raise FileNotFoundError(f"sha_file not found: {p}")
    out: list[str] = []
    seen: set[str] = set()
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sha = _normalize_sha_token(line.split(",", 1)[0])
        if sha is None or sha in seen:
            continue
        seen.add(sha)
        out.append(sha)
    return out


@dataclass
class BatchGrowConfig:
    sha_file: str = "configs/sha_dev_200.txt"
    anchor_json_dir: str = "data/reports/anchor_discovery"
    clue_json_dir: str = "data/reports/behavior_clues"
    fcg_dir: str = "data/fcg"
    embedding_dir: str = "data/gnn_embeddings/gnn_bgrl_relay_package"
    output_dir: str = "data/reports/behavior_unit_sets_dev200"
    max_units: int = 6
    max_steps: int = 30
    max_nodes: int = 60
    tau_add: float = 0.01
    tau_quality_delta: float = -0.01
    tau_candidate_sim: float = 0.10
    min_nodes_target: int = 6
    trim_boundary: bool = True
    max_workers: int = 8


def _run_one(cfg: BatchGrowConfig, sha: str) -> dict[str, Any]:
    units = grow_representative_behavior_units(
        apk_sha=sha,
        anchor_json_path=f"{cfg.anchor_json_dir}/{sha}.anchors.json",
        clue_json_path=f"{cfg.clue_json_dir}/{sha}.clues.json",
        fcg_dir=cfg.fcg_dir,
        embedding_npz_path=f"{cfg.embedding_dir}/{sha}.npz",
        max_steps=cfg.max_steps,
        max_nodes=cfg.max_nodes,
        tau_add=cfg.tau_add,
        tau_quality_delta=cfg.tau_quality_delta,
        tau_candidate_sim=cfg.tau_candidate_sim,
        min_nodes_target=cfg.min_nodes_target,
        trim_boundary=cfg.trim_boundary,
        max_units=cfg.max_units,
    )
    payload = {
        "apk_sha": sha,
        "behavior_units": [asdict(u) for u in units],
    }
    return payload


def main(**kwargs) -> None:
    cfg = BatchGrowConfig(**kwargs)
    shas = _load_sha_file(cfg.sha_file)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payloads: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=cfg.max_workers) as ex:
        futs = {ex.submit(_run_one, cfg, sha): sha for sha in shas}
        for fut in as_completed(futs):
            sha = futs[fut]
            payload = fut.result()
            payloads.append(payload)
            (out_dir / f"{sha}.behavior_unit_set.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[batch-bu] done {sha}")

    payloads.sort(key=lambda x: x["apk_sha"])

    sizes = []
    conductance = []
    infos = []
    label_counts = Counter()
    terminated = Counter()
    meaningful_units = 0
    meaningful_examples = []
    total_units = 0
    for payload in payloads:
        for bu in payload["behavior_units"]:
            total_units += 1
            s = bu["stats"]
            sizes.append(s["n_total_nodes"])
            conductance.append(s["conductance_proxy"])
            infos.append(s["info_score"])
            label = s.get("behavior_label") or "NONE"
            label_counts[label] += 1
            terminated[s.get("terminated_by") or "NONE"] += 1
            if s["n_total_nodes"] >= 4 and label not in {"NONE"}:
                meaningful_units += 1
                if len(meaningful_examples) < 50:
                    meaningful_examples.append(
                        {
                            "apk_sha": payload["apk_sha"],
                            "anchor_full_id": bu["anchor_full_id"],
                            "anchor_category": bu["anchor_category"],
                            "behavior_label": label,
                            "n_total_nodes": s["n_total_nodes"],
                            "conductance_proxy": s["conductance_proxy"],
                            "info_score": s["info_score"],
                            "terminated_by": s.get("terminated_by"),
                            "node_preview": bu["node_full_ids"][:10],
                        }
                    )

    summary = {
        "config": asdict(cfg),
        "n_apks": len(payloads),
        "total_units": total_units,
        "mean_size": mean(sizes) if sizes else 0.0,
        "median_size": median(sizes) if sizes else 0.0,
        "tiny_ratio": sum(1 for x in sizes if x <= 3) / max(1, len(sizes)),
        "target_ratio": sum(1 for x in sizes if 4 <= x <= 15) / max(1, len(sizes)),
        "large_ratio": sum(1 for x in sizes if x >= 20) / max(1, len(sizes)),
        "mean_conductance": mean(conductance) if conductance else 0.0,
        "mean_info_score": mean(infos) if infos else 0.0,
        "behavior_label_counts": dict(label_counts),
        "terminated_by_counts": dict(terminated),
        "meaningful_units": meaningful_units,
        "meaningful_ratio": meaningful_units / max(1, total_units),
        "meaningful_examples": meaningful_examples,
    }

    out_path = out_dir / "batch_behavior_unit_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[batch-bu] wrote {out_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    fire.Fire(main)

