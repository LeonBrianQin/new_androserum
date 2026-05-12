#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sweep Phase 1 growth thresholds on one or more APKs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import json
from pathlib import Path
from statistics import mean, median
from typing import Any
from concurrent.futures import ProcessPoolExecutor, as_completed

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


def _normalize_family(label: str | None) -> str | None:
    if label is None:
        return None
    fam_map = {
        "NETWORK": "NETWORK",
        "NETWORK_INFORMATION": "NETWORK",
        "FILE": "FILE",
        "LOG": "LOGGING",
        "LOGGING": "LOGGING",
        "LOCATION_INFORMATION": "LOCATION",
        "DATABASE_INFORMATION": "DATABASE",
        "UNIQUE_IDENTIFIER": "IDENTIFIER",
        "IDENTIFIER": "IDENTIFIER",
        "AUDIO": "AUDIO",
        "BLUETOOTH_INFORMATION": "BLUETOOTH",
        "CALENDAR_INFORMATION": "CALENDAR",
        "MIXED": "MIXED",
        "NO_CATEGORY": "NO_CATEGORY",
    }
    return fam_map.get(label, label)


@dataclass
class SweepConfig:
    apk_shas: str = ""  # comma-separated
    sha_file: str = ""
    anchor_json_dir: str = "data/reports/anchor_discovery"
    clue_json_dir: str = "data/reports/behavior_clues"
    fcg_dir: str = "data/fcg"
    embedding_dir: str = "data/gnn_embeddings/gnn_bgrl_relay_package"
    output_dir: str = "data/reports/behavior_growth_sweeps"
    max_units: int = 6
    max_steps: int = 30
    max_nodes: int = 60
    min_nodes_target: int = 6
    tau_add_values: str = "0.01,0.005,0.0"
    tau_quality_delta_values: str = "0.005,0.0,-0.005"
    tau_candidate_sim_values: str = "0.10,0.08,0.06"
    max_workers: int = 8


def _parse_floats(s) -> list[float]:
    if isinstance(s, (list, tuple)):
        out: list[float] = []
        for item in s:
            out.extend(_parse_floats(item))
        return out
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def _score_config(units: list[dict[str, Any]]) -> dict[str, Any]:
    sizes = [u["stats"]["n_total_nodes"] for u in units]
    conductance = [u["stats"]["conductance_proxy"] for u in units]
    infos = [u["stats"]["info_score"] for u in units]
    tiny = sum(1 for x in sizes if x <= 3)
    target = sum(1 for x in sizes if 4 <= x <= 15)
    large = sum(1 for x in sizes if x >= 20)

    align_ok = 0
    align_total = 0
    physical_like = 0
    for u in units:
        anchor = _normalize_family(u["anchor_category"])
        label = _normalize_family(u["stats"].get("behavior_label"))
        if anchor in {None, "NO_CATEGORY", "MIXED"}:
            continue
        align_total += 1
        if anchor == label:
            align_ok += 1
        if label in {"NETWORK", "FILE", "DATABASE", "LOCATION", "IDENTIFIER", "LOGGING"}:
            physical_like += 1

    return {
        "n_units": len(units),
        "mean_size": mean(sizes) if sizes else 0.0,
        "median_size": median(sizes) if sizes else 0.0,
        "tiny_ratio": tiny / max(1, len(sizes)),
        "target_ratio": target / max(1, len(sizes)),
        "large_ratio": large / max(1, len(sizes)),
        "mean_conductance": mean(conductance) if conductance else 0.0,
        "mean_info_score": mean(infos) if infos else 0.0,
        "label_alignment": align_ok / max(1, align_total) if align_total else None,
        "physical_like_ratio": physical_like / max(1, len(units)),
    }


def _run_one_apk(
    cfg: SweepConfig,
    sha: str,
    tau_add: float,
    tau_qd: float,
    tau_sim: float,
) -> list[dict[str, Any]]:
    units = grow_representative_behavior_units(
        apk_sha=sha,
        anchor_json_path=f"{cfg.anchor_json_dir}/{sha}.anchors.json",
        clue_json_path=f"{cfg.clue_json_dir}/{sha}.clues.json",
        fcg_dir=cfg.fcg_dir,
        embedding_npz_path=f"{cfg.embedding_dir}/{sha}.npz",
        max_steps=cfg.max_steps,
        max_nodes=cfg.max_nodes,
        tau_add=tau_add,
        tau_quality_delta=tau_qd,
        tau_candidate_sim=tau_sim,
        min_nodes_target=cfg.min_nodes_target,
        trim_boundary=True,
        max_units=cfg.max_units,
    )
    return [
        {
            "apk_sha": sha,
            "anchor_full_id": u.anchor_full_id,
            "anchor_category": u.anchor_category,
            "stats": u.stats,
        }
        for u in units
    ]


def main(**kwargs) -> None:
    cfg = SweepConfig(**kwargs)
    if cfg.sha_file:
        apk_shas = _load_sha_file(cfg.sha_file)
    else:
        apk_shas = [x.strip() for x in cfg.apk_shas.split(",") if x.strip()]
    if not apk_shas:
        raise ValueError("provide either sha_file or apk_shas")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tau_add_values = _parse_floats(cfg.tau_add_values)
    tau_qd_values = _parse_floats(cfg.tau_quality_delta_values)
    tau_sim_values = _parse_floats(cfg.tau_candidate_sim_values)

    results: list[dict[str, Any]] = []
    for tau_add, tau_qd, tau_sim in itertools.product(
        tau_add_values, tau_qd_values, tau_sim_values
    ):
        all_units: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=cfg.max_workers) as ex:
            futs = {
                ex.submit(_run_one_apk, cfg, sha, tau_add, tau_qd, tau_sim): sha
                for sha in apk_shas
            }
            for fut in as_completed(futs):
                all_units.extend(fut.result())

        score = _score_config(all_units)
        result = {
            "tau_add": tau_add,
            "tau_quality_delta": tau_qd,
            "tau_candidate_sim": tau_sim,
            "score": score,
            "units": all_units,
        }
        results.append(result)
        print(
            json.dumps(
                {
                    "tau_add": tau_add,
                    "tau_quality_delta": tau_qd,
                    "tau_candidate_sim": tau_sim,
                    **score,
                },
                ensure_ascii=False,
            )
        )

    # Rank configs by a balanced heuristic for physical interpretability:
    # prefer 4-15 nodes, low tiny ratio, decent label alignment, low conductance.
    def rank_key(r: dict[str, Any]) -> tuple[float, float, float, float]:
        s = r["score"]
        return (
            s["target_ratio"],
            -(s["tiny_ratio"]),
            s["physical_like_ratio"],
            -(s["mean_conductance"]),
        )

    results.sort(key=rank_key, reverse=True)

    summary = {
        "config": asdict(cfg),
        "results": results,
        "top5": results[:5],
    }
    out_path = out_dir / "sweep_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[sweep] wrote {out_path}")
    print(json.dumps(summary["top5"], ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    fire.Fire(main)
