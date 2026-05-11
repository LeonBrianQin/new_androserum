#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 0: discover SAPI / key API anchor candidates for one APK."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import fire

from androserum.behavior.anchors import discover_anchor_candidates


@dataclass
class AnchorDiscoveryConfig:
    apk_sha: str
    methods_dir: str = "data/methods"
    fcg_dir: str = "data/fcg"
    susi_sources: str = "third_party/susi/Ouput_CatSources_v0_9.txt"
    susi_sinks: str = "third_party/susi/Ouput_CatSinks_v0_9.txt"
    output_dir: str = "data/reports/anchor_discovery"
    top_k_per_category: int = 0
    min_degree: int = 1


def main(**kwargs) -> None:
    cfg = AnchorDiscoveryConfig(**kwargs)
    result = discover_anchor_candidates(
        apk_sha=cfg.apk_sha,
        methods_dir=cfg.methods_dir,
        fcg_dir=cfg.fcg_dir,
        susi_sources=cfg.susi_sources,
        susi_sinks=cfg.susi_sinks,
        top_k_per_category=cfg.top_k_per_category,
        min_degree=cfg.min_degree,
    )
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{result.apk_sha}.anchors.json"
    payload = {
        "config": asdict(cfg),
        "result": {
            "apk_sha": result.apk_sha,
            "stats": result.stats,
            "anchors": [asdict(a) for a in result.anchors],
            "context_candidates": [asdict(a) for a in result.context_candidates],
        },
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[phase0] wrote {out_path}")
    print(json.dumps(result.stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    fire.Fire(main)
