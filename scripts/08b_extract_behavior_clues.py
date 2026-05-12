#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 0.5: extract lightweight symbolic clues for one APK."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import fire

from androserum.behavior.clues import extract_method_clues


@dataclass
class ClueExtractionConfig:
    apk_sha: str
    methods_dir: str = "data/methods"
    output_dir: str = "data/reports/behavior_clues"


def main(**kwargs) -> None:
    cfg = ClueExtractionConfig(**kwargs)
    result = extract_method_clues(
        apk_sha=cfg.apk_sha,
        methods_dir=cfg.methods_dir,
    )
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{result.apk_sha}.clues.json"
    payload = {
        "config": asdict(cfg),
        "result": {
            "apk_sha": result.apk_sha,
            "stats": result.stats,
            "clues": [asdict(c) for c in result.clues],
        },
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[phase0.5] wrote {out_path}")
    print(json.dumps(result.stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    fire.Fire(main)

