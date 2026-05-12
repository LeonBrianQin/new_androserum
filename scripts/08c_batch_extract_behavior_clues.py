#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch Phase 0.5 clue extraction over many APKs."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import fire

from androserum.behavior.clues import extract_method_clues


def _normalize_sha_token(s: str) -> str | None:
    s = s.strip().upper()
    if len(s) != 64:
        return None
    if any(c not in "0123456789ABCDEF" for c in s):
        return None
    return s


def load_sha_list(sha_file: str) -> list[str]:
    p = Path(sha_file)
    if not p.is_file():
        raise FileNotFoundError(f"sha_file not found: {p}")
    seen: set[str] = set()
    out: list[str] = []
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
class BatchClueConfig:
    sha_file: str = "configs/sha_dev_200.txt"
    methods_dir: str = "data/methods"
    output_dir: str = "data/reports/behavior_clues"
    per_apk_output_dir: str = "data/reports/behavior_clues/per_apk"
    limit: int = 0
    max_workers: int = 8


def _run_one(cfg: BatchClueConfig, sha: str) -> dict[str, Any]:
    result = extract_method_clues(apk_sha=sha, methods_dir=cfg.methods_dir)
    out = {
        "apk_sha": sha,
        "stats": result.stats,
    }
    return out, {
        "config": asdict(cfg),
        "result": {
            "apk_sha": result.apk_sha,
            "stats": result.stats,
            "clues": [asdict(c) for c in result.clues],
        },
    }


def main(**kwargs) -> None:
    cfg = BatchClueConfig(**kwargs)
    shas = load_sha_list(cfg.sha_file)
    if cfg.limit and cfg.limit > 0:
        shas = shas[: cfg.limit]

    out_dir = Path(cfg.output_dir)
    per_apk_dir = Path(cfg.per_apk_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_apk_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=cfg.max_workers) as ex:
        futs = {ex.submit(_run_one, cfg, sha): sha for sha in shas}
        for fut in as_completed(futs):
            sha = futs[fut]
            summary_row, full_payload = fut.result()
            results.append(summary_row)
            (per_apk_dir / f"{sha}.json").write_text(
                json.dumps(full_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            # Also keep compatibility with single-file readers used in Phase 1
            (out_dir / f"{sha}.clues.json").write_text(
                json.dumps(full_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[batch-clues] done {sha}")

    results.sort(key=lambda x: x["apk_sha"])
    n_network = sum(r["stats"]["n_network_like"] for r in results)
    n_file = sum(r["stats"]["n_file_like"] for r in results)
    n_reflection = sum(r["stats"]["n_reflection_like"] for r in results)
    n_db = sum(r["stats"]["n_db_like"] for r in results)
    n_location = sum(r["stats"]["n_location_like"] for r in results)
    n_identifier = sum(r["stats"]["n_identifier_like"] for r in results)

    summary = {
        "config": asdict(cfg),
        "n_apks": len(results),
        "total_network_like_methods": n_network,
        "total_file_like_methods": n_file,
        "total_reflection_like_methods": n_reflection,
        "total_db_like_methods": n_db,
        "total_location_like_methods": n_location,
        "total_identifier_like_methods": n_identifier,
        "results": results,
    }
    out_path = out_dir / "batch_behavior_clue_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[batch-clues] wrote {out_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    fire.Fire(main)
