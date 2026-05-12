#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch Phase 0 anchor discovery over many APKs and emit summary metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import fire
import pandas as pd

from androserum.behavior.anchors import discover_anchor_candidates


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
class BatchAnchorDiscoveryConfig:
    sha_file: str = "configs/sha_dev_200.txt"
    methods_dir: str = "data/methods"
    fcg_dir: str = "data/fcg"
    susi_sources: str = "third_party/susi/Ouput_CatSources_v0_9.txt"
    susi_sinks: str = "third_party/susi/Ouput_CatSinks_v0_9.txt"
    output_dir: str = "data/reports/anchor_discovery"
    per_apk_output_dir: str = "data/reports/anchor_discovery/per_apk"
    limit: int = 0
    top_k_per_category: int = 5
    min_degree: int = 1
    max_workers: int = 8


def _run_one(cfg: BatchAnchorDiscoveryConfig, sha: str) -> dict[str, Any]:
    result = discover_anchor_candidates(
        apk_sha=sha,
        methods_dir=cfg.methods_dir,
        fcg_dir=cfg.fcg_dir,
        susi_sources=cfg.susi_sources,
        susi_sinks=cfg.susi_sinks,
        top_k_per_category=cfg.top_k_per_category,
        min_degree=cfg.min_degree,
    )
    out = {
        "apk_sha": sha,
        "stats": result.stats,
        "hard_anchor_count": len(result.anchors),
        "context_candidate_count": len(result.context_candidates),
        "hard_anchor_confidence_mean": (
            float(sum(a.confidence for a in result.anchors) / len(result.anchors))
            if result.anchors
            else 0.0
        ),
        "hard_anchor_degree_mean": (
            float(sum(a.degree for a in result.anchors) / len(result.anchors))
            if result.anchors
            else 0.0
        ),
        "hard_anchor_categories": [a.category for a in result.anchors],
        "hard_anchor_full_ids": [a.full_id for a in result.anchors],
        "context_anchor_full_ids": [a.full_id for a in result.context_candidates[:100]],
    }
    payload = {
        "config": asdict(cfg),
        "result": {
            "apk_sha": result.apk_sha,
            "stats": result.stats,
            "anchors": [asdict(a) for a in result.anchors],
            "context_candidates": [asdict(a) for a in result.context_candidates],
        },
    }
    return out, payload


def main(**kwargs) -> None:
    cfg = BatchAnchorDiscoveryConfig(**kwargs)
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
            res, payload = fut.result()
            results.append(res)
            (per_apk_dir / f"{sha}.summary.json").write_text(
                json.dumps(res, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (out_dir / f"{sha}.anchors.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[batch-anchor] done {sha}")

    results.sort(key=lambda x: x["apk_sha"])

    # Aggregate quantitative metrics.
    hard_counts = [r["hard_anchor_count"] for r in results]
    context_counts = [r["context_candidate_count"] for r in results]
    all_hard_categories = Counter()
    all_hard_ids = Counter()
    all_conf = []
    all_deg = []
    apks_with_hard = 0
    for r in results:
        if r["hard_anchor_count"] > 0:
            apks_with_hard += 1
        all_hard_categories.update([c or "NONE" for c in r["hard_anchor_categories"]])
        all_hard_ids.update(r["hard_anchor_full_ids"])
        all_conf.extend([r["hard_anchor_confidence_mean"]] if r["hard_anchor_count"] else [])
        all_deg.extend([r["hard_anchor_degree_mean"]] if r["hard_anchor_count"] else [])

    summary = {
        "config": asdict(cfg),
        "n_apks": len(results),
        "apks_with_hard_anchor": apks_with_hard,
        "apk_coverage": float(apks_with_hard / max(1, len(results))),
        "hard_anchor_total": int(sum(hard_counts)),
        "context_candidate_total": int(sum(context_counts)),
        "hard_anchor_mean_per_apk": float(sum(hard_counts) / max(1, len(results))),
        "hard_anchor_median_per_apk": float(pd.Series(hard_counts).median()) if hard_counts else 0.0,
        "hard_anchor_min_per_apk": int(min(hard_counts)) if hard_counts else 0,
        "hard_anchor_max_per_apk": int(max(hard_counts)) if hard_counts else 0,
        "context_candidate_mean_per_apk": float(sum(context_counts) / max(1, len(results))),
        "context_candidate_median_per_apk": float(pd.Series(context_counts).median()) if context_counts else 0.0,
        "hard_anchor_category_counts": dict(all_hard_categories),
        "top_recurrent_hard_anchors": [
            {"full_id": fid, "count": cnt}
            for fid, cnt in all_hard_ids.most_common(50)
        ],
        "mean_anchor_confidence_proxy": float(sum(all_conf) / max(1, len(all_conf))) if all_conf else 0.0,
        "mean_anchor_degree_proxy": float(sum(all_deg) / max(1, len(all_deg))) if all_deg else 0.0,
        "results": results,
    }

    summary_path = out_dir / "batch_anchor_discovery_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    table = pd.DataFrame(
        [
            {
                "apk_sha": r["apk_sha"],
                "hard_anchor_count": r["hard_anchor_count"],
                "context_candidate_count": r["context_candidate_count"],
                "hard_anchor_confidence_mean": r["hard_anchor_confidence_mean"],
                "hard_anchor_degree_mean": r["hard_anchor_degree_mean"],
                "top_hard_category": Counter([c or "NONE" for c in r["hard_anchor_categories"]]).most_common(1)[0][0]
                if r["hard_anchor_categories"]
                else None,
            }
            for r in results
        ]
    )
    table.to_csv(out_dir / "batch_anchor_discovery_table.csv", index=False)

    print(f"[batch-anchor] wrote {summary_path}")
    print(json.dumps(
        {
            "n_apks": summary["n_apks"],
            "apk_coverage": summary["apk_coverage"],
            "hard_anchor_total": summary["hard_anchor_total"],
            "hard_anchor_mean_per_apk": summary["hard_anchor_mean_per_apk"],
            "hard_anchor_category_counts_top10": dict(Counter(summary["hard_anchor_category_counts"]).most_common(10)),
            "top_recurrent_hard_anchors": summary["top_recurrent_hard_anchors"][:10],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    fire.Fire(main)
