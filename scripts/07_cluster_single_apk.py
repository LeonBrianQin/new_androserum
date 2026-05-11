#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cluster methods inside a single APK and emit human-readable summaries.

Default input is the Phase 6 relay+package embedding directory.
This script is intended for qualitative smoke tests:

  1. load one APK's method rows
  2. align them with one embedding npz
  3. run UMAP + HDBSCAN inside that APK only
  4. summarize each cluster by SuSi labels, API calls, classes, and method names
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any

import fire
import hdbscan
import numpy as np
import pandas as pd
import umap


def _normalize_sha_token(s: str) -> str | None:
    s = s.strip().upper()
    if len(s) != 64:
        return None
    if any(c not in "0123456789ABCDEF" for c in s):
        return None
    return s


def _method_name_from_sig(method_sig: str) -> str:
    head = method_sig.split("(", 1)[0]
    return head.strip()


def _class_tail(class_name: str) -> str:
    s = class_name.strip()
    if s.startswith("L") and s.endswith(";"):
        s = s[1:-1]
    return s.rsplit("/", 1)[-1]


def _normalize_label(raw: object) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, float) and np.isnan(raw):
        return None
    s = str(raw).strip()
    if not s or s.upper() == "NO_CATEGORY":
        return None
    return s


def _flatten_listish(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, np.ndarray):
        return [str(x).strip() for x in value.tolist() if str(x).strip()]
    s = str(value).strip()
    return [s] if s else []


def _api_tail(full_id: str) -> str:
    try:
        class_part, method_part = full_id.split("->", 1)
    except ValueError:
        return full_id
    return f"{_class_tail(class_part)}->{_method_name_from_sig(method_part)}"


@dataclass
class ClusterSingleApkConfig:
    apk_sha: str | None = None
    methods_dir: str = "data/methods"
    embeddings_dir: str = "data/gnn_embeddings/gnn_bgrl_relay_package"
    output_dir: str = "data/reports/single_apk_clusters"
    sample_apk_by: str = "largest"
    skip_filtered: bool = True
    min_cluster_size: int = 15
    min_samples: int = 3
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.0
    random_state: int = 13
    top_k_labels: int = 5
    top_k_apis: int = 10
    top_k_classes: int = 8
    top_k_methods: int = 10
    top_k_examples: int = 12


def _select_apk(cfg: ClusterSingleApkConfig) -> str:
    if cfg.apk_sha:
        sha = _normalize_sha_token(cfg.apk_sha)
        if sha is None:
            raise ValueError(f"invalid apk_sha: {cfg.apk_sha!r}")
        return sha

    methods_root = Path(cfg.methods_dir)
    emb_root = Path(cfg.embeddings_dir)
    candidates: list[tuple[int, str]] = []
    for npz in sorted(emb_root.glob("*.npz")):
        sha = _normalize_sha_token(npz.stem)
        if sha is None:
            continue
        pq = methods_root / f"{sha}.parquet"
        if not pq.is_file():
            continue
        z = np.load(npz, allow_pickle=True)
        n = int(z["embedding"].shape[0])
        candidates.append((n, sha))

    if not candidates:
        raise RuntimeError("no APKs found in embeddings_dir with matching methods parquet")

    if cfg.sample_apk_by == "largest":
        candidates.sort(reverse=True)
        return candidates[0][1]
    if cfg.sample_apk_by == "smallest":
        candidates.sort()
        return candidates[0][1]
    raise ValueError(f"unsupported sample_apk_by: {cfg.sample_apk_by}")


def _load_aligned_rows(cfg: ClusterSingleApkConfig, sha: str) -> tuple[pd.DataFrame, np.ndarray]:
    methods_path = Path(cfg.methods_dir) / f"{sha}.parquet"
    npz_path = Path(cfg.embeddings_dir) / f"{sha}.npz"
    if not methods_path.is_file():
        raise FileNotFoundError(f"methods parquet not found: {methods_path}")
    if not npz_path.is_file():
        raise FileNotFoundError(f"embedding npz not found: {npz_path}")

    df = pd.read_parquet(methods_path, engine="pyarrow")
    z = np.load(npz_path, allow_pickle=True)
    emb = z["embedding"].astype(np.float32, copy=False)
    full_ids = z["full_id"]

    if len(full_ids) != len(emb):
        raise RuntimeError(f"npz full_id / embedding length mismatch for {sha}")

    by_full_id = df.set_index("full_id", drop=False)
    missing = [fid for fid in full_ids if fid not in by_full_id.index]
    if missing:
        raise RuntimeError(f"{len(missing)} embedding ids missing in methods parquet for {sha}")

    aligned = by_full_id.loc[list(full_ids)].reset_index(drop=True)
    if cfg.skip_filtered:
        keep = ~aligned["filtered"].astype(bool).to_numpy()
        aligned = aligned.loc[keep].reset_index(drop=True)
        emb = emb[keep]

    if len(aligned) != len(emb):
        raise RuntimeError(f"aligned rows / embedding mismatch after filtering for {sha}")
    if len(aligned) == 0:
        raise RuntimeError(f"no rows left after filtering for {sha}")

    return aligned, emb


def _cluster(emb: np.ndarray, cfg: ClusterSingleApkConfig) -> tuple[np.ndarray, np.ndarray]:
    reducer = umap.UMAP(
        n_neighbors=cfg.umap_n_neighbors,
        min_dist=cfg.umap_min_dist,
        metric="cosine",
        random_state=cfg.random_state,
    )
    xy = reducer.fit_transform(emb)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg.min_cluster_size,
        min_samples=cfg.min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(xy)
    return xy, labels


def _top_counter(items: list[str], k: int) -> list[dict[str, Any]]:
    c = Counter(items)
    return [{"item": item, "count": count} for item, count in c.most_common(k)]


def _summarize_cluster(
    df: pd.DataFrame,
    cluster_id: int,
    cfg: ClusterSingleApkConfig,
) -> dict[str, Any]:
    rows = df[df["cluster_id"] == cluster_id].copy()
    labels = [_normalize_label(v) for v in rows["susi_dominant_cat"].tolist()]
    label_items = [x for x in labels if x is not None]

    api_items: list[str] = []
    for v in rows["api_calls"].tolist():
        api_items.extend(_api_tail(x) for x in _flatten_listish(v))

    class_items = [_class_tail(v) for v in rows["class_name"].tolist()]
    method_items = [_method_name_from_sig(v) for v in rows["method_sig"].tolist()]

    examples = []
    for _, rec in rows.head(cfg.top_k_examples).iterrows():
        examples.append(
            {
                "class": rec["class_name"],
                "method_sig": rec["method_sig"],
                "method_name": _method_name_from_sig(rec["method_sig"]),
                "susi_dominant_cat": _normalize_label(rec["susi_dominant_cat"]),
                "api_calls_preview": [_api_tail(x) for x in _flatten_listish(rec["api_calls"])[:5]],
            }
        )

    return {
        "cluster_id": int(cluster_id),
        "size": int(len(rows)),
        "dominant_susi_labels": _top_counter(label_items, cfg.top_k_labels),
        "top_api_calls": _top_counter(api_items, cfg.top_k_apis),
        "top_classes": _top_counter(class_items, cfg.top_k_classes),
        "top_method_names": _top_counter(method_items, cfg.top_k_methods),
        "example_methods": examples,
    }


def run(cfg: ClusterSingleApkConfig) -> dict[str, Any]:
    sha = _select_apk(cfg)
    df, emb = _load_aligned_rows(cfg, sha)
    xy, cluster_labels = _cluster(emb, cfg)

    df = df.copy()
    df["cluster_id"] = cluster_labels
    df["umap_x"] = xy[:, 0]
    df["umap_y"] = xy[:, 1]

    cluster_ids = sorted(int(x) for x in set(cluster_labels) if int(x) != -1)
    clusters = [_summarize_cluster(df, cid, cfg) for cid in cluster_ids]
    clusters.sort(key=lambda x: x["size"], reverse=True)

    out = {
        "apk_sha": sha,
        "config": asdict(cfg),
        "n_methods_clustered": int(len(df)),
        "n_clusters_excluding_noise": int(len(cluster_ids)),
        "noise_points": int(np.sum(cluster_labels == -1)),
        "cluster_summaries": clusters,
    }

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sha}.cluster_summary.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[single-apk-cluster] wrote {out_path}")
    print(
        json.dumps(
            {
                "apk_sha": sha,
                "n_methods_clustered": out["n_methods_clustered"],
                "n_clusters_excluding_noise": out["n_clusters_excluding_noise"],
                "noise_points": out["noise_points"],
                "largest_clusters": [
                    {
                        "cluster_id": c["cluster_id"],
                        "size": c["size"],
                        "dominant_susi_labels": c["dominant_susi_labels"][:3],
                        "top_method_names": c["top_method_names"][:5],
                    }
                    for c in clusters[:5]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return out


def main(**kwargs) -> None:
    cfg = ClusterSingleApkConfig(**kwargs)
    run(cfg)


if __name__ == "__main__":
    fire.Fire(main)
