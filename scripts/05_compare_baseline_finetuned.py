#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare baseline vs Phase-4 finetuned embeddings on the same sampled methods.

Outputs:
  * one side-by-side UMAP scatter plot
  * one JSON with clustering metrics and sample stats

This script intentionally aligns rows via ``data/methods/<SHA>.parquet`` row
order, because the finetuned ``full_id`` object array may be unreadable across
some NumPy build combinations. The embedding matrices themselves are plain
float32 arrays and read fine, so we trust row-order parity:

  Phase 2 parquet rows -> Phase 3 baseline npz -> Phase 4 finetuned npz
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import random
import sys

import fire
import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import umap


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


def _normalize_label(raw: object) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, float) and math.isnan(raw):
        return None
    s = str(raw).strip()
    if not s or s.upper() == "NO_CATEGORY":
        return None
    return s


def _load_embedding_matrix(npz_path: Path) -> np.ndarray:
    z = np.load(npz_path, allow_pickle=False)
    emb = z["embedding"]
    if emb.ndim != 2:
        raise RuntimeError(f"expected 2D embedding matrix in {npz_path}, got {emb.shape}")
    return emb.astype(np.float32, copy=False)


@dataclass
class CompareConfig:
    sha_file: str = "configs/sha_dev_200.txt"
    methods_dir: str = "data/methods"
    baseline_dir: str = "data/embeddings/baseline"
    finetuned_dir: str = "data/embeddings/finetuned/p4_dev200_run1"
    out_dir: str = "data/reports"
    sample_size: int = 50000
    seed: int = 13
    min_cluster_size: int = 50
    min_samples: int = 5
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.0
    skip_filtered: bool = True
    labeled_only_for_metrics: bool = True


def _collect_aligned_rows(
    cfg: CompareConfig,
) -> tuple[np.ndarray, np.ndarray, list[str | None], list[str], dict]:
    rng = random.Random(cfg.seed)
    methods_root = Path(cfg.methods_dir)
    baseline_root = Path(cfg.baseline_dir)
    finetuned_root = Path(cfg.finetuned_dir)

    baseline_rows: list[np.ndarray] = []
    finetuned_rows: list[np.ndarray] = []
    labels: list[str | None] = []
    shas: list[str] = []
    skipped_empty_schema: list[str] = []

    for sha in load_sha_list(cfg.sha_file):
        pq = methods_root / f"{sha}.parquet"
        b_npz = baseline_root / f"{sha}.npz"
        f_npz = finetuned_root / f"{sha}.npz"
        if not (pq.is_file() and b_npz.is_file() and f_npz.is_file()):
            continue

        # Some local pyarrow builds are picky about projected-column reads on
        # these parquet files; read the full table and select the 2 fields in
        # pandas to keep the compare script robust across environments.
        df = pd.read_parquet(pq, engine="pyarrow")
        if df.empty and len(df.columns) == 0:
            skipped_empty_schema.append(sha)
            continue
        df = df[["filtered", "susi_dominant_cat"]]
        baseline = _load_embedding_matrix(b_npz)
        finetuned = _load_embedding_matrix(f_npz)

        if len(df) != len(baseline) or len(df) != len(finetuned):
            raise RuntimeError(
                f"row-count mismatch for {sha}: parquet={len(df)} "
                f"baseline={len(baseline)} finetuned={len(finetuned)}"
            )

        indices = list(range(len(df)))
        if cfg.skip_filtered:
            indices = [i for i in indices if not bool(df.iloc[i]["filtered"])]

        for i in indices:
            baseline_rows.append(baseline[i])
            finetuned_rows.append(finetuned[i])
            labels.append(_normalize_label(df.iloc[i]["susi_dominant_cat"]))
            shas.append(sha)

    if not baseline_rows:
        raise RuntimeError("no aligned rows found across methods/baseline/finetuned")

    if cfg.sample_size > 0 and len(baseline_rows) > cfg.sample_size:
        picks = sorted(rng.sample(range(len(baseline_rows)), k=cfg.sample_size))
        baseline_rows = [baseline_rows[i] for i in picks]
        finetuned_rows = [finetuned_rows[i] for i in picks]
        labels = [labels[i] for i in picks]
        shas = [shas[i] for i in picks]

    meta = {
        "skipped_empty_schema_apks": skipped_empty_schema,
        "skipped_empty_schema_count": len(skipped_empty_schema),
    }

    return (
        np.stack(baseline_rows, axis=0),
        np.stack(finetuned_rows, axis=0),
        labels,
        shas,
        meta,
    )


def _cluster_and_score(
    emb: np.ndarray,
    labels: list[str | None],
    cfg: CompareConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    reducer = umap.UMAP(
        n_neighbors=cfg.umap_n_neighbors,
        min_dist=cfg.umap_min_dist,
        metric="cosine",
        random_state=cfg.seed,
    )
    reduced = reducer.fit_transform(emb)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg.min_cluster_size,
        min_samples=cfg.min_samples,
        metric="euclidean",
    )
    pred = clusterer.fit_predict(reduced)

    metrics: dict[str, float | int | None] = {
        "n_points": int(len(pred)),
        "n_clusters_excluding_noise": int(len(set(pred)) - (1 if -1 in pred else 0)),
        "noise_points": int(np.sum(pred == -1)),
    }

    non_noise = pred != -1
    if np.sum(non_noise) >= 2 and len(set(pred[non_noise])) >= 2:
        metrics["silhouette_non_noise"] = float(silhouette_score(reduced[non_noise], pred[non_noise]))
    else:
        metrics["silhouette_non_noise"] = None

    metric_idx = [
        i for i, lab in enumerate(labels)
        if lab is not None and (not cfg.labeled_only_for_metrics or pred[i] != -1)
    ]
    if metric_idx:
        gold = [labels[i] for i in metric_idx]
        y_pred = [int(pred[i]) for i in metric_idx]
        metrics["labeled_points_for_metrics"] = len(metric_idx)
        metrics["nmi_vs_susi"] = float(normalized_mutual_info_score(gold, y_pred))
        metrics["ari_vs_susi"] = float(adjusted_rand_score(gold, y_pred))
    else:
        metrics["labeled_points_for_metrics"] = 0
        metrics["nmi_vs_susi"] = None
        metrics["ari_vs_susi"] = None

    return reduced, pred, metrics


def _plot_side_by_side(
    base_xy: np.ndarray,
    base_pred: np.ndarray,
    fin_xy: np.ndarray,
    fin_pred: np.ndarray,
    out_png: Path,
    title_suffix: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    datasets = [
        (axes[0], base_xy, base_pred, "Frozen DexBERT baseline"),
        (axes[1], fin_xy, fin_pred, "Phase 4 finetuned (A+B)"),
    ]

    for ax, xy, pred, title in datasets:
        colors = np.where(pred == -1, -1, pred)
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=colors,
            s=3,
            alpha=0.75,
            cmap="tab20",
            linewidths=0,
        )
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"UMAP + HDBSCAN comparison ({title_suffix})", fontsize=13)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def run(cfg: CompareConfig) -> dict:
    baseline, finetuned, labels, shas, collect_meta = _collect_aligned_rows(cfg)

    base_xy, base_pred, base_metrics = _cluster_and_score(baseline, labels, cfg)
    fin_xy, fin_pred, fin_metrics = _cluster_and_score(finetuned, labels, cfg)

    out_root = Path(cfg.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    stem = f"baseline_vs_finetuned_dev200_{baseline.shape[0]//1000}k"
    plot_path = out_root / f"{stem}.png"
    json_path = out_root / f"{stem}.json"

    _plot_side_by_side(
        base_xy,
        base_pred,
        fin_xy,
        fin_pred,
        plot_path,
        title_suffix=f"n={baseline.shape[0]}",
    )

    payload = {
        "config": asdict(cfg),
        "sampled_points": int(baseline.shape[0]),
        "sampled_apk_count": len(set(shas)),
        "collection": collect_meta,
        "baseline": base_metrics,
        "finetuned": fin_metrics,
        "notes": {
            "alignment": "rows aligned by methods parquet order; full_id object arrays not required",
            "baseline_extra_file_ignored": "0D64BB3C121E1986766505E182F16FB8DCC4188224F3094F99B9F905873DDC4A.npz",
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[compare] wrote plot: {plot_path}")
    print(f"[compare] wrote metrics: {json_path}")
    print("[compare] baseline metrics:")
    print(json.dumps(base_metrics, indent=2, sort_keys=True))
    print("[compare] finetuned metrics:")
    print(json.dumps(fin_metrics, indent=2, sort_keys=True))
    return payload


def main(**kwargs) -> None:
    cfg = CompareConfig(**kwargs)
    run(cfg)


if __name__ == "__main__":
    fire.Fire(main)
