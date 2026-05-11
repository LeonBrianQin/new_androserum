"""Tests for Phase 6 graph dataset construction."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

from androserum.gnn.dataset import (
    EXTERNAL_NODE_KIND,
    FcgGraphDataset,
    INTERNAL_NODE_KIND,
    INTERNAL_UNALIGNED_NODE_KIND,
)


def _write_graph_bundle(tmp_path: Path, sha: str) -> tuple[Path, Path, Path]:
    fcg_dir = tmp_path / "fcg"
    emb_dir = tmp_path / "emb"
    fcg_dir.mkdir()
    emb_dir.mkdir()

    aligned = pd.DataFrame(
        {
            "apk_sha": [sha, sha],
            "node_idx": [0, 1],
            "full_id": [
                "Lcom/example/A;->alpha()V",
                "Lcom/example/B;->beta()V",
            ],
            "class_name": ["Lcom/example/A;", "Lcom/example/B;"],
            "method_sig": ["alpha()V", "beta()V"],
            "filtered": [False, True],
            "graph_present": [True, True],
            "entrypoint": [False, False],
            "internal_in_degree": [0, 1],
            "internal_out_degree": [1, 0],
            "external_in_degree": [1, 0],
            "external_out_degree": [0, 1],
        }
    )
    aligned.to_parquet(fcg_dir / f"{sha}.aligned_nodes.parquet", index=False, engine="pyarrow")

    internal_edges = pd.DataFrame(
        {
            "apk_sha": [sha],
            "src_idx": [0],
            "dst_idx": [1],
            "src_full_id": ["Lcom/example/A;->alpha()V"],
            "dst_full_id": ["Lcom/example/B;->beta()V"],
        }
    )
    internal_edges.to_parquet(
        fcg_dir / f"{sha}.internal_edges.parquet",
        index=False,
        engine="pyarrow",
    )

    boundary_edges = pd.DataFrame(
        {
            "apk_sha": [sha, sha],
            "internal_idx": [1, 0],
            "internal_full_id": [
                "Lcom/example/B;->beta()V",
                "Lcom/example/A;->alpha()V",
            ],
            "other_full_id": [
                "Ljava/lang/String;->valueOf(I)Ljava/lang/String;",
                "Lcom/example/Hidden;->relay()V",
            ],
            "direction": ["out", "in"],
            "other_external": [True, False],
        }
    )
    boundary_edges.to_parquet(
        fcg_dir / f"{sha}.boundary_edges.parquet",
        index=False,
        engine="pyarrow",
    )

    summary = {
        "apk_sha": sha,
        "methods_rows": 2,
        "graph_nodes_total": 4,
        "graph_edges_total": 3,
        "graph_internal_nodes_total": 4,
        "graph_external_nodes_total": 1,
        "aligned_graph_present": 2,
        "missing_graph_nodes_count": 0,
        "extra_internal_graph_nodes_count": 1,
        "internal_edges_count": 1,
        "boundary_edges_count": 2,
        "entrypoint_nodes_count": 0,
        "missing_graph_nodes_sample": [],
        "extra_internal_graph_nodes_sample": ["Lcom/example/Hidden;->relay()V"],
    }
    (fcg_dir / f"{sha}.summary.json").write_text(json.dumps(summary), encoding="utf-8")

    np.savez_compressed(
        emb_dir / f"{sha}.npz",
        full_id=np.array(
            [
                "Lcom/example/A;->alpha()V",
                "Lcom/example/B;->beta()V",
            ],
            dtype=object,
        ),
        embedding=np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32),
    )

    sha_file = tmp_path / "sha.txt"
    sha_file.write_text(sha + "\n", encoding="utf-8")
    return fcg_dir, emb_dir, sha_file


def test_internal_only_graph_mode(tmp_path: Path):
    sha = "A" * 64
    fcg_dir, emb_dir, sha_file = _write_graph_bundle(tmp_path, sha)
    ds = FcgGraphDataset.from_dirs(
        fcg_dir=fcg_dir,
        embeddings_dir=emb_dir,
        sha_file=str(sha_file),
        graph_mode="internal_only",
        external_prior_mode="none",
        add_reverse_edges=True,
    )
    sample = ds.load(0)
    assert sample.apk_sha == sha
    assert sample.num_nodes == 2
    assert sample.num_internal_nodes == 2
    assert sample.edge_index.shape[1] == 2  # forward + reverse
    assert sample.node_kind.tolist() == [INTERNAL_NODE_KIND, INTERNAL_NODE_KIND]
    assert sample.family_ids.tolist() == [-1, -1]


def test_relay_graph_mode_adds_boundary_nodes(tmp_path: Path):
    sha = "B" * 64
    fcg_dir, emb_dir, sha_file = _write_graph_bundle(tmp_path, sha)
    ds = FcgGraphDataset.from_dirs(
        fcg_dir=fcg_dir,
        embeddings_dir=emb_dir,
        sha_file=str(sha_file),
        graph_mode="relay",
        external_prior_mode="none",
        add_reverse_edges=True,
    )
    sample = ds.load(0)
    assert sample.num_nodes == 4
    assert sample.num_internal_nodes == 2
    assert Counter(sample.node_kind.tolist()) == {
        INTERNAL_NODE_KIND: 2,
        EXTERNAL_NODE_KIND: 1,
        INTERNAL_UNALIGNED_NODE_KIND: 1,
    }
    assert all(x == -1 for x in sample.family_ids.tolist())


def test_relay_package_prior_builds_family_vocab(tmp_path: Path):
    sha = "C" * 64
    fcg_dir, emb_dir, sha_file = _write_graph_bundle(tmp_path, sha)
    ds = FcgGraphDataset.from_dirs(
        fcg_dir=fcg_dir,
        embeddings_dir=emb_dir,
        sha_file=str(sha_file),
        graph_mode="relay",
        external_prior_mode="package",
        add_reverse_edges=False,
    )
    sample = ds.load(0)
    assert "java" in ds.family_to_id
    assert sample.num_nodes == 4
    # One external relay node gets a family id; the internal-unaligned relay does not.
    assert sorted(sample.family_ids.tolist()) == [-1, -1, -1, 0]
