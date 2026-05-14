"""Phase 6 dataset builders over Phase 5 FCG sidecars."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from androserum.train.dataset import load_sha_list

__all__ = [
    "EXTERNAL_NODE_KIND",
    "FcgGraphDataset",
    "FcgGraphEntry",
    "FcgGraphSample",
    "INTERNAL_NODE_KIND",
    "INTERNAL_UNALIGNED_NODE_KIND",
    "external_family_key",
]

INTERNAL_NODE_KIND = 0
EXTERNAL_NODE_KIND = 1
INTERNAL_UNALIGNED_NODE_KIND = 2


@dataclass(frozen=True)
class FcgGraphEntry:
    """Lightweight on-disk index for one APK graph."""

    apk_sha: str
    aligned_nodes_path: Path
    internal_edges_path: Path
    boundary_edges_path: Path
    summary_path: Path
    summary: dict


@dataclass
class FcgGraphSample:
    """One APK graph ready for Phase 6 training / export."""

    apk_sha: str
    x: torch.Tensor
    edge_index: torch.Tensor
    node_kind: torch.Tensor
    filtered_mask: torch.Tensor
    internal_mask: torch.Tensor
    family_keys: list[str | None]
    family_ids: torch.Tensor
    full_ids: list[str]
    internal_full_ids: list[str]
    internal_node_indices: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def num_internal_nodes(self) -> int:
        return int(self.internal_mask.sum().item())


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_sha_token(s: str) -> str | None:
    s = s.strip().upper()
    if len(s) != 64:
        return None
    if any(c not in "0123456789ABCDEF" for c in s):
        return None
    return s


def _load_graph_shas(sha_file: str | None, fcg_dir: Path) -> list[str]:
    if sha_file:
        return load_sha_list(sha_file, fcg_dir)
    out: list[str] = []
    for p in sorted(fcg_dir.glob("*.summary.json")):
        sha = _normalize_sha_token(p.name.split(".", 1)[0])
        if sha is not None:
            out.append(sha)
    return out


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return pd.read_parquet(path, engine="pyarrow")


def _load_embedding_matrix(npz_path: Path) -> np.ndarray:
    z = np.load(npz_path, allow_pickle=False)
    emb = z["embedding"]
    if emb.ndim != 2:
        raise RuntimeError(f"expected 2D embedding matrix in {npz_path}, got {emb.shape}")
    return emb.astype(np.float32, copy=False)


def external_family_key(full_id: str) -> str:
    """Coarse family id for external relay nodes.

    The goal is not perfect API semantics; it is just a stable, low-cardinality
    prior that lets us test whether a tiny amount of external typing helps.
    """
    if not full_id.startswith("L"):
        return "UNKNOWN"
    cls = full_id[1:].split(";->", 1)[0]
    parts = [p for p in cls.split("/") if p]
    if not parts:
        return "UNKNOWN"
    if parts[0] in {"android", "androidx", "java", "javax", "kotlin", "dalvik"}:
        return parts[0]
    if parts[0] in {"com", "org", "net"} and len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return parts[0]


def _dedupe_edges(src_dst_pairs: list[tuple[int, int]]) -> torch.Tensor:
    if not src_dst_pairs:
        return torch.zeros((2, 0), dtype=torch.long)
    uniq = sorted(set(src_dst_pairs))
    arr = np.array(uniq, dtype=np.int64)
    return torch.from_numpy(arr.T.copy())


def _bool_series(df: pd.DataFrame, col: str) -> list[bool]:
    if col not in df.columns:
        return [False] * len(df)
    return [bool(x) for x in df[col].tolist()]


class FcgGraphDataset:
    """On-demand loader over Phase 5 FCG sidecars and Phase 4 embeddings."""

    def __init__(
        self,
        *,
        entries: list[FcgGraphEntry],
        embeddings_dir: str | Path,
        graph_mode: str,
        external_prior_mode: str,
        add_reverse_edges: bool,
        family_to_id: dict[str, int] | None = None,
    ) -> None:
        self.entries = entries
        self.embeddings_dir = Path(embeddings_dir)
        self.graph_mode = graph_mode
        self.external_prior_mode = external_prior_mode
        self.add_reverse_edges = add_reverse_edges
        self.family_to_id = family_to_id or {}

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> FcgGraphSample:
        return self.load(index)

    @classmethod
    def from_dirs(
        cls,
        *,
        fcg_dir: str | Path,
        embeddings_dir: str | Path,
        sha_file: str | None = "configs/sha_dev_200.txt",
        limit: int = 0,
        graph_mode: str = "internal_only",
        external_prior_mode: str = "none",
        add_reverse_edges: bool = True,
        family_to_id: dict[str, int] | None = None,
    ) -> "FcgGraphDataset":
        if graph_mode not in {"internal_only", "relay"}:
            raise ValueError(f"unsupported graph_mode: {graph_mode}")
        if external_prior_mode not in {"none", "global", "package"}:
            raise ValueError(f"unsupported external_prior_mode: {external_prior_mode}")

        fcg_root = Path(fcg_dir)
        emb_root = Path(embeddings_dir)
        shas = _load_graph_shas(sha_file, fcg_root)
        if limit and limit > 0:
            shas = shas[:limit]

        entries: list[FcgGraphEntry] = []
        missing: list[str] = []
        zero_method_shas: list[str] = []
        for sha in shas:
            summary_path = fcg_root / f"{sha}.summary.json"
            aligned_nodes_path = fcg_root / f"{sha}.aligned_nodes.parquet"
            internal_edges_path = fcg_root / f"{sha}.internal_edges.parquet"
            boundary_edges_path = fcg_root / f"{sha}.boundary_edges.parquet"
            embedding_path = emb_root / f"{sha}.npz"
            if not (
                summary_path.is_file()
                and aligned_nodes_path.is_file()
                and internal_edges_path.is_file()
                and boundary_edges_path.is_file()
                and embedding_path.is_file()
            ):
                missing.append(sha)
                continue
            summary = _load_json(summary_path)
            if int(summary.get("methods_rows", 0)) == 0:
                zero_method_shas.append(sha)
                continue
            entries.append(
                FcgGraphEntry(
                    apk_sha=sha,
                    aligned_nodes_path=aligned_nodes_path,
                    internal_edges_path=internal_edges_path,
                    boundary_edges_path=boundary_edges_path,
                    summary_path=summary_path,
                    summary=summary,
                )
            )

        resolved_family_to_id: dict[str, int] = dict(family_to_id or {})
        if graph_mode == "relay" and external_prior_mode == "package" and not resolved_family_to_id:
            families: set[str] = set()
            for entry in entries:
                bdf = _read_parquet(entry.boundary_edges_path)
                if bdf.empty:
                    continue
                for full_id, other_external in zip(
                    bdf["other_full_id"].astype(str),
                    _bool_series(bdf, "other_external"),
                ):
                    if other_external:
                        families.add(external_family_key(full_id))
            resolved_family_to_id = {fam: idx for idx, fam in enumerate(sorted(families))}

        ds = cls(
            entries=entries,
            embeddings_dir=emb_root,
            graph_mode=graph_mode,
            external_prior_mode=external_prior_mode,
            add_reverse_edges=add_reverse_edges,
            family_to_id=resolved_family_to_id,
        )
        ds._missing_shas = missing
        ds._zero_method_shas = zero_method_shas
        return ds

    def stats(self) -> dict[str, object]:
        extra = [int(e.summary.get("extra_internal_graph_nodes_count", 0)) for e in self.entries]
        methods = [int(e.summary.get("methods_rows", 0)) for e in self.entries]
        return {
            "graphs_total": len(self.entries),
            "missing_graph_or_embedding_inputs": len(getattr(self, "_missing_shas", [])),
            "zero_method_graphs_skipped": len(getattr(self, "_zero_method_shas", [])),
            "graph_mode": self.graph_mode,  # type: ignore[dict-item]
            "external_prior_mode": self.external_prior_mode,  # type: ignore[dict-item]
            "external_family_vocab": len(self.family_to_id),
            "methods_rows_total": int(sum(methods)),
            "methods_rows_median": int(np.median(methods)) if methods else 0,
            "extra_internal_nodes_median": int(np.median(extra)) if extra else 0,
        }

    def load(self, index: int) -> FcgGraphSample:
        entry = self.entries[index]
        aligned_df = _read_parquet(entry.aligned_nodes_path)
        internal_df = _read_parquet(entry.internal_edges_path)
        boundary_df = _read_parquet(entry.boundary_edges_path)
        emb = _load_embedding_matrix(self.embeddings_dir / f"{entry.apk_sha}.npz")

        if len(aligned_df) != len(emb):
            raise RuntimeError(
                f"row-count mismatch for {entry.apk_sha}: "
                f"aligned_nodes={len(aligned_df)} embedding={len(emb)}"
            )

        internal_full_ids = aligned_df["full_id"].astype(str).tolist()
        filtered_flags = torch.tensor(_bool_series(aligned_df, "filtered"), dtype=torch.bool)
        x_internal = torch.from_numpy(emb.copy())

        full_ids = list(internal_full_ids)
        family_keys: list[str | None] = [None] * len(internal_full_ids)
        node_kind = [INTERNAL_NODE_KIND] * len(internal_full_ids)
        relay_index: dict[str, int] = {}

        edge_pairs: list[tuple[int, int]] = [
            (int(src), int(dst))
            for src, dst in zip(internal_df.get("src_idx", []), internal_df.get("dst_idx", []))
        ]

        if self.graph_mode == "relay" and not boundary_df.empty:
            for rec in boundary_df.to_dict("records"):
                other_full_id = str(rec["other_full_id"])
                relay_idx = relay_index.get(other_full_id)
                if relay_idx is None:
                    relay_idx = len(full_ids)
                    relay_index[other_full_id] = relay_idx
                    full_ids.append(other_full_id)
                    is_external = bool(rec.get("other_external", False))
                    node_kind.append(EXTERNAL_NODE_KIND if is_external else INTERNAL_UNALIGNED_NODE_KIND)
                    if is_external and self.external_prior_mode == "package":
                        family_keys.append(external_family_key(other_full_id))
                    else:
                        family_keys.append(None)

                internal_idx = int(rec["internal_idx"])
                if str(rec["direction"]) == "out":
                    edge_pairs.append((internal_idx, relay_idx))
                else:
                    edge_pairs.append((relay_idx, internal_idx))

        if self.add_reverse_edges:
            edge_pairs.extend((dst, src) for src, dst in list(edge_pairs))

        edge_index = _dedupe_edges(edge_pairs)

        if len(full_ids) > len(internal_full_ids):
            extra_rows = len(full_ids) - len(internal_full_ids)
            x_extra = torch.zeros((extra_rows, x_internal.shape[1]), dtype=x_internal.dtype)
            x = torch.cat([x_internal, x_extra], dim=0)
        else:
            x = x_internal

        internal_mask = torch.zeros(len(full_ids), dtype=torch.bool)
        internal_mask[: len(internal_full_ids)] = True

        filtered_mask = torch.zeros(len(full_ids), dtype=torch.bool)
        filtered_mask[: len(internal_full_ids)] = filtered_flags

        node_kind_tensor = torch.tensor(node_kind, dtype=torch.long)
        family_ids = torch.full((len(full_ids),), -1, dtype=torch.long)
        if self.graph_mode == "relay" and self.external_prior_mode == "package":
            for idx, fam in enumerate(family_keys):
                if fam is not None:
                    family_ids[idx] = self.family_to_id[fam]

        return FcgGraphSample(
            apk_sha=entry.apk_sha,
            x=x,
            edge_index=edge_index,
            node_kind=node_kind_tensor,
            filtered_mask=filtered_mask,
            internal_mask=internal_mask,
            family_keys=family_keys,
            family_ids=family_ids,
            full_ids=full_ids,
            internal_full_ids=internal_full_ids,
            internal_node_indices=torch.arange(len(internal_full_ids), dtype=torch.long),
        )
