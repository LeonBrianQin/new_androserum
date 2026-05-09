"""Phase 5: FCG extraction and method-row alignment helpers.

This module intentionally separates two concerns:

1. Use Androguard to extract the APK-level call graph.
2. Align internal graph nodes back to Phase 2/3 ``full_id`` row order so Phase 6
   can consume node features without any fuzzy matching.

The aligned node table is written in the same order as
``data/methods/<SHA>.parquet`` and ``data/embeddings/*/<SHA>.npz``. That makes
Phase 6 straightforward: row ``i`` in the aligned node parquet corresponds to
row ``i`` in the embedding matrix.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

from loguru import logger
import networkx as nx
import pandas as pd
from androguard.misc import AnalyzeAPK

from androserum.data.method_parquet import read_methods_parquet
from androserum.data.schema import MethodRecord, make_full_id

__all__ = [
    "FcgAlignedNodeRecord",
    "FcgBoundaryEdgeRecord",
    "FcgBuildSummary",
    "FcgInternalEdgeRecord",
    "align_call_graph_to_method_rows",
    "extract_call_graph_for_apk",
    "extract_fcg_bundle_for_apk",
    "method_to_full_id",
    "write_fcg_bundle",
]


@dataclass(frozen=True)
class FcgAlignedNodeRecord:
    """One internal APK method aligned to its Phase 2 row index."""

    apk_sha: str
    node_idx: int
    full_id: str
    class_name: str
    method_sig: str
    filtered: bool
    graph_present: bool
    entrypoint: bool
    internal_in_degree: int
    internal_out_degree: int
    external_in_degree: int
    external_out_degree: int


@dataclass(frozen=True)
class FcgInternalEdgeRecord:
    """A call edge between two internal methods in aligned node-index space."""

    apk_sha: str
    src_idx: int
    dst_idx: int
    src_full_id: str
    dst_full_id: str


@dataclass(frozen=True)
class FcgBoundaryEdgeRecord:
    """A call edge between one aligned internal node and one non-aligned node."""

    apk_sha: str
    internal_idx: int
    internal_full_id: str
    other_full_id: str
    direction: str
    other_external: bool


@dataclass(frozen=True)
class FcgBuildSummary:
    """Compact diagnostics for one APK's Phase 5 extraction output."""

    apk_sha: str
    methods_rows: int
    graph_nodes_total: int
    graph_edges_total: int
    graph_internal_nodes_total: int
    graph_external_nodes_total: int
    aligned_graph_present: int
    missing_graph_nodes_count: int
    extra_internal_graph_nodes_count: int
    internal_edges_count: int
    boundary_edges_count: int
    entrypoint_nodes_count: int
    missing_graph_nodes_sample: list[str]
    extra_internal_graph_nodes_sample: list[str]


def _method_sig(method_name: str, descriptor: str) -> str:
    return f"{method_name}{descriptor}"


def method_to_full_id(method: object) -> str:
    """Normalize an Androguard method-like object to our canonical ``full_id``."""
    class_name = str(method.get_class_name()).strip()
    method_name = str(method.get_name()).strip()
    # Androguard may pretty-print descriptors with spaces after commas /
    # between adjacent reference types. Our Phase 2 ``full_id`` strings are
    # space-free Dalvik descriptors, so we collapse all ASCII spaces here.
    descriptor = str(method.get_descriptor()).strip().replace(" ", "")
    return make_full_id(class_name, _method_sig(method_name, descriptor))


def extract_call_graph_for_apk(
    apk_path: str | Path,
    *,
    no_isolated: bool = False,
    entry_points: list[str] | None = None,
) -> nx.DiGraph:
    """Return the Androguard call graph for one APK."""
    logger.remove()  # avoid noisy Androguard per-method info logs
    _, _, dx = AnalyzeAPK(str(apk_path))
    return dx.get_call_graph(
        no_isolated=no_isolated,
        entry_points=list(entry_points or []),
    )


def align_call_graph_to_method_rows(
    apk_sha: str,
    method_rows: list[MethodRecord],
    call_graph: nx.DiGraph,
    *,
    include_boundary_edges: bool = True,
) -> tuple[
    list[FcgAlignedNodeRecord],
    list[FcgInternalEdgeRecord],
    list[FcgBoundaryEdgeRecord],
    FcgBuildSummary,
]:
    """Project a raw call graph into Phase-2 row order plus aligned edge tables."""
    aligned_index = {row.full_id: idx for idx, row in enumerate(method_rows)}

    node_meta: dict[str, dict] = {}
    graph_internal_ids: set[str] = set()
    graph_external_ids: set[str] = set()

    for node, attrs in call_graph.nodes(data=True):
        full_id = method_to_full_id(node)
        external = bool(attrs.get("external", False))
        meta = {
            "full_id": full_id,
            "entrypoint": bool(attrs.get("entrypoint", False)),
            "external": external,
        }
        if full_id not in node_meta:
            node_meta[full_id] = meta
        if external:
            graph_external_ids.add(full_id)
        else:
            graph_internal_ids.add(full_id)

    n_rows = len(method_rows)
    internal_in = [0] * n_rows
    internal_out = [0] * n_rows
    external_in = [0] * n_rows
    external_out = [0] * n_rows

    internal_edges: list[FcgInternalEdgeRecord] = []
    boundary_edges: list[FcgBoundaryEdgeRecord] = []

    for src_node, dst_node in call_graph.edges():
        src_id = method_to_full_id(src_node)
        dst_id = method_to_full_id(dst_node)
        src_idx = aligned_index.get(src_id)
        dst_idx = aligned_index.get(dst_id)

        if src_idx is not None and dst_idx is not None:
            internal_out[src_idx] += 1
            internal_in[dst_idx] += 1
            internal_edges.append(
                FcgInternalEdgeRecord(
                    apk_sha=apk_sha.upper(),
                    src_idx=src_idx,
                    dst_idx=dst_idx,
                    src_full_id=src_id,
                    dst_full_id=dst_id,
                )
            )
            continue

        if not include_boundary_edges:
            continue

        if src_idx is not None:
            external_out[src_idx] += 1
            boundary_edges.append(
                FcgBoundaryEdgeRecord(
                    apk_sha=apk_sha.upper(),
                    internal_idx=src_idx,
                    internal_full_id=src_id,
                    other_full_id=dst_id,
                    direction="out",
                    other_external=bool(node_meta.get(dst_id, {}).get("external", False)),
                )
            )
        elif dst_idx is not None:
            external_in[dst_idx] += 1
            boundary_edges.append(
                FcgBoundaryEdgeRecord(
                    apk_sha=apk_sha.upper(),
                    internal_idx=dst_idx,
                    internal_full_id=dst_id,
                    other_full_id=src_id,
                    direction="in",
                    other_external=bool(node_meta.get(src_id, {}).get("external", False)),
                )
            )

    aligned_nodes: list[FcgAlignedNodeRecord] = []
    for idx, row in enumerate(method_rows):
        meta = node_meta.get(row.full_id, {})
        aligned_nodes.append(
            FcgAlignedNodeRecord(
                apk_sha=row.apk_sha,
                node_idx=idx,
                full_id=row.full_id,
                class_name=row.class_name,
                method_sig=row.method_sig,
                filtered=row.filtered,
                graph_present=row.full_id in graph_internal_ids,
                entrypoint=bool(meta.get("entrypoint", False)),
                internal_in_degree=internal_in[idx],
                internal_out_degree=internal_out[idx],
                external_in_degree=external_in[idx],
                external_out_degree=external_out[idx],
            )
        )

    missing_graph_nodes = [row.full_id for row in method_rows if row.full_id not in graph_internal_ids]
    extra_internal_graph_nodes = sorted(graph_internal_ids.difference(aligned_index))
    entrypoint_nodes = sum(1 for row in aligned_nodes if row.entrypoint)

    summary = FcgBuildSummary(
        apk_sha=apk_sha.upper(),
        methods_rows=n_rows,
        graph_nodes_total=call_graph.number_of_nodes(),
        graph_edges_total=call_graph.number_of_edges(),
        graph_internal_nodes_total=len(graph_internal_ids),
        graph_external_nodes_total=len(graph_external_ids),
        aligned_graph_present=sum(1 for row in aligned_nodes if row.graph_present),
        missing_graph_nodes_count=len(missing_graph_nodes),
        extra_internal_graph_nodes_count=len(extra_internal_graph_nodes),
        internal_edges_count=len(internal_edges),
        boundary_edges_count=len(boundary_edges),
        entrypoint_nodes_count=entrypoint_nodes,
        missing_graph_nodes_sample=missing_graph_nodes[:20],
        extra_internal_graph_nodes_sample=extra_internal_graph_nodes[:20],
    )

    return aligned_nodes, internal_edges, boundary_edges, summary


def _write_records_parquet(records: list[dataclass], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if records:
        df = pd.DataFrame([asdict(r) for r in records], columns=columns)
    else:
        df = pd.DataFrame(columns=columns)
    df.to_parquet(path, index=False, engine="pyarrow")


def write_fcg_bundle(
    apk_sha: str,
    out_dir: str | Path,
    aligned_nodes: list[FcgAlignedNodeRecord],
    internal_edges: list[FcgInternalEdgeRecord],
    boundary_edges: list[FcgBoundaryEdgeRecord],
    summary: FcgBuildSummary,
) -> dict[str, str]:
    """Write one APK's Phase 5 sidecars to ``out_dir``."""
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    sha = apk_sha.upper()

    aligned_path = out_root / f"{sha}.aligned_nodes.parquet"
    internal_edges_path = out_root / f"{sha}.internal_edges.parquet"
    boundary_edges_path = out_root / f"{sha}.boundary_edges.parquet"
    summary_path = out_root / f"{sha}.summary.json"

    _write_records_parquet(
        aligned_nodes,
        aligned_path,
        columns=[
            "apk_sha",
            "node_idx",
            "full_id",
            "class_name",
            "method_sig",
            "filtered",
            "graph_present",
            "entrypoint",
            "internal_in_degree",
            "internal_out_degree",
            "external_in_degree",
            "external_out_degree",
        ],
    )
    _write_records_parquet(
        internal_edges,
        internal_edges_path,
        columns=["apk_sha", "src_idx", "dst_idx", "src_full_id", "dst_full_id"],
    )
    _write_records_parquet(
        boundary_edges,
        boundary_edges_path,
        columns=[
            "apk_sha",
            "internal_idx",
            "internal_full_id",
            "other_full_id",
            "direction",
            "other_external",
        ],
    )
    summary_path.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "aligned_nodes": str(aligned_path),
        "internal_edges": str(internal_edges_path),
        "boundary_edges": str(boundary_edges_path),
        "summary": str(summary_path),
    }


def extract_fcg_bundle_for_apk(
    apk_path: str | Path,
    methods_parquet_path: str | Path,
    out_dir: str | Path,
    *,
    apk_sha: str | None = None,
    no_isolated: bool = False,
    include_boundary_edges: bool = True,
    entry_points: list[str] | None = None,
) -> FcgBuildSummary:
    """End-to-end helper: APK -> raw graph -> aligned Phase-5 sidecars."""
    apk_p = Path(apk_path)
    methods_p = Path(methods_parquet_path)
    if apk_sha is None:
        apk_sha = apk_p.stem.upper()

    method_rows = read_methods_parquet(methods_p)
    graph = extract_call_graph_for_apk(
        apk_p,
        no_isolated=no_isolated,
        entry_points=entry_points,
    )
    aligned_nodes, internal_edges, boundary_edges, summary = align_call_graph_to_method_rows(
        apk_sha,
        method_rows,
        graph,
        include_boundary_edges=include_boundary_edges,
    )
    write_fcg_bundle(
        apk_sha,
        out_dir,
        aligned_nodes,
        internal_edges,
        boundary_edges,
        summary,
    )
    return summary
