"""Phase 1/2: adaptive behavior-subgraph growth with integrated boundary control."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from androserum.behavior.schema import BehaviorExpansionStep, BehaviorUnit

__all__ = [
    "grow_behavior_unit",
    "grow_representative_behavior_units",
]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _load_anchor_payload(anchor_json_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(anchor_json_path).read_text(encoding="utf-8"))


def _load_clue_payload(clue_json_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(clue_json_path).read_text(encoding="utf-8"))


def _load_embedding_npz(npz_path: str | Path) -> tuple[list[str], np.ndarray]:
    z = np.load(npz_path, allow_pickle=True)
    full_ids = [str(x) for x in z["full_id"].tolist()]
    emb = z["embedding"].astype(np.float32, copy=False)
    return full_ids, emb


def _build_full_graph(
    *,
    aligned_nodes: pd.DataFrame,
    internal_edges: pd.DataFrame,
    boundary_edges: pd.DataFrame,
) -> tuple[dict[str, set[str]], dict[str, str]]:
    adj: dict[str, set[str]] = defaultdict(set)
    node_kind: dict[str, str] = {}

    for rec in aligned_nodes.to_dict("records"):
        fid = str(rec["full_id"])
        node_kind[fid] = "internal"
        adj.setdefault(fid, set())

    for rec in internal_edges.to_dict("records"):
        s = str(rec["src_full_id"])
        d = str(rec["dst_full_id"])
        adj[s].add(d)
        adj[d].add(s)

    for rec in boundary_edges.to_dict("records"):
        internal = str(rec["internal_full_id"])
        other = str(rec["other_full_id"])
        adj[internal].add(other)
        adj[other].add(internal)
        node_kind.setdefault(other, "external")

    return adj, node_kind


def _extract_boilerplate(full_id: str) -> bool:
    tail = full_id.split("->", 1)[1] if "->" in full_id else full_id
    name = tail.split("(", 1)[0]
    if name in {"<init>", "<clinit>", "toString", "valueOf", "createFromParcel", "describeContents"}:
        return True
    if name.startswith("lambda"):
        return True
    return False


def _compute_conductance_proxy(subgraph_set: set[str], adj: dict[str, set[str]]) -> float:
    if not subgraph_set:
        return 1.0
    cut = 0
    vol = 0
    for u in subgraph_set:
        neigh = adj.get(u, set())
        vol += len(neigh)
        for v in neigh:
            if v not in subgraph_set:
                cut += 1
    if vol == 0:
        return 1.0
    return float(cut / vol)


def _compute_info_score(
    subgraph: list[str],
    clue_map: dict[str, dict[str, Any]],
) -> tuple[float, dict[str, float]]:
    n = len(subgraph)
    if n == 0:
        return 0.0, {"size": 0.0, "clue": 0.0, "boilerplate_penalty": 0.0}
    clue_sum = sum(float(clue_map.get(fid, {}).get("clue_score", 0.0)) for fid in subgraph)
    boiler_ratio = sum(1 for fid in subgraph if _extract_boilerplate(fid)) / max(1, n)
    size_term = float(np.log1p(n))
    clue_term = float(clue_sum / max(1, n))
    penalty = 0.8 * boiler_ratio
    score = size_term + clue_term - penalty
    return score, {
        "size_term": size_term,
        "clue_term": clue_term,
        "boilerplate_penalty": -penalty,
    }


def _count_internal_edges(subgraph_set: set[str], adj: dict[str, set[str]]) -> int:
    seen: set[tuple[str, str]] = set()
    count = 0
    for u in subgraph_set:
        for v in adj.get(u, set()):
            if v not in subgraph_set or u == v:
                continue
            key = tuple(sorted((u, v)))
            if key in seen:
                continue
            seen.add(key)
            count += 1
    return count


def _compute_subgraph_quality(
    *,
    subgraph: list[str],
    clue_map: dict[str, dict[str, Any]],
    adj: dict[str, set[str]],
    emb_map: dict[str, np.ndarray],
    node_kind_map: dict[str, str],
) -> tuple[float, dict[str, float]]:
    subgraph_set = set(subgraph)
    conductance = _compute_conductance_proxy(subgraph_set, adj)
    internal_edge_count = _count_internal_edges(subgraph_set, adj)
    n = len(subgraph)
    max_edges = max(1.0, n * (n - 1) / 2.0)
    edge_density = float(internal_edge_count / max_edges)

    vecs = [emb_map[fid] for fid in subgraph if fid in emb_map]
    if len(vecs) >= 2:
        center = np.mean(np.stack(vecs, axis=0), axis=0)
        semantic_cohesion = float(
            np.mean([_cosine(emb_map[fid], center) for fid in subgraph if fid in emb_map])
        )
    elif len(vecs) == 1:
        semantic_cohesion = 1.0
    else:
        semantic_cohesion = 0.0

    clue_mean = float(
        np.mean([float(clue_map.get(fid, {}).get("clue_score", 0.0)) for fid in subgraph])
    ) if subgraph else 0.0
    boilerplate_ratio = sum(1 for fid in subgraph if _extract_boilerplate(fid)) / max(1, n)
    external_ratio = (
        sum(1 for fid in subgraph if node_kind_map.get(fid, "internal") == "external") / max(1, n)
    )

    # For very small subgraphs, conductance is unstable and tends to over-penalize
    # legitimate early growth. We therefore anneal its impact by size.
    conductance_weight = 0.15 * min(1.0, n / 6.0)
    quality = (
        0.32 * semantic_cohesion
        + 0.18 * clue_mean
        + 0.28 * edge_density
        - conductance_weight * conductance
        - 0.10 * boilerplate_ratio
        - 0.05 * external_ratio
    )
    return quality, {
        "semantic_cohesion": semantic_cohesion,
        "clue_mean": clue_mean,
        "edge_density": edge_density,
        "conductance_proxy": conductance,
        "conductance_weight": conductance_weight,
        "boilerplate_ratio": -float(boilerplate_ratio),
        "external_ratio": -float(external_ratio),
    }


def _trim_low_contribution_boundary(
    *,
    subgraph: list[str],
    clue_map: dict[str, dict[str, Any]],
    adj: dict[str, set[str]],
    emb_map: dict[str, np.ndarray],
    node_kind_map: dict[str, str],
    anchor_full_id: str,
    trim_tau: float = 0.005,
) -> tuple[list[str], int]:
    """Iteratively trim low-value boundary leaves when removal improves overall quality."""
    keep = list(subgraph)
    removed = 0
    while True:
        keep_set = set(keep)
        current_quality, _ = _compute_subgraph_quality(
            subgraph=keep,
            clue_map=clue_map,
            adj=adj,
            emb_map=emb_map,
            node_kind_map=node_kind_map,
        )
        best_delta = 0.0
        best_idx = None
        for idx, fid in enumerate(keep):
            if fid == anchor_full_id:
                continue
            internal_neighbors = [n for n in adj.get(fid, set()) if n in keep_set]
            if len(internal_neighbors) > 1:
                continue
            trial = keep[:idx] + keep[idx + 1 :]
            if not trial:
                continue
            trial_quality, _ = _compute_subgraph_quality(
                subgraph=trial,
                clue_map=clue_map,
                adj=adj,
                emb_map=emb_map,
                node_kind_map=node_kind_map,
            )
            delta = trial_quality - current_quality
            if delta > best_delta:
                best_delta = delta
                best_idx = idx
        if best_idx is None or best_delta < trim_tau:
            break
        keep.pop(best_idx)
        removed += 1
    return keep, removed


def _infer_behavior_label(
    *,
    anchor_category: str | None,
    node_full_ids: list[str],
    clue_map: dict[str, dict[str, Any]],
) -> tuple[str | None, str]:
    if anchor_category and anchor_category not in {"NO_CATEGORY", "MIXED"}:
        return anchor_category, f"derived from anchor category {anchor_category}"

    counters = Counter()
    for fid in node_full_ids:
        clue = clue_map.get(fid, {})
        if clue.get("has_network_api"):
            counters["NETWORK"] += 1
        if clue.get("has_file_api"):
            counters["FILE"] += 1
        if clue.get("has_reflection_api"):
            counters["REFLECTION"] += 1
        if clue.get("has_db_api"):
            counters["DATABASE"] += 1
        if clue.get("has_location_api"):
            counters["LOCATION"] += 1
        if clue.get("has_identifier_api"):
            counters["IDENTIFIER"] += 1
        if clue.get("has_log_api"):
            counters["LOGGING"] += 1
    if counters:
        label, count = counters.most_common(1)[0]
        return label, f"inferred from dominant clue signal {label} ({count} supporting nodes)"
    return None, "no dominant category; fallback to descriptive summary only"


def grow_behavior_unit(
    *,
    apk_sha: str,
    anchor_json_path: str | Path,
    clue_json_path: str | Path,
    fcg_dir: str | Path,
    embedding_npz_path: str | Path,
    anchor_full_id: str,
    max_steps: int = 40,
    max_nodes: int = 80,
    tau_add: float = 0.01,
    tau_quality_delta: float = 0.005,
    tau_candidate_sim: float = 0.10,
    min_nodes_target: int = 6,
    trim_boundary: bool = True,
) -> BehaviorUnit:
    anchor_payload = _load_anchor_payload(anchor_json_path)
    clue_payload = _load_clue_payload(clue_json_path)
    full_ids, emb = _load_embedding_npz(embedding_npz_path)
    emb_map = {fid: emb[i] for i, fid in enumerate(full_ids)}
    clue_map = {c["full_id"]: c for c in clue_payload["result"]["clues"]}

    root = Path(fcg_dir)
    aligned = pd.read_parquet(root / f"{apk_sha}.aligned_nodes.parquet", engine="pyarrow")
    internal_edges = pd.read_parquet(root / f"{apk_sha}.internal_edges.parquet", engine="pyarrow")
    boundary_edges = pd.read_parquet(root / f"{apk_sha}.boundary_edges.parquet", engine="pyarrow")
    adj, node_kind_map = _build_full_graph(
        aligned_nodes=aligned,
        internal_edges=internal_edges,
        boundary_edges=boundary_edges,
    )

    anchors = anchor_payload["result"]["anchors"]
    anchor_meta = next(a for a in anchors if a["full_id"] == anchor_full_id)

    subgraph: list[str] = [anchor_full_id]
    subgraph_set = {anchor_full_id}
    internal_nodes = [anchor_full_id] if node_kind_map.get(anchor_full_id, "internal") == "internal" else []
    external_nodes = [anchor_full_id] if node_kind_map.get(anchor_full_id, "internal") == "external" else []
    steps: list[BehaviorExpansionStep] = []

    def current_center() -> np.ndarray | None:
        vecs = [emb_map[fid] for fid in subgraph if fid in emb_map]
        if not vecs:
            return None
        return np.mean(np.stack(vecs, axis=0), axis=0)

    frontier = set(adj.get(anchor_full_id, set()))
    current_info_score, current_info_breakdown = _compute_info_score(subgraph, clue_map)
    current_quality, current_quality_breakdown = _compute_subgraph_quality(
        subgraph=subgraph,
        clue_map=clue_map,
        adj=adj,
        emb_map=emb_map,
        node_kind_map=node_kind_map,
    )

    for step_id in range(1, max_steps + 1):
        frontier -= subgraph_set
        if not frontier:
            steps.append(
                BehaviorExpansionStep(
                    step_id=step_id,
                    selected_full_id=anchor_full_id,
                    selected_node_kind=node_kind_map.get(anchor_full_id, "internal"),
                    gain=0.0,
                    frontier_size_before=0,
                    subgraph_size_after=len(subgraph),
                    stop_reason="empty_frontier",
                )
            )
            break

        center = current_center()
        ranked: list[tuple[float, str, dict[str, float]]] = []
        for fid in frontier:
            sim = _cosine(emb_map[fid], center) if (center is not None and fid in emb_map) else 0.0
            if sim < tau_candidate_sim:
                continue
            clue = clue_map.get(fid, {})
            clue_score = float(clue.get("clue_score", 0.0))
            neighbor_links = len([n for n in adj.get(fid, set()) if n in subgraph_set])
            path_support = min(0.25, 0.08 * neighbor_links)
            boiler = 0.20 if _extract_boilerplate(fid) else 0.0
            external_penalty = 0.08 if node_kind_map.get(fid, "internal") == "external" else 0.0
            trial_nodes = subgraph + [fid]
            trial_set = set(trial_nodes)
            trial_info_score, _ = _compute_info_score(trial_nodes, clue_map)
            info_gain = trial_info_score - current_info_score
            trial_quality, trial_quality_breakdown = _compute_subgraph_quality(
                subgraph=trial_nodes,
                clue_map=clue_map,
                adj=adj,
                emb_map=emb_map,
                node_kind_map=node_kind_map,
            )
            quality_delta = trial_quality - current_quality
            trial_conductance = trial_quality_breakdown["conductance_proxy"]
            conductance_gain = current_quality_breakdown["conductance_proxy"] - trial_conductance
            gain = (
                0.60 * quality_delta
                + 0.20 * path_support
                + 0.10 * clue_score
                + 0.10 * sim
                - boiler
                - external_penalty
            )
            ranked.append(
                (
                    float(gain),
                    fid,
                    {
                        "sim": float(sim),
                        "clue_score": float(clue_score),
                        "path_support": float(path_support),
                        "info_gain": float(info_gain),
                        "quality_delta": float(quality_delta),
                        "conductance_gain": float(conductance_gain),
                        "boilerplate_penalty": -float(boiler),
                        "external_penalty": -float(external_penalty),
                    },
                )
            )

        if not ranked:
            steps.append(
                BehaviorExpansionStep(
                    step_id=step_id,
                    selected_full_id=anchor_full_id,
                    selected_node_kind=node_kind_map.get(anchor_full_id, "internal"),
                    gain=0.0,
                    frontier_size_before=len(frontier),
                    subgraph_size_after=len(subgraph),
                    stop_reason="no_candidate_passed_similarity",
                )
            )
            break

        ranked.sort(reverse=True, key=lambda x: x[0])
        best_gain, best_fid, comp = ranked[0]
        stop_reason = None
        if len(subgraph) < min_nodes_target:
            # Warm-up growth: require basic plausibility, but do not stop solely
            # on overall quality yet. This avoids collapsing to 2-node fragments.
            if comp.get("sim", 0.0) < tau_candidate_sim or comp.get("path_support", 0.0) <= 0.0:
                stop_reason = "warmup_no_plausible_candidate"
            elif len(subgraph) >= max_nodes:
                stop_reason = "max_nodes"
        elif best_gain < tau_add:
            stop_reason = "gain_below_threshold"
        elif len(subgraph) >= max_nodes:
            stop_reason = "max_nodes"
        elif comp.get("quality_delta", 0.0) < tau_quality_delta:
            stop_reason = "quality_delta_below_threshold"

        if stop_reason is not None:
            steps.append(
                BehaviorExpansionStep(
                    step_id=step_id,
                    selected_full_id=best_fid,
                    selected_node_kind=node_kind_map.get(best_fid, "internal"),
                    gain=float(best_gain),
                    score_components=comp,
                    frontier_size_before=len(frontier),
                    subgraph_size_after=len(subgraph),
                    stop_reason=stop_reason,
                )
            )
            break

        subgraph.append(best_fid)
        subgraph_set.add(best_fid)
        if node_kind_map.get(best_fid, "internal") == "internal":
            internal_nodes.append(best_fid)
        else:
            external_nodes.append(best_fid)
        frontier |= set(adj.get(best_fid, set()))
        current_info_score, current_info_breakdown = _compute_info_score(subgraph, clue_map)
        current_quality, current_quality_breakdown = _compute_subgraph_quality(
            subgraph=subgraph,
            clue_map=clue_map,
            adj=adj,
            emb_map=emb_map,
            node_kind_map=node_kind_map,
        )
        steps.append(
            BehaviorExpansionStep(
                step_id=step_id,
                selected_full_id=best_fid,
                selected_node_kind=node_kind_map.get(best_fid, "internal"),
                gain=float(best_gain),
                score_components=comp,
                frontier_size_before=len(frontier),
                subgraph_size_after=len(subgraph),
            )
        )

    pretrim_nodes = list(subgraph)
    if trim_boundary:
        subgraph, trimmed_count = _trim_low_contribution_boundary(
            subgraph=subgraph,
            clue_map=clue_map,
            adj=adj,
            emb_map=emb_map,
            node_kind_map=node_kind_map,
            anchor_full_id=anchor_full_id,
        )
        subgraph_set = set(subgraph)
        internal_nodes = [fid for fid in subgraph if node_kind_map.get(fid, "internal") == "internal"]
        external_nodes = [fid for fid in subgraph if node_kind_map.get(fid, "internal") == "external"]
        current_info_score, current_info_breakdown = _compute_info_score(subgraph, clue_map)
        current_quality, current_quality_breakdown = _compute_subgraph_quality(
            subgraph=subgraph,
            clue_map=clue_map,
            adj=adj,
            emb_map=emb_map,
            node_kind_map=node_kind_map,
        )
    else:
        trimmed_count = 0

    label, label_reason = _infer_behavior_label(
        anchor_category=anchor_meta["category"],
        node_full_ids=subgraph,
        clue_map=clue_map,
    )
    stats = {
        "n_total_nodes": len(subgraph),
        "n_internal_nodes": len(internal_nodes),
        "n_external_nodes": len(external_nodes),
        "anchor_confidence": anchor_meta["confidence"],
        "anchor_category": anchor_meta["category"],
        "info_score": current_info_score,
        "info_breakdown": current_info_breakdown,
        "quality_score": current_quality,
        "quality_breakdown": current_quality_breakdown,
        "conductance_proxy": current_quality_breakdown["conductance_proxy"],
        "boilerplate_ratio": (
            sum(1 for fid in subgraph if _extract_boilerplate(fid)) / max(1, len(subgraph))
        ),
        "min_nodes_target": min_nodes_target,
        "pretrim_n_total_nodes": len(pretrim_nodes),
        "posttrim_removed_nodes": trimmed_count,
        "behavior_label": label,
        "behavior_label_reason": label_reason,
        "terminated_by": steps[-1].stop_reason if steps else None,
    }
    return BehaviorUnit(
        apk_sha=apk_sha,
        anchor_full_id=anchor_full_id,
        anchor_kind=anchor_meta["anchor_kind"],
        anchor_category=anchor_meta["category"],
        node_full_ids=subgraph,
        internal_node_full_ids=internal_nodes,
        external_node_full_ids=external_nodes,
        steps=steps,
        stats=stats,
    )


def grow_representative_behavior_units(
    *,
    apk_sha: str,
    anchor_json_path: str | Path,
    clue_json_path: str | Path,
    fcg_dir: str | Path,
    embedding_npz_path: str | Path,
    max_steps: int = 40,
    max_nodes: int = 80,
    tau_add: float = 0.01,
    tau_quality_delta: float = 0.005,
    tau_candidate_sim: float = 0.10,
    min_nodes_target: int = 6,
    trim_boundary: bool = True,
    max_units: int = 5,
) -> list[BehaviorUnit]:
    payload = _load_anchor_payload(anchor_json_path)
    anchors = payload["result"]["anchors"]
    # Prefer one strong hard anchor per explicit category, then fill with remaining top anchors.
    picked: list[dict[str, Any]] = []
    seen_cat: set[str] = set()
    for a in anchors:
        cat = a.get("category") or "NONE"
        if cat not in seen_cat:
            picked.append(a)
            seen_cat.add(cat)
        if len(picked) >= max_units:
            break
    if len(picked) < max_units:
        seen = {a["full_id"] for a in picked}
        for a in anchors:
            if a["full_id"] in seen:
                continue
            picked.append(a)
            seen.add(a["full_id"])
            if len(picked) >= max_units:
                break

    units: list[BehaviorUnit] = []
    for a in picked:
        units.append(
            grow_behavior_unit(
                apk_sha=apk_sha,
                anchor_json_path=anchor_json_path,
                clue_json_path=clue_json_path,
                fcg_dir=fcg_dir,
                embedding_npz_path=embedding_npz_path,
                anchor_full_id=a["full_id"],
                max_steps=max_steps,
                max_nodes=max_nodes,
                tau_add=tau_add,
                tau_quality_delta=tau_quality_delta,
                tau_candidate_sim=tau_candidate_sim,
                min_nodes_target=min_nodes_target,
                trim_boundary=trim_boundary,
            )
        )
    return units
