"""Phase 0: discover SAPI / key API anchor candidates in the full FCG."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from androserum.data.susi_index import build_susi_index
from androserum.behavior.schema import AnchorCandidate, AnchorDiscoveryResult

__all__ = [
    "discover_anchor_candidates",
]


def _normalize_sha_token(s: str) -> str | None:
    s = s.strip().upper()
    if len(s) != 64:
        return None
    if any(c not in "0123456789ABCDEF" for c in s):
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


def _read_full_fcg_views(
    fcg_dir: str | Path,
    sha: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    root = Path(fcg_dir)
    aligned = pd.read_parquet(root / f"{sha}.aligned_nodes.parquet", engine="pyarrow")
    internal_edges = pd.read_parquet(root / f"{sha}.internal_edges.parquet", engine="pyarrow")
    boundary_edges = pd.read_parquet(root / f"{sha}.boundary_edges.parquet", engine="pyarrow")
    summary = {}
    summary_path = root / f"{sha}.summary.json"
    if summary_path.is_file():
        import json

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return aligned, internal_edges, boundary_edges, summary


def _load_external_node_summaries(
    boundary_edges: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for rec in boundary_edges.to_dict("records"):
        other = str(rec["other_full_id"])
        bucket = out.setdefault(other, {"count": 0, "directions": Counter(), "examples": []})
        bucket["count"] += 1
        bucket["directions"][str(rec["direction"])] += 1
        bucket.setdefault("internal_ids", set()).add(str(rec["internal_full_id"]))
        if len(bucket["examples"]) < 3:
            bucket["examples"].append(
                {
                    "internal_full_id": rec["internal_full_id"],
                    "direction": rec["direction"],
                    "other_external": bool(rec["other_external"]),
                }
            )
    return out


def _resolve_seed_type(category: str | None) -> str:
    if not category:
        return "other"
    if category in {"NETWORK", "NETWORK_INFORMATION"}:
        return "network"
    if category in {"FILE"}:
        return "file"
    if category in {"SMS", "PHONE_CONNECTION", "CONTACTS", "LOCATION", "CAMERA", "AUDIO"}:
        return "sensitive_api"
    return "sapi"


def _base_confidence(category: str | None, degree: int, is_external: bool) -> float:
    base = 0.55
    if category in {"NETWORK", "FILE"}:
        base += 0.12
    elif category:
        base += 0.08
    base += min(0.18, 0.02 * degree)
    if is_external:
        base -= 0.12
    return float(max(0.05, min(base, 0.95)))


def _is_benign_boilerplate(full_id: str, api_calls: list[str]) -> bool:
    tail = full_id.split("->", 1)[1] if "->" in full_id else full_id
    if tail.startswith("<init>") or tail.startswith("<clinit>"):
        return True
    boilerplate_names = {
        "toString",
        "valueOf",
        "hashCode",
        "equals",
        "clone",
        "writeToParcel",
        "describeContents",
        "createFromParcel",
    }
    if tail.split("(", 1)[0] in boilerplate_names:
        return True
    if any("StringBuilder->" in api or "Object-><init>" in api for api in api_calls):
        return True
    return False


def discover_anchor_candidates(
    *,
    apk_sha: str,
    methods_dir: str | Path,
    fcg_dir: str | Path,
    susi_sources: str | Path,
    susi_sinks: str | Path,
    top_k_per_category: int = 0,
    min_degree: int = 1,
) -> AnchorDiscoveryResult:
    """Discover Phase 0 anchor candidates from the whole FCG.

    This is intentionally high-recall:
    - internal methods can be anchors
    - external relay nodes can also be anchors
    - SAPI/category hits are the primary signal
    """
    sha = _normalize_sha_token(apk_sha)
    if sha is None:
        raise ValueError(f"invalid apk_sha: {apk_sha!r}")

    methods_df = pd.read_parquet(Path(methods_dir) / f"{sha}.parquet", engine="pyarrow")
    aligned, internal_edges, boundary_edges, summary = _read_full_fcg_views(fcg_dir, sha)

    # Build an API -> SuSi category index once per call; this is the primary seed source.
    index = build_susi_index([susi_sources, susi_sinks])

    aligned_index = aligned.set_index("full_id", drop=False)
    degree_cols = aligned_index[
        [
            "internal_in_degree",
            "internal_out_degree",
            "external_in_degree",
            "external_out_degree",
        ]
    ].fillna(0)
    degree_map = degree_cols.sum(axis=1).to_dict()
    row_map = aligned_index.to_dict("index")
    external_summaries = _load_external_node_summaries(boundary_edges)

    anchors: list[AnchorCandidate] = []
    context_candidates: list[AnchorCandidate] = []
    anchor_seen: set[tuple[str, str]] = set()

    # Internal methods first.
    for rec in methods_df.to_dict("records"):
        full_id = str(rec["full_id"])
        api_calls = _flatten_listish(rec.get("api_calls", []))
        cats: list[str] = []
        exact_matches: list[str] = []
        for api in api_calls:
            hit = sorted(index.categories_for_api(api))
            if hit:
                exact_matches.append(api)
                cats.extend(hit)
        cats = sorted(set(cats))
        if not cats:
            continue
        degree = int(degree_map.get(full_id, 0))
        if degree < min_degree:
            continue
        boilerplate = _is_benign_boilerplate(full_id, api_calls)
        seed_type = _resolve_seed_type(cats[0] if len(cats) == 1 else "MIXED")
        confidence = _base_confidence(cats[0] if len(cats) == 1 else "MIXED", degree, False)
        confidence = confidence - (0.15 if boilerplate else 0.0)
        key = (full_id, "internal")
        if key in anchor_seen:
            continue
        anchor_seen.add(key)
        hard_anchor = bool(exact_matches) and not boilerplate
        source = "internal_susi" if exact_matches else "internal_context"
        evidence = {
            "api_calls": api_calls,
            "susi_cats": cats,
            "filtered": bool(rec.get("filtered", False)),
            "graph_present": bool(rec.get("graph_present", True)),
            "exact_matches": exact_matches,
        }
        candidate = AnchorCandidate(
            apk_sha=sha,
            full_id=full_id,
            class_name=str(rec["class_name"]),
            method_sig=str(rec["method_sig"]),
            node_kind="internal",
            is_external=False,
            seed_type=seed_type,
            category=cats[0] if len(cats) == 1 else "MIXED",
            confidence=float(max(0.05, min(confidence, 0.99))),
            degree=degree,
            anchor_kind="hard" if hard_anchor else "context",
            source=source,
            exact_match=bool(exact_matches),
            constraint_flags={
                "boilerplate": boilerplate,
                "min_degree_ok": degree >= min_degree,
                "exact_susi_match": bool(exact_matches),
                "graph_present": bool(rec.get("graph_present", True)),
            },
            score_components={
                "base": _base_confidence(cats[0] if len(cats) == 1 else "MIXED", degree, False),
                "degree": float(min(0.18, 0.02 * degree)),
                "category_bias": 0.12 if cats[0] in {"NETWORK", "FILE"} else 0.08 if cats else 0.0,
                "boilerplate_penalty": -0.15 if boilerplate else 0.0,
            },
            future_eval={
                "coverage_check": True,
                "purity_check": True,
                "info_gain_check": False,
            },
            notes="seed candidate derived from exact SAPI lookup" if hard_anchor else "context candidate adjacent to SAPI evidence",
            api_calls=api_calls,
            evidence=evidence,
        )
        if hard_anchor:
            anchors.append(candidate)
        else:
            context_candidates.append(candidate)

    # External relay nodes: treat them as possible anchors if they appear repeatedly
    # across boundary edges and are informative enough.
    for other_full_id, info in external_summaries.items():
        if info["count"] < min_degree:
            continue
        connected_internal_ids = sorted(info.get("internal_ids", set()))
        connected_cats: list[str] = []
        exact_match = False
        for fid in connected_internal_ids:
            if fid not in aligned_index.index:
                continue
            row = row_map[fid]
            for api in _flatten_listish(row.get("api_calls", [])):
                hit = sorted(index.categories_for_api(api))
                if hit:
                    exact_match = True
                    connected_cats.extend(hit)
        connected_cats = sorted(set(connected_cats))
        degree = int(info["count"])
        boilerplate = _is_benign_boilerplate(other_full_id, [])
        confidence = _base_confidence(
            connected_cats[0] if len(connected_cats) == 1 else ("MIXED" if connected_cats else None),
            degree,
            True,
        )
        confidence = confidence - (0.12 if boilerplate else 0.0)
        key = (other_full_id, "external")
        if key in anchor_seen:
            continue
        anchor_seen.add(key)
        candidate = AnchorCandidate(
            apk_sha=sha,
            full_id=other_full_id,
            class_name=other_full_id.split("->", 1)[0],
            method_sig=other_full_id.split("->", 1)[1] if "->" in other_full_id else other_full_id,
            node_kind="external",
            is_external=True,
            seed_type="sapi" if exact_match else "other",
            category=connected_cats[0] if len(connected_cats) == 1 else ("MIXED" if connected_cats else None),
            confidence=float(max(0.05, min(confidence, 0.95))),
            degree=degree,
            anchor_kind="hard" if exact_match and not boilerplate else "context",
            source="external_susi" if exact_match else "external_context",
            exact_match=exact_match,
            constraint_flags={
                "boilerplate": boilerplate,
                "min_degree_ok": degree >= min_degree,
                "exact_susi_match": exact_match,
                "boundary_supported": True,
            },
            score_components={
                "base": _base_confidence(
                    connected_cats[0] if len(connected_cats) == 1 else ("MIXED" if connected_cats else None),
                    degree,
                    True,
                ),
                "degree": float(min(0.18, 0.02 * degree)),
                "boundary_support": float(min(0.15, 0.01 * len(connected_internal_ids))),
                "boilerplate_penalty": -0.12 if boilerplate else 0.0,
            },
            future_eval={
                "coverage_check": True,
                "purity_check": True,
                "info_gain_check": False,
            },
            notes="external anchor candidate derived from boundary context" if not exact_match else "external hard anchor from exact SAPI lookup",
            api_calls=[],
            evidence={
                "direction_counts": dict(info["directions"]),
                "examples": info["examples"],
                "connected_internal_count": len(set(connected_internal_ids)),
                "connected_categories": connected_cats,
            },
        )
        if candidate.anchor_kind == "hard":
            anchors.append(candidate)
        else:
            context_candidates.append(candidate)

    anchors.sort(key=lambda x: (x.confidence, x.degree), reverse=True)
    context_candidates.sort(key=lambda x: (x.confidence, x.degree), reverse=True)

    if top_k_per_category and top_k_per_category > 0:
        seen: dict[str, int] = {}
        pruned: list[AnchorCandidate] = []
        for a in anchors:
            key = a.category or "UNCATEGORIZED"
            if seen.get(key, 0) >= top_k_per_category:
                continue
            seen[key] = seen.get(key, 0) + 1
            pruned.append(a)
        anchors = pruned

    stats = {
        "methods_rows": len(methods_df),
        "graph_internal_nodes_total": int(summary.get("graph_internal_nodes_total", len(aligned))),
        "graph_external_nodes_total": int(summary.get("graph_external_nodes_total", len(external_summaries))),
        "anchor_candidates": len(anchors),
        "context_candidates": len(context_candidates),
        "internal_anchor_candidates": sum(1 for a in anchors if not a.is_external),
        "external_anchor_candidates": sum(1 for a in anchors if a.is_external),
    }
    return AnchorDiscoveryResult(apk_sha=sha, anchors=anchors, context_candidates=context_candidates, stats=stats)
