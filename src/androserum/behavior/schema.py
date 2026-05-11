"""Schemas for behavior-subgraph discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AnchorCandidate:
    apk_sha: str
    full_id: str
    class_name: str
    method_sig: str
    node_kind: str  # internal / external
    is_external: bool
    seed_type: str  # sapi / resource / callback / other
    category: str | None
    confidence: float
    degree: int
    anchor_kind: str = "hard"  # hard / context / rejected
    source: str = ""  # internal_susi / external_susi / external_context
    exact_match: bool = False
    constraint_flags: dict[str, Any] = field(default_factory=dict)
    score_components: dict[str, float] = field(default_factory=dict)
    future_eval: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    api_calls: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnchorDiscoveryResult:
    apk_sha: str
    anchors: list[AnchorCandidate]
    context_candidates: list[AnchorCandidate]
    stats: dict[str, Any]
