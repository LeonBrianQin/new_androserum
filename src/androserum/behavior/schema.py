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


@dataclass
class MethodClue:
    apk_sha: str
    full_id: str
    class_name: str
    method_sig: str
    susi_cats: list[str] = field(default_factory=list)
    susi_dominant_cat: str | None = None
    has_network_api: bool = False
    has_file_api: bool = False
    has_reflection_api: bool = False
    has_db_api: bool = False
    has_location_api: bool = False
    has_identifier_api: bool = False
    has_log_api: bool = False
    has_stringbuilder_pattern: bool = False
    string_literals: list[str] = field(default_factory=list)
    url_like_strings: list[str] = field(default_factory=list)
    file_like_strings: list[str] = field(default_factory=list)
    clue_score: float = 0.0
    clue_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class MethodClueResult:
    apk_sha: str
    clues: list[MethodClue]
    stats: dict[str, Any]


@dataclass
class BehaviorExpansionStep:
    step_id: int
    selected_full_id: str
    selected_node_kind: str
    gain: float
    score_components: dict[str, float] = field(default_factory=dict)
    frontier_size_before: int = 0
    subgraph_size_after: int = 0
    stop_reason: str | None = None


@dataclass
class BehaviorUnit:
    apk_sha: str
    anchor_full_id: str
    anchor_kind: str
    anchor_category: str | None
    node_full_ids: list[str] = field(default_factory=list)
    internal_node_full_ids: list[str] = field(default_factory=list)
    external_node_full_ids: list[str] = field(default_factory=list)
    steps: list[BehaviorExpansionStep] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
