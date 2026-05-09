"""Phase 5: function call graph extraction (Androguard) + method-id alignment."""

from .extract import (
    FcgAlignedNodeRecord,
    FcgBoundaryEdgeRecord,
    FcgBuildSummary,
    FcgInternalEdgeRecord,
    align_call_graph_to_method_rows,
    extract_call_graph_for_apk,
    extract_fcg_bundle_for_apk,
    method_to_full_id,
    write_fcg_bundle,
)

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
