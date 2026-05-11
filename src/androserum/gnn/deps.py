"""Dependency helpers for Phase 6 optional PyG components."""

from __future__ import annotations


def require_torch_geometric():
    """Import the PyG pieces used by Phase 6 or raise a helpful error."""
    try:
        from torch_geometric.data import Data
        from torch_geometric.nn import SAGEConv
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError(
            "Phase 6 requires torch-geometric and its compiled extensions. "
            "Run `bash scripts/setup_gnn_env.sh` in the active env first."
        ) from exc
    return Data, SAGEConv
