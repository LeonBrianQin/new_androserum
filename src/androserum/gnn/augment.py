"""Graph augmentations for Phase 6 BGRL."""

from __future__ import annotations

import torch

__all__ = [
    "drop_edge",
    "mask_node_features",
]


def drop_edge(
    edge_index: torch.Tensor,
    *,
    drop_prob: float,
    training: bool,
) -> torch.Tensor:
    """Randomly drop edges from a ``[2, E]`` edge index."""
    if not training or drop_prob <= 0.0 or edge_index.numel() == 0:
        return edge_index
    keep = torch.rand(edge_index.shape[1], device=edge_index.device) >= drop_prob
    if bool(keep.any()):
        return edge_index[:, keep]
    return edge_index[:, :0]


def mask_node_features(
    x: torch.Tensor,
    *,
    mask_prob: float,
    training: bool,
) -> torch.Tensor:
    """Element-wise feature masking used by BGRL view generation."""
    if not training or mask_prob <= 0.0 or x.numel() == 0:
        return x
    keep = torch.rand_like(x) >= mask_prob
    return x * keep.to(dtype=x.dtype)
