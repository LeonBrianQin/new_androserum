"""GraphSAGE encoder and feature adapters for Phase 6."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from androserum.gnn.dataset import (
    INTERNAL_NODE_KIND,
)
from androserum.gnn.deps import require_torch_geometric

__all__ = [
    "GraphSageEncoder",
    "MlpPredictor",
]


class GraphSageEncoder(nn.Module):
    """2-layer GraphSAGE encoder with optional relay-node priors."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        external_prior_mode: str = "none",
        external_family_vocab: int = 0,
        num_node_kinds: int = 3,
    ) -> None:
        super().__init__()
        _, SAGEConv = require_torch_geometric()
        if external_prior_mode not in {"none", "global", "package"}:
            raise ValueError(f"unsupported external_prior_mode: {external_prior_mode}")

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.node_kind_embed = nn.Embedding(num_node_kinds, hidden_dim)
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = float(dropout)
        self.external_prior_mode = external_prior_mode

        if external_prior_mode == "global":
            self.external_seed = nn.Parameter(torch.zeros(hidden_dim))
            nn.init.normal_(self.external_seed, std=0.02)
            self.external_family_embed = None
        elif external_prior_mode == "package":
            if external_family_vocab <= 0:
                raise ValueError("package external prior requires external_family_vocab > 0")
            self.external_seed = None
            self.external_family_embed = nn.Embedding(external_family_vocab, hidden_dim)
        else:
            self.external_seed = None
            self.external_family_embed = None

    def _encode_inputs(
        self,
        x: torch.Tensor,
        node_kind: torch.Tensor,
        family_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        h = self.input_proj(x) + self.node_kind_embed(node_kind)
        relay_mask = node_kind != INTERNAL_NODE_KIND
        if self.external_prior_mode == "global" and self.external_seed is not None:
            h = h.clone()
            h[relay_mask] = h[relay_mask] + self.external_seed
        elif self.external_prior_mode == "package" and self.external_family_embed is not None:
            if family_ids is None:
                raise ValueError("package external prior requires family_ids")
            valid = family_ids >= 0
            if bool(valid.any()):
                h = h.clone()
                h[valid] = h[valid] + self.external_family_embed(family_ids[valid])
        return h

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_kind: torch.Tensor,
        family_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self._encode_inputs(x, node_kind, family_ids)
        h = self.conv1(h, edge_index)
        h = self.norm1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return h


class MlpPredictor(nn.Module):
    """Small predictor head used by BGRL."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
