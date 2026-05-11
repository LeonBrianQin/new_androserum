"""Minimal BGRL wrapper for Phase 6."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F

from androserum.gnn.models import GraphSageEncoder, MlpPredictor

__all__ = [
    "BgrlModel",
    "bgrl_regression_loss",
]


def bgrl_regression_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cosine regression loss used by BGRL."""
    pred = F.normalize(pred, dim=-1, p=2)
    target = F.normalize(target, dim=-1, p=2)
    return 2.0 - 2.0 * (pred * target).sum(dim=-1)


class BgrlModel(nn.Module):
    """Online/target encoder pair with EMA updates."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        predictor_hidden_dim: int,
        dropout: float = 0.0,
        external_prior_mode: str = "none",
        external_family_vocab: int = 0,
        ema_decay: float = 0.99,
    ) -> None:
        super().__init__()
        self.online_encoder = GraphSageEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            external_prior_mode=external_prior_mode,
            external_family_vocab=external_family_vocab,
        )
        self.target_encoder = deepcopy(self.online_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.predictor = MlpPredictor(output_dim, predictor_hidden_dim)
        self.ema_decay = float(ema_decay)

    @torch.no_grad()
    def update_target_network(self) -> None:
        for online, target in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            target.data.mul_(self.ema_decay).add_(online.data, alpha=1.0 - self.ema_decay)

    def encode_online(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_kind: torch.Tensor,
        family_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.online_encoder(x, edge_index, node_kind, family_ids)

    @torch.no_grad()
    def encode_target(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_kind: torch.Tensor,
        family_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.target_encoder(x, edge_index, node_kind, family_ids)

    def predict(self, h: torch.Tensor) -> torch.Tensor:
        return self.predictor(h)
