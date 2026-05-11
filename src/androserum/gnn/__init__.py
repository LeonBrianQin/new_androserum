"""Phase 6: self-supervised GraphSAGE+BGRL over aligned FCG graphs."""

from .dataset import FcgGraphDataset, FcgGraphSample, external_family_key
from .trainer import (
    GnnTrainConfig,
    export_gnn_embeddings,
    load_gnn_config,
    train_bgrl_graphsage,
)

__all__ = [
    "FcgGraphDataset",
    "FcgGraphSample",
    "GnnTrainConfig",
    "export_gnn_embeddings",
    "external_family_key",
    "load_gnn_config",
    "train_bgrl_graphsage",
]
