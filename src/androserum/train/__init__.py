"""Phase 4: contrastive fine-tuning of the encoder (SimCSE + SuSi same-category)."""

from .contrastive_model import ContrastiveDexBertModel
from .dataset import ContrastiveMethodDataset
from .losses import ab_contrastive_loss, abce_contrastive_loss, abe_contrastive_loss
from .samplers import PositiveGroupBatchSampler, SusiGroupBatchSampler
from .trainer import ContrastiveTrainConfig, export_finetuned_embeddings, train_contrastive_ab

__all__ = [
    "ContrastiveDexBertModel",
    "ContrastiveMethodDataset",
    "ContrastiveTrainConfig",
    "PositiveGroupBatchSampler",
    "SusiGroupBatchSampler",
    "ab_contrastive_loss",
    "abce_contrastive_loss",
    "abe_contrastive_loss",
    "export_finetuned_embeddings",
    "train_contrastive_ab",
]
