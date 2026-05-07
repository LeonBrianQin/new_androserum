"""DexBERT encoder + projection head for Phase 4 contrastive fine-tuning."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from androserum.encoder import load_pretrained_encoder

__all__ = ["ContrastiveDexBertModel"]


class ContrastiveDexBertModel(nn.Module):
    """Wrap the pretrained transformer with a simple contrastive projection head."""

    def __init__(
        self,
        *,
        encoder: nn.Module,
        cfg,
        projection_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.cfg = cfg
        self.projection = nn.Linear(cfg.dim, projection_dim)

    @classmethod
    def from_pretrained(
        cls,
        *,
        projection_dim: int = 256,
        cfg_path: str | Path | None = None,
        weights_path: str | Path | None = None,
        vocab_path: str | Path | None = None,
        device: str | torch.device = "cpu",
    ) -> tuple["ContrastiveDexBertModel", object, object]:
        """Instantiate from the pretrained DexBERT encoder checkpoint."""
        encoder, tokenizer, cfg = load_pretrained_encoder(
            cfg_path=cfg_path,
            weights_path=weights_path,
            vocab_path=vocab_path,
            device=device,
        )
        model = cls(encoder=encoder, cfg=cfg, projection_dim=projection_dim)
        model.to(device)
        return model, tokenizer, cfg

    def freeze_bottom_layers(
        self,
        n_layers: int,
        *,
        freeze_embeddings: bool = True,
    ) -> None:
        """Freeze the lower part of the encoder, keeping upper blocks trainable."""
        if freeze_embeddings:
            for param in self.encoder.embed.parameters():
                param.requires_grad = False

        n_layers = max(0, min(n_layers, len(self.encoder.blocks)))
        for i, block in enumerate(self.encoder.blocks):
            requires_grad = i >= n_layers
            for param in block.parameters():
                param.requires_grad = requires_grad

    def encode_cls(
        self,
        input_ids: torch.Tensor,
        seg_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the CLS representation used downstream for clustering."""
        hidden = self.encoder(input_ids, seg_ids, mask)
        return hidden[:, 0, :]

    def project_cls(self, cls_vecs: torch.Tensor) -> torch.Tensor:
        """Projection head used only for the contrastive objective."""
        return F.normalize(self.projection(cls_vecs), dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        seg_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cls_vecs = self.encode_cls(input_ids, seg_ids, mask)
        return cls_vecs, self.project_cls(cls_vecs)

    def trainable_parameter_count(self) -> int:
        """Count parameters with ``requires_grad=True`` after freezing."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
