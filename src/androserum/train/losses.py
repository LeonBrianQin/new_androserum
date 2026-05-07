"""A+B contrastive losses for Phase 4."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import torch
import torch.nn.functional as F

__all__ = [
    "ab_contrastive_loss",
    "build_ab_positive_mask",
    "count_b_positive_pairs",
    "multi_positive_info_nce_loss",
]


def build_ab_positive_mask(
    susi_labels: Sequence[str | None],
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build the 2N×2N positive mask for A+B.

    A:
        same method, different dropout view

    B:
        same non-empty SuSi dominant category across different methods
    """
    n = len(susi_labels)
    mask = torch.zeros((2 * n, 2 * n), dtype=torch.bool, device=device)

    # A: each sample's two dropout views are always positives.
    for i in range(n):
        mask[i, i + n] = True
        mask[i + n, i] = True

    # B: same SuSi label across different methods.
    groups: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(susi_labels):
        if label is not None:
            groups[label].append(i)

    for indices in groups.values():
        if len(indices) < 2:
            continue
        for i in indices:
            for j in indices:
                if i == j:
                    continue
                mask[i, j] = True
                mask[i, j + n] = True
                mask[i + n, j] = True
                mask[i + n, j + n] = True

    return mask


def count_b_positive_pairs(susi_labels: Sequence[str | None]) -> int:
    """Count cross-method positive pairs contributed by signal B."""
    groups: dict[str, int] = defaultdict(int)
    for label in susi_labels:
        if label is not None:
            groups[label] += 1
    total = 0
    for count in groups.values():
        if count >= 2:
            total += count * (count - 1) // 2
    return total


def multi_positive_info_nce_loss(
    embeddings: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Multi-positive InfoNCE / SupCon-style loss with an explicit positive mask."""
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape={tuple(embeddings.shape)}")
    if positive_mask.shape != (embeddings.size(0), embeddings.size(0)):
        raise ValueError(
            "positive_mask shape mismatch: "
            f"got {tuple(positive_mask.shape)} for embeddings {tuple(embeddings.shape)}"
        )

    logits = embeddings @ embeddings.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    eye = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
    valid_mask = ~eye
    positive_mask = positive_mask.to(device=logits.device, dtype=torch.bool) & valid_mask

    exp_logits = torch.exp(logits) * valid_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_counts = positive_mask.sum(dim=1).clamp_min(1)
    loss = -(log_prob * positive_mask.float()).sum(dim=1) / positive_counts.float()
    return loss.mean()


def ab_contrastive_loss(
    view1: torch.Tensor,
    view2: torch.Tensor,
    susi_labels: Sequence[str | None],
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Phase 4 A+B loss over two encoder dropout views."""
    if view1.shape != view2.shape:
        raise ValueError(
            f"view shapes must match, got {tuple(view1.shape)} vs {tuple(view2.shape)}"
        )
    z = torch.cat([F.normalize(view1, dim=-1), F.normalize(view2, dim=-1)], dim=0)
    positive_mask = build_ab_positive_mask(susi_labels, device=z.device)
    return multi_positive_info_nce_loss(z, positive_mask, temperature=temperature)
