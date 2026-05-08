"""Batch samplers for Phase 4 contrastive fine-tuning."""

from __future__ import annotations

import math
import random

from torch.utils.data import Sampler

__all__ = ["PositiveGroupBatchSampler", "SusiGroupBatchSampler"]


def _sample_without_conflicts(
    pool: list[int],
    k: int,
    rng: random.Random,
    used: set[int],
) -> list[int]:
    """Prefer distinct indices inside a batch, then fall back to reuse if needed."""
    if not pool or k <= 0:
        return []

    available = [idx for idx in pool if idx not in used]
    if len(available) >= k:
        picks = rng.sample(available, k=k)
        used.update(picks)
        return picks

    picks = list(available)
    used.update(available)
    while len(picks) < k:
        picks.append(rng.choice(pool))
    return picks


class SusiGroupBatchSampler(Sampler[list[int]]):
    """Mixed batch sampler: some grouped SuSi positives, some global random samples."""

    def __init__(
        self,
        *,
        all_indices: list[int],
        label_to_indices: dict[str, list[int]],
        batch_size: int,
        label_group_size: int = 2,
        label_fraction: float = 0.5,
        steps_per_epoch: int | None = None,
        seed: int = 13,
    ) -> None:
        if batch_size < 2:
            raise ValueError("batch_size must be at least 2 for contrastive learning")
        if label_group_size < 2:
            raise ValueError("label_group_size must be at least 2")
        if not 0.0 <= label_fraction <= 1.0:
            raise ValueError("label_fraction must be in [0, 1]")
        if not all_indices:
            raise ValueError("all_indices must not be empty")

        self.all_indices = list(all_indices)
        self.label_to_indices = {
            label: list(indices)
            for label, indices in label_to_indices.items()
            if len(indices) >= 2
        }
        self.label_names = sorted(self.label_to_indices)
        self.batch_size = batch_size
        self.label_group_size = label_group_size
        self.label_fraction = label_fraction
        self.seed = seed
        self.epoch = 0
        self.steps_per_epoch = steps_per_epoch or max(
            1, math.ceil(len(self.all_indices) / self.batch_size)
        )

    def __len__(self) -> int:
        return self.steps_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _label_slots(self) -> int:
        if not self.label_names or self.label_fraction <= 0.0:
            return 0
        slots = int(self.batch_size * self.label_fraction)
        slots = max(self.label_group_size, slots)
        slots = min(self.batch_size, slots)
        slots = (slots // self.label_group_size) * self.label_group_size
        if slots == 0:
            slots = min(self.batch_size, self.label_group_size)
        return slots

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        label_slots = self._label_slots()
        n_groups = label_slots // self.label_group_size

        for _ in range(self.steps_per_epoch):
            batch: list[int] = []
            used: set[int] = set()

            if self.label_names and n_groups > 0:
                if n_groups <= len(self.label_names):
                    chosen_labels = rng.sample(self.label_names, k=n_groups)
                else:
                    chosen_labels = [rng.choice(self.label_names) for _ in range(n_groups)]
                for label in chosen_labels:
                    batch.extend(
                        _sample_without_conflicts(
                            self.label_to_indices[label],
                            self.label_group_size,
                            rng,
                            used,
                        )
                    )

            remaining = self.batch_size - len(batch)
            if remaining > 0:
                batch.extend(
                    _sample_without_conflicts(self.all_indices, remaining, rng, used)
                )

            rng.shuffle(batch)
            yield batch[: self.batch_size]


class PositiveGroupBatchSampler(Sampler[list[int]]):
    """Batch sampler that can mix multiple positive-group sources.

    Example sources:
      * SuSi label groups (signal B)
      * override-key groups (signal E)
    """

    def __init__(
        self,
        *,
        all_indices: list[int],
        group_maps: list[dict[str, list[int]]],
        batch_size: int,
        group_size: int = 2,
        grouped_fraction: float = 0.5,
        steps_per_epoch: int | None = None,
        seed: int = 13,
    ) -> None:
        if batch_size < 2:
            raise ValueError("batch_size must be at least 2")
        if group_size < 2:
            raise ValueError("group_size must be at least 2")
        if not all_indices:
            raise ValueError("all_indices must not be empty")

        self.all_indices = list(all_indices)
        self.group_maps = [
            {k: list(v) for k, v in gm.items() if len(v) >= 2}
            for gm in group_maps
            if gm
        ]
        self.batch_size = batch_size
        self.group_size = group_size
        self.grouped_fraction = grouped_fraction
        self.seed = seed
        self.epoch = 0
        self.steps_per_epoch = steps_per_epoch or max(1, math.ceil(len(all_indices) / batch_size))

    def __len__(self) -> int:
        return self.steps_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        grouped_slots = int(self.batch_size * self.grouped_fraction)
        grouped_slots = min(self.batch_size, max(0, grouped_slots))
        grouped_slots = (grouped_slots // self.group_size) * self.group_size
        n_groups = grouped_slots // self.group_size

        for _ in range(self.steps_per_epoch):
            batch: list[int] = []
            used: set[int] = set()

            for _group_idx in range(n_groups):
                if not self.group_maps:
                    break
                group_map = rng.choice(self.group_maps)
                if not group_map:
                    continue
                label = rng.choice(sorted(group_map))
                batch.extend(
                    _sample_without_conflicts(
                        group_map[label],
                        self.group_size,
                        rng,
                        used,
                    )
                )

            remaining = self.batch_size - len(batch)
            if remaining > 0:
                batch.extend(_sample_without_conflicts(self.all_indices, remaining, rng, used))

            rng.shuffle(batch)
            yield batch[: self.batch_size]
