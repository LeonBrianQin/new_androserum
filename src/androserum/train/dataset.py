"""Phase 4 dataset utilities for A+B contrastive fine-tuning.

Design notes
------------
We do *not* keep every method from every APK in memory. On a 200-APK dev set
that can easily turn into gigabytes of joined instruction text.

Instead we build a training pool with:

* all usable SuSi-labelled methods (signal B)
* a bounded random sample of unlabeled methods (signal A regularization)

That keeps the MVP runnable on a single workstation while preserving the core
contrastive signal design from ``PROJECT_HANDOFF.md``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import math
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

__all__ = [
    "ContrastiveBatch",
    "ContrastiveMethodDataset",
    "MethodTextSample",
    "build_contrastive_collate_fn",
    "load_sha_list",
    "normalize_susi_label",
    "texts_to_model_inputs",
]


def _normalize_sha_token(s: str) -> str | None:
    s = s.strip().upper()
    if len(s) != 64:
        return None
    if any(c not in "0123456789ABCDEF" for c in s):
        return None
    return s


def load_sha_list(sha_file: str | None, methods_dir: Path) -> list[str]:
    """Load SHAs from a file or infer them from ``methods_dir/*.parquet``."""
    if sha_file:
        p = Path(sha_file)
        if not p.is_file():
            raise FileNotFoundError(f"sha_file not found: {p}")
        seen: set[str] = set()
        out: list[str] = []
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sha = _normalize_sha_token(line.split(",", 1)[0])
            if sha is None or sha in seen:
                continue
            seen.add(sha)
            out.append(sha)
        return out

    shas: list[str] = []
    for p in sorted(methods_dir.glob("*.parquet")):
        sha = _normalize_sha_token(p.stem)
        if sha is not None:
            shas.append(sha)
    return sorted(set(shas))


def normalize_susi_label(raw: object) -> str | None:
    """Map missing / unusable SuSi values to ``None`` for Phase 4."""
    if raw is None:
        return None
    if isinstance(raw, float) and math.isnan(raw):
        return None
    s = str(raw).strip()
    if not s:
        return None
    if s.upper() == "NO_CATEGORY":
        return None
    return s


@dataclass(frozen=True)
class MethodTextSample:
    """One training sample kept in the Phase 4 in-memory pool."""

    apk_sha: str
    full_id: str
    text: str
    susi_label: str | None
    override_keys: list[str] = field(default_factory=list)
    library_keys: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ContrastiveBatch:
    """Model-ready tensors plus the metadata needed for A+B loss."""

    input_ids: torch.Tensor
    seg_ids: torch.Tensor
    mask: torch.Tensor
    susi_labels: list[str | None]
    override_keys: list[list[str]]
    library_keys: list[list[str]]
    apk_shas: list[str]
    full_ids: list[str]


def _pad_batch(
    ids_rows: list[list[int]],
    max_len: int,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build ``input_ids``, ``seg`` and ``mask`` with the Phase 3 convention."""
    b = len(ids_rows)
    inp = torch.full((b, max_len), pad_id, dtype=torch.long)
    seg = torch.zeros((b, max_len), dtype=torch.long)
    mask = torch.zeros((b, max_len), dtype=torch.long)
    for i, row in enumerate(ids_rows):
        n = min(len(row), max_len)
        inp[i, :n] = torch.tensor(row[:n], dtype=torch.long)
        mask[i, :n] = 1
    return inp, seg, mask


def texts_to_model_inputs(
    texts: Sequence[str],
    tokenizer,
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize raw instruction blobs exactly like Phase 3."""
    cls_id = tokenizer.vocab["[CLS]"]
    sep_id = tokenizer.vocab["[SEP]"]
    pad_id = tokenizer.vocab["[PAD]"]
    cap = max_len - 2
    rows: list[list[int]] = []
    for text in texts:
        uni = tokenizer.convert_to_unicode(text or "")
        toks = tokenizer.tokenize(uni)
        if len(toks) > cap:
            toks = toks[:cap]
        rows.append([cls_id] + tokenizer.convert_tokens_to_ids(toks) + [sep_id])
    return _pad_batch(rows, max_len, pad_id)


def build_contrastive_collate_fn(tokenizer, cfg) -> Callable[[list[MethodTextSample]], ContrastiveBatch]:
    """Create a DataLoader collate function bound to the DexBERT tokenizer/config."""

    def _collate(batch: list[MethodTextSample]) -> ContrastiveBatch:
        texts = [row.text for row in batch]
        input_ids, seg_ids, mask = texts_to_model_inputs(texts, tokenizer, cfg.max_len)
        return ContrastiveBatch(
            input_ids=input_ids,
            seg_ids=seg_ids,
            mask=mask,
            susi_labels=[row.susi_label for row in batch],
            override_keys=[row.override_keys for row in batch],
            library_keys=[row.library_keys for row in batch],
            apk_shas=[row.apk_sha for row in batch],
            full_ids=[row.full_id for row in batch],
        )

    return _collate


class ContrastiveMethodDataset(Dataset[MethodTextSample]):
    """Sampled in-memory training pool for Phase 4 A+B fine-tuning."""

    def __init__(
        self,
        samples: list[MethodTextSample],
        source_shas: list[str],
        *,
        skipped_filtered: int = 0,
        skipped_unlabeled: int = 0,
    ) -> None:
        self.samples = samples
        self.source_shas = source_shas
        self.skipped_filtered = skipped_filtered
        self.skipped_unlabeled = skipped_unlabeled
        self.label_to_indices: dict[str, list[int]] = defaultdict(list)
        self.unlabeled_indices: list[int] = []
        for idx, sample in enumerate(self.samples):
            if sample.susi_label is None:
                self.unlabeled_indices.append(idx)
            else:
                self.label_to_indices[sample.susi_label].append(idx)
        self.usable_label_to_indices = {
            label: indices
            for label, indices in self.label_to_indices.items()
            if len(indices) >= 2
        }
        self.label_counts = Counter(
            sample.susi_label for sample in self.samples if sample.susi_label is not None
        )
        self.override_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, sample in enumerate(self.samples):
            for key in sample.override_keys:
                self.override_to_indices[key].append(idx)
        self.usable_override_to_indices = {
            key: indices
            for key, indices in self.override_to_indices.items()
            if len(indices) >= 2
        }
        self.library_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, sample in enumerate(self.samples):
            for key in sample.library_keys:
                self.library_to_indices[key].append(idx)
        self.usable_library_to_indices = {
            key: indices
            for key, indices in self.library_to_indices.items()
            if len(indices) >= 2
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> MethodTextSample:
        return self.samples[index]

    @property
    def all_indices(self) -> list[int]:
        return list(range(len(self.samples)))

    def stats(self) -> dict[str, int]:
        return {
            "source_apks": len(self.source_shas),
            "samples_total": len(self.samples),
            "samples_labeled": sum(len(v) for v in self.label_to_indices.values()),
            "samples_unlabeled": len(self.unlabeled_indices),
            "susi_labels_total": len(self.label_to_indices),
            "susi_labels_usable": len(self.usable_label_to_indices),
            "override_keys_total": len(self.override_to_indices),
            "override_keys_usable": len(self.usable_override_to_indices),
            "library_keys_total": len(self.library_to_indices),
            "library_keys_usable": len(self.usable_library_to_indices),
            "skipped_filtered": self.skipped_filtered,
            "skipped_unlabeled": self.skipped_unlabeled,
        }

    @classmethod
    def from_methods_dir(
        cls,
        methods_dir: str | Path,
        *,
        sha_file: str | None = None,
        limit: int = 0,
        max_unlabeled_per_apk: int = 256,
        unlabeled_keep_ratio: float = 0.05,
        overrides_dir: str | Path | None = "data/overrides",
        libraries_dir: str | Path | None = "data/library_keys",
        seed: int = 13,
        show_progress: bool = True,
    ) -> "ContrastiveMethodDataset":
        """Build a sampled Phase 4 pool from ``methods/<SHA>.parquet`` files."""
        mdir = Path(methods_dir)
        shas = load_sha_list(sha_file, mdir)
        if limit and limit > 0:
            shas = shas[:limit]

        rng = random.Random(seed)
        samples: list[MethodTextSample] = []
        skipped_filtered = 0
        skipped_unlabeled = 0
        if overrides_dir is None:
            overrides_root = None
        else:
            overrides_root = Path(overrides_dir)
        if libraries_dir is None:
            libraries_root = None
        else:
            libraries_root = Path(libraries_dir)
        iterator = shas
        if show_progress:
            iterator = tqdm(shas, desc="phase4_dataset", unit="apk")

        for sha in iterator:
            pq = mdir / f"{sha}.parquet"
            if not pq.is_file():
                continue
            df = pd.read_parquet(pq, engine="pyarrow")
            if df.empty:
                continue
            if overrides_root is None:
                override_map: dict[str, list[str]] = {}
            else:
                from androserum.data.override_index import read_override_parquet

                override_map = read_override_parquet(overrides_root / f"{sha}.parquet")
            if libraries_root is None:
                library_map: dict[str, list[str]] = {}
            else:
                from androserum.data.library_index import read_library_parquet

                library_map = read_library_parquet(libraries_root / f"{sha}.parquet")

            filtered_mask = df["filtered"].fillna(False).astype(bool)
            skipped_filtered += int(filtered_mask.sum())
            df = df.loc[~filtered_mask].copy()
            if df.empty:
                continue

            labels = [normalize_susi_label(v) for v in df["susi_dominant_cat"].tolist()]
            df["phase4_label"] = labels

            labeled_rows = df[df["phase4_label"].notna()]
            unlabeled_rows = df[df["phase4_label"].isna()]

            unlabeled_idx = list(unlabeled_rows.index)
            if unlabeled_keep_ratio <= 0.0 or not unlabeled_idx:
                keep_unlabeled_idx: set[int] = set()
            else:
                keep_n = len(unlabeled_idx)
                if unlabeled_keep_ratio < 1.0:
                    keep_n = min(
                        keep_n,
                        max(1, int(round(len(unlabeled_idx) * unlabeled_keep_ratio))),
                    )
                if max_unlabeled_per_apk > 0:
                    keep_n = min(keep_n, max_unlabeled_per_apk)
                keep_unlabeled_idx = set(rng.sample(unlabeled_idx, k=keep_n))

            skipped_unlabeled += max(0, len(unlabeled_idx) - len(keep_unlabeled_idx))

            selected = pd.concat(
                [
                    labeled_rows,
                    unlabeled_rows.loc[sorted(keep_unlabeled_idx)] if keep_unlabeled_idx else unlabeled_rows.iloc[0:0],
                ],
                axis=0,
            )
            if selected.empty:
                continue

            for row in selected.itertuples(index=False):
                instructions = row.instructions
                if isinstance(instructions, np.ndarray):
                    instructions = instructions.tolist()
                elif not isinstance(instructions, list):
                    instructions = list(instructions)
                samples.append(
                    MethodTextSample(
                        apk_sha=str(row.apk_sha),
                        full_id=str(row.full_id),
                        text="\n".join(str(x) for x in instructions),
                        susi_label=normalize_susi_label(row.phase4_label),
                        override_keys=list(override_map.get(str(row.full_id), [])),
                        library_keys=list(library_map.get(str(row.full_id), [])),
                    )
                )

        if not samples:
            raise RuntimeError(
                "Phase 4 dataset is empty. Check methods_dir/sha_file and whether "
                "methods were filtered too aggressively."
            )
        return cls(
            samples=samples,
            source_shas=shas,
            skipped_filtered=skipped_filtered,
            skipped_unlabeled=skipped_unlabeled,
        )
