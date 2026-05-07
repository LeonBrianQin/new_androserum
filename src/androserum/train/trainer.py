"""Trainer and export helpers for Phase 4 A+B contrastive fine-tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from androserum.data.method_parquet import read_methods_parquet
from androserum.encoder.utils import set_seeds
from androserum.train.contrastive_model import ContrastiveDexBertModel
from androserum.train.dataset import (
    ContrastiveMethodDataset,
    build_contrastive_collate_fn,
    load_sha_list,
    texts_to_model_inputs,
)
from androserum.train.losses import ab_contrastive_loss, count_b_positive_pairs
from androserum.train.samplers import SusiGroupBatchSampler

__all__ = [
    "ContrastiveTrainConfig",
    "export_finetuned_embeddings",
    "save_training_summary",
    "train_contrastive_ab",
]


@dataclass
class ContrastiveTrainConfig:
    """User-facing Phase 4 configuration."""

    sha_file: str | None = "configs/sha_dev_200.txt"
    methods_dir: str = "data/methods"
    checkpoint_dir: str = "data/checkpoints"
    finetuned_dir: str = "data/embeddings/finetuned"
    device: str = "cuda"
    batch_size: int = 16
    epochs: int = 10
    lr: float = 2e-5
    weight_decay: float = 1e-2
    temperature: float = 0.07
    projection_dim: int = 256
    freeze_n_layers: int = 4
    freeze_embeddings: bool = True
    label_group_size: int = 2
    label_fraction: float = 0.5
    steps_per_epoch: int = 1000
    max_unlabeled_per_apk: int = 256
    unlabeled_keep_ratio: float = 0.05
    num_workers: int = 0
    limit: int = 0
    seed: int = 13
    log_every: int = 20
    grad_clip_norm: float = 1.0
    export_after_train: bool = False
    cfg_path: str | None = None
    weights_path: str | None = None
    vocab_path: str | None = None


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _save_checkpoint(
    path: Path,
    *,
    model: ContrastiveDexBertModel,
    optimizer: torch.optim.Optimizer,
    cfg: ContrastiveTrainConfig,
    epoch: int,
    step: int,
    mean_loss: float,
    dataset_stats: dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "step": step,
        "mean_loss": mean_loss,
        "train_config": asdict(cfg),
        "dataset_stats": dataset_stats,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "projection_dim": model.projection.out_features,
        "encoder_dim": model.cfg.dim,
    }
    torch.save(payload, path)


def save_training_summary(path: str | Path, summary: dict[str, Any]) -> None:
    """Write a small JSON summary for experiment bookkeeping."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def export_finetuned_embeddings(
    *,
    model: ContrastiveDexBertModel,
    tokenizer,
    encoder_cfg,
    methods_dir: str | Path,
    out_dir: str | Path,
    batch_size: int,
    device: str | torch.device,
    sha_file: str | None = None,
    limit: int = 0,
) -> int:
    """Encode every method with the *finetuned encoder CLS* and write ``.npz`` files."""
    dev = _resolve_device(device)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    methods_root = Path(methods_dir)
    shas = load_sha_list(sha_file, methods_root)
    if limit and limit > 0:
        shas = shas[:limit]

    model.eval()
    written = 0
    for sha in tqdm(shas, desc="phase4_export", unit="apk"):
        pq = methods_root / f"{sha}.parquet"
        if not pq.is_file():
            continue
        rows = read_methods_parquet(pq)
        texts = ["\n".join(r.instructions) for r in rows]
        full_ids = [r.full_id for r in rows]
        out_parts: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            input_ids, seg_ids, mask = texts_to_model_inputs(chunk, tokenizer, encoder_cfg.max_len)
            input_ids = input_ids.to(dev)
            seg_ids = seg_ids.to(dev)
            mask = mask.to(dev)
            with torch.no_grad():
                cls_vecs = model.encode_cls(input_ids, seg_ids, mask)
            out_parts.append(cls_vecs.detach().float().cpu().numpy())

        if out_parts:
            emb = np.concatenate(out_parts, axis=0)
        else:
            emb = np.zeros((0, encoder_cfg.dim), dtype=np.float32)

        np.savez_compressed(
            out_root / f"{sha}.npz",
            full_id=np.array(full_ids, dtype=object),
            embedding=emb.astype(np.float32, copy=False),
        )
        written += 1
    return written


def train_contrastive_ab(cfg: ContrastiveTrainConfig) -> dict[str, Any]:
    """Run Phase 4 A+B contrastive fine-tuning and optionally export embeddings."""
    set_seeds(cfg.seed)
    dev = _resolve_device(cfg.device)

    dataset = ContrastiveMethodDataset.from_methods_dir(
        cfg.methods_dir,
        sha_file=cfg.sha_file,
        limit=cfg.limit,
        max_unlabeled_per_apk=cfg.max_unlabeled_per_apk,
        unlabeled_keep_ratio=cfg.unlabeled_keep_ratio,
        seed=cfg.seed,
        show_progress=True,
    )
    dataset_stats = dataset.stats()

    print("[phase4] dataset stats:")
    for key, value in dataset_stats.items():
        print(f"  - {key}: {value}")

    model, tokenizer, encoder_cfg = ContrastiveDexBertModel.from_pretrained(
        projection_dim=cfg.projection_dim,
        cfg_path=cfg.cfg_path,
        weights_path=cfg.weights_path,
        vocab_path=cfg.vocab_path,
        device=dev,
    )
    model.freeze_bottom_layers(
        cfg.freeze_n_layers,
        freeze_embeddings=cfg.freeze_embeddings,
    )
    print(f"[phase4] device: {dev}")
    print(f"[phase4] trainable params: {model.trainable_parameter_count():,}")

    batch_sampler = SusiGroupBatchSampler(
        all_indices=dataset.all_indices,
        label_to_indices=dataset.usable_label_to_indices,
        batch_size=cfg.batch_size,
        label_group_size=cfg.label_group_size,
        label_fraction=cfg.label_fraction,
        steps_per_epoch=cfg.steps_per_epoch,
        seed=cfg.seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=build_contrastive_collate_fn(tokenizer, encoder_cfg),
        num_workers=cfg.num_workers,
    )

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    ckpt_root = Path(cfg.checkpoint_dir)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    last_ckpt = ckpt_root / "contrastive_ab_last.pt"
    best_ckpt = ckpt_root / "contrastive_ab_best.pt"

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        batch_sampler.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_batches = 0

        for step, batch in enumerate(loader, start=1):
            input_ids = batch.input_ids.to(dev)
            seg_ids = batch.seg_ids.to(dev)
            mask = batch.mask.to(dev)

            optimizer.zero_grad(set_to_none=True)
            _, proj1 = model(input_ids, seg_ids, mask)
            _, proj2 = model(input_ids, seg_ids, mask)
            loss = ab_contrastive_loss(
                proj1,
                proj2,
                batch.susi_labels,
                temperature=cfg.temperature,
            )
            loss.backward()
            if cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
            optimizer.step()

            batch_loss = float(loss.detach().cpu().item())
            epoch_loss += batch_loss
            epoch_batches += 1
            global_step += 1

            if step == 1 or step % cfg.log_every == 0 or step == len(loader):
                labeled = sum(label is not None for label in batch.susi_labels)
                b_pairs = count_b_positive_pairs(batch.susi_labels)
                print(
                    "[phase4] "
                    f"epoch={epoch}/{cfg.epochs} "
                    f"step={step}/{len(loader)} "
                    f"loss={batch_loss:.4f} "
                    f"labeled_in_batch={labeled}/{len(batch.susi_labels)} "
                    f"b_pairs={b_pairs}"
                )

        mean_loss = epoch_loss / max(1, epoch_batches)
        history.append({"epoch": float(epoch), "mean_loss": float(mean_loss)})
        print(f"[phase4] epoch {epoch} mean_loss={mean_loss:.4f}")

        _save_checkpoint(
            last_ckpt,
            model=model,
            optimizer=optimizer,
            cfg=cfg,
            epoch=epoch,
            step=global_step,
            mean_loss=mean_loss,
            dataset_stats=dataset_stats,
        )
        if mean_loss < best_loss:
            best_loss = mean_loss
            _save_checkpoint(
                best_ckpt,
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                epoch=epoch,
                step=global_step,
                mean_loss=mean_loss,
                dataset_stats=dataset_stats,
            )

    summary: dict[str, Any] = {
        "train_config": asdict(cfg),
        "dataset_stats": dataset_stats,
        "history": history,
        "best_mean_loss": best_loss,
        "last_checkpoint": str(last_ckpt),
        "best_checkpoint": str(best_ckpt),
    }

    if cfg.export_after_train:
        written = export_finetuned_embeddings(
            model=model,
            tokenizer=tokenizer,
            encoder_cfg=encoder_cfg,
            methods_dir=cfg.methods_dir,
            out_dir=cfg.finetuned_dir,
            batch_size=cfg.batch_size,
            device=dev,
            sha_file=cfg.sha_file,
            limit=cfg.limit,
        )
        summary["exported_npz_files"] = written
        print(f"[phase4] exported {written} finetuned embedding files")

    save_training_summary(ckpt_root / "contrastive_ab_summary.json", summary)
    return summary
