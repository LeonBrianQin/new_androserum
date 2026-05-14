"""Phase 6 BGRL trainer over Phase 5 FCG sidecars."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm
import yaml

from androserum.encoder.utils import set_seeds
from androserum.gnn.augment import drop_edge, mask_node_features
from androserum.gnn.bgrl import BgrlModel, bgrl_regression_loss
from androserum.gnn.dataset import FcgGraphDataset, FcgGraphSample

__all__ = [
    "GnnTrainConfig",
    "export_gnn_embeddings",
    "load_gnn_config",
    "save_gnn_summary",
    "train_bgrl_graphsage",
]


@dataclass
class GnnTrainConfig:
    """User-facing Phase 6 configuration."""

    sha_file: str | None = "configs/sha_dev_200.txt"
    fcg_dir: str = "data/fcg"
    embeddings_dir: str = "data/embeddings/finetuned/p4_dev200_abe_run1"
    checkpoint_dir: str = "data/checkpoints/gnn_bgrl"
    gnn_embeddings_dir: str = "data/gnn_embeddings/gnn_bgrl"
    device: str = "cuda"
    graph_mode: str = "internal_only"
    external_prior_mode: str = "none"
    add_reverse_edges: bool = True
    hidden_dim: int = 256
    output_dim: int = 256
    predictor_hidden_dim: int = 512
    encoder_dropout: float = 0.1
    edge_drop_prob: float = 0.3
    feature_mask_prob: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 20
    ema_decay: float = 0.99
    limit: int = 0
    seed: int = 13
    log_every: int = 10
    exclude_filtered_from_loss: bool = True
    loss_scope: str = "internal"
    export_after_train: bool = True
    export_encoder: str = "target"
    cfg_path: str | None = None


def load_gnn_config(cfg_path: str | Path) -> GnnTrainConfig:
    """Load a YAML config file into ``GnnTrainConfig``."""
    p = Path(cfg_path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"expected mapping in {p}, got {type(raw)!r}")
    return GnnTrainConfig(**raw)


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def save_gnn_summary(path: str | Path, summary: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _training_mask(sample: FcgGraphSample, cfg: GnnTrainConfig) -> torch.Tensor:
    if cfg.loss_scope not in {"internal", "all"}:
        raise ValueError(f"unsupported loss_scope: {cfg.loss_scope}")
    if cfg.loss_scope == "internal":
        mask = sample.internal_mask.clone()
    else:
        mask = torch.ones_like(sample.internal_mask, dtype=torch.bool)
    if cfg.exclude_filtered_from_loss:
        mask = mask & (~sample.filtered_mask | ~sample.internal_mask)
    return mask


def _move_sample_to_device(sample: FcgGraphSample, device: torch.device) -> FcgGraphSample:
    return FcgGraphSample(
        apk_sha=sample.apk_sha,
        x=sample.x.to(device),
        edge_index=sample.edge_index.to(device),
        node_kind=sample.node_kind.to(device),
        filtered_mask=sample.filtered_mask.to(device),
        internal_mask=sample.internal_mask.to(device),
        family_keys=list(sample.family_keys),
        family_ids=sample.family_ids.to(device),
        full_ids=list(sample.full_ids),
        internal_full_ids=list(sample.internal_full_ids),
        internal_node_indices=sample.internal_node_indices.to(device),
    )


def _make_view(
    sample: FcgGraphSample,
    *,
    cfg: GnnTrainConfig,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = mask_node_features(
        sample.x,
        mask_prob=cfg.feature_mask_prob,
        training=training,
    )
    edge_index = drop_edge(
        sample.edge_index,
        drop_prob=cfg.edge_drop_prob,
        training=training,
    )
    return x, edge_index


def _graph_loss(
    *,
    model: BgrlModel,
    sample: FcgGraphSample,
    cfg: GnnTrainConfig,
) -> torch.Tensor:
    x1, edge1 = _make_view(sample, cfg=cfg, training=True)
    x2, edge2 = _make_view(sample, cfg=cfg, training=True)

    h1 = model.encode_online(x1, edge1, sample.node_kind, sample.family_ids)
    h2 = model.encode_online(x2, edge2, sample.node_kind, sample.family_ids)
    p1 = model.predict(h1)
    p2 = model.predict(h2)

    with torch.no_grad():
        z1 = model.encode_target(x1, edge1, sample.node_kind, sample.family_ids)
        z2 = model.encode_target(x2, edge2, sample.node_kind, sample.family_ids)

    mask = _training_mask(sample, cfg)
    if not bool(mask.any()):
        raise ValueError(f"empty training mask for APK {sample.apk_sha}")

    loss_12 = bgrl_regression_loss(p1[mask], z2[mask]).mean()
    loss_21 = bgrl_regression_loss(p2[mask], z1[mask]).mean()
    return 0.5 * (loss_12 + loss_21)


def _save_checkpoint(
    path: Path,
    *,
    model: BgrlModel,
    optimizer: torch.optim.Optimizer,
    cfg: GnnTrainConfig,
    epoch: int,
    mean_loss: float,
    dataset_stats: dict[str, Any],
    family_to_id: dict[str, int] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "mean_loss": mean_loss,
        "train_config": asdict(cfg),
        "dataset_stats": dataset_stats,
        "family_to_id": dict(family_to_id or {}),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(payload, path)


def export_gnn_embeddings(
    *,
    model: BgrlModel,
    dataset: FcgGraphDataset,
    out_dir: str | Path,
    device: str | torch.device,
    encoder_name: str = "target",
) -> int:
    """Encode every graph and write internal-node embeddings in aligned order."""
    dev = _resolve_device(device)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    if encoder_name not in {"target", "online"}:
        raise ValueError(f"unsupported export_encoder: {encoder_name}")
    encoder = model.target_encoder if encoder_name == "target" else model.online_encoder
    encoder.eval()

    written = 0
    for idx in tqdm(range(len(dataset)), desc="phase6_export", unit="apk"):
        sample = _move_sample_to_device(dataset.load(idx), dev)
        with torch.no_grad():
            h = encoder(sample.x, sample.edge_index, sample.node_kind, sample.family_ids)
        internal = h[sample.internal_mask].detach().float().cpu().numpy()
        np.savez_compressed(
            out_root / f"{sample.apk_sha}.npz",
            full_id=np.array(sample.internal_full_ids, dtype=object),
            embedding=internal.astype(np.float32, copy=False),
        )
        written += 1
    return written


def train_bgrl_graphsage(cfg: GnnTrainConfig) -> dict[str, Any]:
    """Train Phase 6 GraphSAGE+BGRL on Phase 5 FCG sidecars."""
    if cfg.graph_mode == "internal_only" and cfg.external_prior_mode != "none":
        raise ValueError("internal_only mode must use external_prior_mode='none'")

    set_seeds(cfg.seed)
    dev = _resolve_device(cfg.device)
    dataset = FcgGraphDataset.from_dirs(
        fcg_dir=cfg.fcg_dir,
        embeddings_dir=cfg.embeddings_dir,
        sha_file=cfg.sha_file,
        limit=cfg.limit,
        graph_mode=cfg.graph_mode,
        external_prior_mode=cfg.external_prior_mode,
        add_reverse_edges=cfg.add_reverse_edges,
    )
    dataset_stats = dataset.stats()

    print("[phase6] dataset stats:")
    for key, value in dataset_stats.items():
        print(f"  - {key}: {value}")

    if len(dataset) == 0:
        raise RuntimeError("no non-empty Phase 6 graphs found")

    sample0 = dataset.load(0)
    model = BgrlModel(
        input_dim=int(sample0.x.shape[1]),
        hidden_dim=cfg.hidden_dim,
        output_dim=cfg.output_dim,
        predictor_hidden_dim=cfg.predictor_hidden_dim,
        dropout=cfg.encoder_dropout,
        external_prior_mode=cfg.external_prior_mode,
        external_family_vocab=len(dataset.family_to_id),
        ema_decay=cfg.ema_decay,
    ).to(dev)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    ckpt_root = Path(cfg.checkpoint_dir)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_root / "bgrl_graphsage_best.pt"
    last_path = ckpt_root / "bgrl_graphsage_last.pt"
    summary_path = ckpt_root / "bgrl_graphsage_summary.json"

    best_loss = float("inf")
    history: list[dict[str, float]] = []
    skipped_empty_training_graphs = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = list(range(len(dataset)))
        random.Random(cfg.seed + epoch).shuffle(order)
        loss_values: list[float] = []
        skipped_this_epoch = 0

        pbar = tqdm(order, desc=f"phase6_train e{epoch}", unit="apk")
        for step, idx in enumerate(pbar, start=1):
            sample = _move_sample_to_device(dataset.load(idx), dev)
            optimizer.zero_grad(set_to_none=True)
            try:
                loss = _graph_loss(model=model, sample=sample, cfg=cfg)
            except ValueError as exc:
                if "empty training mask" not in str(exc):
                    raise
                skipped_empty_training_graphs += 1
                skipped_this_epoch += 1
                continue
            loss.backward()
            optimizer.step()
            model.update_target_network()

            loss_values.append(float(loss.item()))
            if step % cfg.log_every == 0 or step == len(order):
                pbar.set_postfix(loss=f"{sum(loss_values) / len(loss_values):.4f}")

        mean_loss = float(sum(loss_values) / max(1, len(loss_values)))
        history.append(
            {
                "epoch": float(epoch),
                "mean_loss": mean_loss,
                "skipped_empty_training_graphs": float(skipped_this_epoch),
            }
        )

        _save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            cfg=cfg,
            epoch=epoch,
            mean_loss=mean_loss,
            dataset_stats=dataset_stats,
            family_to_id=dataset.family_to_id,
        )
        if mean_loss < best_loss:
            best_loss = mean_loss
            _save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                epoch=epoch,
                mean_loss=mean_loss,
                dataset_stats=dataset_stats,
                family_to_id=dataset.family_to_id,
            )

        summary = {
            "best_checkpoint": str(best_path),
            "best_mean_loss": best_loss,
            "dataset_stats": dataset_stats,
            "history": history,
            "last_checkpoint": str(last_path),
            "skipped_empty_training_graphs_total": skipped_empty_training_graphs,
            "train_config": asdict(cfg),
        }
        save_gnn_summary(summary_path, summary)

    exported = 0
    if cfg.export_after_train:
        exported = export_gnn_embeddings(
            model=model,
            dataset=dataset,
            out_dir=cfg.gnn_embeddings_dir,
            device=dev,
            encoder_name=cfg.export_encoder,
        )

    summary = {
        "best_checkpoint": str(best_path),
        "best_mean_loss": best_loss,
        "dataset_stats": dataset_stats,
        "exported_npz_files": exported,
        "history": history,
        "last_checkpoint": str(last_path),
        "skipped_empty_training_graphs_total": skipped_empty_training_graphs,
        "train_config": asdict(cfg),
    }
    save_gnn_summary(summary_path, summary)
    return summary
