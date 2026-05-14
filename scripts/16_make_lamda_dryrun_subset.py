#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create tiny dry-run subsets from train/val/test LAMDA splits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import fire
import pandas as pd


@dataclass
class DryrunSubsetConfig:
    train_csv: str = "configs/lamda_train_160.csv"
    val_csv: str = "configs/lamda_val_40.csv"
    test_csv: str = "configs/lamda_test_40.csv"
    out_dir: str = "configs/lamda_dryrun"
    train_n_per_label: int = 3
    val_n_per_label: int = 2
    test_n_per_label: int = 2
    seed: int = 13


def _take_balanced(df: pd.DataFrame, n_per_label: int, seed: int) -> pd.DataFrame:
    parts = []
    for label in [0.0, 1.0]:
        sub = df[df["label"] == label].sample(frac=1.0, random_state=seed + int(label * 10))
        parts.append(sub.head(n_per_label))
    out = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed + 999).reset_index(drop=True)
    return out


def main(**kwargs) -> None:
    cfg = DryrunSubsetConfig(**kwargs)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(cfg.train_csv)
    val = pd.read_csv(cfg.val_csv)
    test = pd.read_csv(cfg.test_csv)

    train_small = _take_balanced(train, cfg.train_n_per_label, cfg.seed)
    val_small = _take_balanced(val, cfg.val_n_per_label, cfg.seed + 10)
    test_small = _take_balanced(test, cfg.test_n_per_label, cfg.seed + 20)

    train_path = out_dir / "lamda_train_dryrun.csv"
    val_path = out_dir / "lamda_val_dryrun.csv"
    test_path = out_dir / "lamda_test_dryrun.csv"
    train_small.to_csv(train_path, index=False)
    val_small.to_csv(val_path, index=False)
    test_small.to_csv(test_path, index=False)

    summary = {
        "config": asdict(cfg),
        "train": {
            "n": len(train_small),
            "rows": train_small[["sha256", "label", "family", "year", "role"]].to_dict("records"),
        },
        "val": {
            "n": len(val_small),
            "rows": val_small[["sha256", "label", "family", "year", "role"]].to_dict("records"),
        },
        "test": {
            "n": len(test_small),
            "rows": test_small[["sha256", "label", "family", "year", "role"]].to_dict("records"),
        },
    }
    (out_dir / "lamda_dryrun.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    fire.Fire(main)

