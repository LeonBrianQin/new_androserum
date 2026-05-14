#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split LAMDA-mini candidates into train/val/test CSVs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import fire
import pandas as pd


@dataclass
class SplitConfig:
    in_csv: str = "configs/lamda_binary_eval_240_candidates.csv"
    out_train: str = "configs/lamda_train_160.csv"
    out_val: str = "configs/lamda_val_40.csv"
    out_test: str = "configs/lamda_test_40.csv"
    out_summary: str = "configs/lamda_train_val_test.summary.json"
    seed: int = 13


def _balanced_take(df: pd.DataFrame, n_benign: int, n_malware: int, seed: int) -> pd.DataFrame:
    benign = df[df["label"] == 0.0].sample(frac=1.0, random_state=seed)
    malware = df[df["label"] == 1.0].sample(frac=1.0, random_state=seed + 1)
    out = pd.concat([benign.head(n_benign), malware.head(n_malware)], axis=0)
    return out.sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)


def main(**kwargs) -> None:
    cfg = SplitConfig(**kwargs)
    df = pd.read_csv(cfg.in_csv)

    primary = df[df["role"] == "primary"].copy()
    reserve = df[df["role"] == "reserve"].copy()

    train = _balanced_take(primary, 80, 80, cfg.seed)
    remain = primary[~primary["sha256"].isin(train["sha256"])].copy()
    val = _balanced_take(remain, 20, 20, cfg.seed + 10)
    test = _balanced_take(reserve, 20, 20, cfg.seed + 20)

    Path(cfg.out_train).parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(cfg.out_train, index=False)
    val.to_csv(cfg.out_val, index=False)
    test.to_csv(cfg.out_test, index=False)

    def stats_of(x: pd.DataFrame) -> dict:
        return {
            "n": int(len(x)),
            "label_counts": x["label"].value_counts().to_dict(),
            "year_counts": x["year"].value_counts().sort_index().to_dict(),
            "top_families": x[x["label"] == 1.0]["family"].value_counts().head(15).to_dict(),
        }

    summary = {
        "config": asdict(cfg),
        "train": stats_of(train),
        "val": stats_of(val),
        "test": stats_of(test),
    }
    Path(cfg.out_summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    fire.Fire(main)

