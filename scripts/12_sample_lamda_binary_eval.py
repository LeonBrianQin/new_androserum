#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sample a small balanced LAMDA subset for BU-based binary evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random

import fire
import pandas as pd


@dataclass
class LamdaSampleConfig:
    metadata_csv: str = "metadata.csv"
    out_csv: str = "configs/lamda_binary_eval_240_candidates.csv"
    out_summary_json: str = "configs/lamda_binary_eval_240_candidates.summary.json"
    seed: int = 13
    benign_primary: int = 100
    malware_primary: int = 100
    benign_reserve: int = 20
    malware_reserve: int = 20
    year_min: int = 2018
    year_max: int = 2023
    malware_family_cap: int = 5
    malware_family_cap_large: int = 8


def _sample_benign(
    df: pd.DataFrame,
    *,
    n_total: int,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    years = sorted(df["year"].dropna().astype(int).unique().tolist())
    per_year = max(1, n_total // max(1, len(years)))
    picks = []
    for year in years:
        sub = df[df["year"] == year].sample(frac=1.0, random_state=rng.randint(1, 10**9))
        picks.append(sub.head(per_year))
    out = pd.concat(picks, axis=0).drop_duplicates("sha256")
    if len(out) < n_total:
        remaining = df[~df["sha256"].isin(out["sha256"])].sample(
            frac=1.0, random_state=rng.randint(1, 10**9)
        )
        out = pd.concat([out, remaining.head(n_total - len(out))], axis=0)
    return out.head(n_total).copy()


def _sample_malware(
    df: pd.DataFrame,
    *,
    n_total: int,
    seed: int,
    family_cap: int,
    family_cap_large: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    years = sorted(df["year"].dropna().astype(int).unique().tolist())
    per_year = max(1, n_total // max(1, len(years)))
    yearly_picks = []
    for year in years:
        sub = df[df["year"] == year].copy()
        fam_counts = sub["family"].value_counts().to_dict()
        picked_rows = []
        for fam, fam_df in sub.groupby("family"):
            cap = family_cap_large if fam_counts.get(fam, 0) > 1000 else family_cap
            fam_df = fam_df.sample(frac=1.0, random_state=rng.randint(1, 10**9))
            picked_rows.append(fam_df.head(cap))
        merged = pd.concat(picked_rows, axis=0).drop_duplicates("sha256")
        merged = merged.sample(frac=1.0, random_state=rng.randint(1, 10**9))
        yearly_picks.append(merged.head(per_year))
    out = pd.concat(yearly_picks, axis=0).drop_duplicates("sha256")
    if len(out) < n_total:
        remaining = df[~df["sha256"].isin(out["sha256"])].copy()
        fam_counts = remaining["family"].value_counts().to_dict()
        picked_rows = []
        for fam, fam_df in remaining.groupby("family"):
            cap = family_cap_large if fam_counts.get(fam, 0) > 1000 else family_cap
            fam_df = fam_df.sample(frac=1.0, random_state=rng.randint(1, 10**9))
            picked_rows.append(fam_df.head(cap))
        merged = pd.concat(picked_rows, axis=0).drop_duplicates("sha256")
        merged = merged.sample(frac=1.0, random_state=rng.randint(1, 10**9))
        out = pd.concat([out, merged.head(n_total - len(out))], axis=0)
    return out.head(n_total).copy()


def main(**kwargs) -> None:
    cfg = LamdaSampleConfig(**kwargs)
    df = pd.read_csv(cfg.metadata_csv)
    df["sha256"] = df["sha256"].str.upper()
    df = df[(df["year"] >= cfg.year_min) & (df["year"] <= cfg.year_max)].copy()

    benign = df[df["label"] == 0].copy()
    malware = df[df["label"] == 1].copy()

    benign_n = cfg.benign_primary + cfg.benign_reserve
    malware_n = cfg.malware_primary + cfg.malware_reserve

    benign_pick = _sample_benign(benign, n_total=benign_n, seed=cfg.seed)
    malware_pick = _sample_malware(
        malware,
        n_total=malware_n,
        seed=cfg.seed + 1,
        family_cap=cfg.malware_family_cap,
        family_cap_large=cfg.malware_family_cap_large,
    )

    benign_pick = benign_pick.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    malware_pick = malware_pick.sample(frac=1.0, random_state=cfg.seed + 1).reset_index(drop=True)

    benign_pick["role"] = ["primary" if i < cfg.benign_primary else "reserve" for i in range(len(benign_pick))]
    malware_pick["role"] = ["primary" if i < cfg.malware_primary else "reserve" for i in range(len(malware_pick))]

    out = pd.concat([benign_pick, malware_pick], axis=0).reset_index(drop=True)
    out = out[
        ["sha256", "label", "family", "year", "year_month", "vt_detection", "role"]
    ].copy()
    Path(cfg.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg.out_csv, index=False)

    summary = {
        "config": asdict(cfg),
        "n_total": int(len(out)),
        "label_counts": out["label"].value_counts().to_dict(),
        "role_counts": out["role"].value_counts().to_dict(),
        "year_counts": out["year"].value_counts().sort_index().to_dict(),
        "benign_year_counts": out[out["label"] == 0]["year"].value_counts().sort_index().to_dict(),
        "malware_year_counts": out[out["label"] == 1]["year"].value_counts().sort_index().to_dict(),
        "top_malware_families": out[out["label"] == 1]["family"].value_counts().head(20).to_dict(),
        "median_vt_detection_malware": float(out[out["label"] == 1]["vt_detection"].median()),
        "median_vt_detection_benign": float(out[out["label"] == 0]["vt_detection"].median()),
    }
    Path(cfg.out_summary_json).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[lamda-sample] wrote {cfg.out_csv}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    fire.Fire(main)

