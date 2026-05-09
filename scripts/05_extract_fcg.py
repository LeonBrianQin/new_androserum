#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 5 — APK -> Androguard FCG -> aligned sidecars.

Outputs per APK under ``data/fcg/``:

  * ``<SHA>.aligned_nodes.parquet``   — node rows aligned to Phase 2/3 row order
  * ``<SHA>.internal_edges.parquet``  — internal method -> internal method edges
  * ``<SHA>.boundary_edges.parquet``  — internal <-> external/non-aligned edges
  * ``<SHA>.summary.json``            — extraction/alignment diagnostics
"""

from __future__ import annotations

from pathlib import Path
import json

import fire
from tqdm import tqdm

from androserum.fcg import extract_fcg_bundle_for_apk


def _normalize_sha_token(s: str) -> str | None:
    s = s.strip().upper()
    if len(s) != 64:
        return None
    if any(c not in "0123456789ABCDEF" for c in s):
        return None
    return s


def load_sha_list(sha_file: str | None, methods_dir: Path) -> list[str]:
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

    return sorted(
        {
            p.stem.upper()
            for p in methods_dir.glob("*.parquet")
            if _normalize_sha_token(p.stem) is not None
        }
    )


def _load_entry_points(entry_points_file: str | None) -> list[str]:
    if not entry_points_file:
        return []
    p = Path(entry_points_file)
    if not p.is_file():
        raise FileNotFoundError(f"entry_points_file not found: {p}")
    out: list[str] = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def _bundle_exists(out_dir: Path, sha: str) -> bool:
    expected = [
        out_dir / f"{sha}.aligned_nodes.parquet",
        out_dir / f"{sha}.internal_edges.parquet",
        out_dir / f"{sha}.boundary_edges.parquet",
        out_dir / f"{sha}.summary.json",
    ]
    return all(p.is_file() for p in expected)


def main(
    sha_file: str | None = "configs/sha_dev_200.txt",
    apks_dir: str = "data/apks",
    methods_dir: str = "data/methods",
    out_dir: str = "data/fcg",
    entry_points_file: str | None = None,
    limit: int = 0,
    no_isolated: bool = False,
    include_boundary_edges: bool = True,
    skip_existing: bool = True,
) -> None:
    apks_root = Path(apks_dir)
    methods_root = Path(methods_dir)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    entry_points = _load_entry_points(entry_points_file)
    shas = load_sha_list(sha_file, methods_root)
    if limit and limit > 0:
        shas = shas[:limit]

    written = 0
    skipped_existing_count = 0
    missing_inputs: list[str] = []
    summary_rows: list[dict] = []

    for sha in tqdm(shas, desc="phase5-fcg", unit="apk"):
        apk_path = apks_root / f"{sha}.apk"
        methods_path = methods_root / f"{sha}.parquet"
        if not apk_path.is_file() or not methods_path.is_file():
            missing_inputs.append(sha)
            continue
        if skip_existing and _bundle_exists(out_root, sha):
            skipped_existing_count += 1
            continue

        summary = extract_fcg_bundle_for_apk(
            apk_path,
            methods_path,
            out_root,
            apk_sha=sha,
            no_isolated=no_isolated,
            include_boundary_edges=include_boundary_edges,
            entry_points=entry_points,
        )
        summary_rows.append(summary.__dict__)
        written += 1

    payload = {
        "requested_shas": len(shas),
        "written_bundles": written,
        "skipped_existing": skipped_existing_count,
        "missing_inputs_count": len(missing_inputs),
        "missing_inputs_sample": missing_inputs[:20],
        "entry_points_count": len(entry_points),
        "out_dir": str(out_root),
        "per_apk_summaries_written_this_run": summary_rows,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    fire.Fire(main)
