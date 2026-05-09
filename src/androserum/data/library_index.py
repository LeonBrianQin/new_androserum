"""Signal C: cross-APK same-library/same-method weak positives.

The handoff recommends LibScout / LibRadar as the strongest cross-APK signal.
To prepare the pipeline *now* without introducing a heavyweight external
dependency into the repo, we implement a high-precision sidecar based on exact
``full_id`` matches across >= 2 APKs.

This is intentionally a provider-style module. The Phase 4 code only depends on
``library_keys`` sidecars, so a stronger extractor can later replace this exact
matching implementation without changing the training pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from androserum.train.dataset import load_sha_list

__all__ = [
    "MethodLibraryRecord",
    "build_exact_full_id_library_sidecars",
    "read_library_parquet",
    "write_library_parquet",
]


@dataclass(frozen=True)
class MethodLibraryRecord:
    apk_sha: str
    full_id: str
    library_keys: list[str]


def _key_for_full_id(full_id: str) -> str:
    return f"EXACT_FULL_ID::{full_id}"


def write_library_parquet(rows: list[MethodLibraryRecord], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "apk_sha": [r.apk_sha for r in rows],
            "full_id": [r.full_id for r in rows],
            "library_keys": [r.library_keys for r in rows],
        }
    )
    df.to_parquet(p, index=False, engine="pyarrow")


def read_library_parquet(path: str | Path) -> dict[str, list[str]]:
    p = Path(path)
    if not p.is_file():
        return {}
    df = pd.read_parquet(p, engine="pyarrow")
    out: dict[str, list[str]] = {}
    for rec in df.to_dict("records"):
        full_id = str(rec["full_id"])
        keys = rec.get("library_keys")
        if keys is None:
            out[full_id] = []
        elif isinstance(keys, list):
            out[full_id] = [str(x) for x in keys]
        else:
            out[full_id] = list(keys)
    return out


def build_exact_full_id_library_sidecars(
    *,
    methods_dir: str | Path,
    out_dir: str | Path,
    sha_file: str | None = "configs/sha_dev_200.txt",
    limit: int = 0,
    min_apk_support: int = 2,
) -> dict[str, int]:
    """Materialize ``data/library_keys/<SHA>.parquet`` via exact ``full_id`` support.

    A method receives a non-empty ``library_keys`` list iff its ``full_id``
    appears in at least ``min_apk_support`` distinct APKs.
    """
    methods_root = Path(methods_dir)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    shas = load_sha_list(sha_file, methods_root)
    if limit and limit > 0:
        shas = shas[:limit]

    full_id_to_apks: dict[str, set[str]] = {}
    rows_by_sha: dict[str, list[str]] = {}
    missing = 0
    empty_schema = 0

    for sha in shas:
        pq = methods_root / f"{sha}.parquet"
        if not pq.is_file():
            missing += 1
            continue
        df = pd.read_parquet(pq, engine="pyarrow")
        if df.empty and len(df.columns) == 0:
            empty_schema += 1
            continue
        full_ids = [str(x) for x in df["full_id"].tolist()]
        rows_by_sha[sha] = full_ids
        for fid in set(full_ids):
            full_id_to_apks.setdefault(fid, set()).add(sha)

    supported = {
        fid for fid, apks in full_id_to_apks.items() if len(apks) >= min_apk_support
    }

    written = 0
    linked_rows = 0
    for sha, full_ids in rows_by_sha.items():
        rows: list[MethodLibraryRecord] = []
        for fid in full_ids:
            keys = [_key_for_full_id(fid)] if fid in supported else []
            if keys:
                linked_rows += 1
            rows.append(
                MethodLibraryRecord(
                    apk_sha=sha,
                    full_id=fid,
                    library_keys=keys,
                )
            )
        write_library_parquet(rows, out_root / f"{sha}.parquet")
        written += 1

    return {
        "requested_shas": len(shas),
        "written_sidecars": written,
        "missing_method_parquets": missing,
        "empty_schema_parquets": empty_schema,
        "supported_library_keys": len(supported),
        "linked_rows": linked_rows,
    }
