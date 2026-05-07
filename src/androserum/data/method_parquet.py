"""Parquet I/O for :class:`~androserum.data.schema.MethodRecord` tables.

Used by Phase 2 (``scripts/02_extract_methods.py``) and later stages that
read ``data/methods/<sha>.parquet`` without re-parsing ``processed/*.txt``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from androserum.data.schema import MethodRecord


def write_methods_parquet(rows: Iterable[MethodRecord], path: str | Path) -> None:
    """Write one APK's methods to a single Parquet file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [r.model_dump() for r in rows]
    df = pd.DataFrame.from_records(records)
    df.to_parquet(path, index=False, engine="pyarrow")


def read_methods_parquet(path: str | Path) -> list[MethodRecord]:
    """Load ``MethodRecord`` rows written by :func:`write_methods_parquet`."""
    df = pd.read_parquet(path, engine="pyarrow")
    rows: list[MethodRecord] = []
    for raw in df.to_dict("records"):
        rec: dict = {}
        for k, v in raw.items():
            if v is None:
                rec[k] = None
            elif isinstance(v, float) and pd.isna(v):
                rec[k] = None
            elif isinstance(v, np.ndarray):
                rec[k] = v.tolist()
            else:
                rec[k] = v
        rows.append(MethodRecord(**rec))
    return rows


__all__ = ["read_methods_parquet", "write_methods_parquet"]
