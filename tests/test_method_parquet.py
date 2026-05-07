"""Parquet round-trip tests for ``MethodRecord``."""

from pathlib import Path

import pytest

from androserum.data.method_parquet import read_methods_parquet, write_methods_parquet
from androserum.data.schema import MethodRecord


def _sample_rows() -> list[MethodRecord]:
    return [
        MethodRecord(
            apk_sha="A" * 64,
            class_name="Ljava/lang/Object;",
            method_sig="<init>()V",
            full_id="Ljava/lang/Object;-><init>()V",
            instructions=["return-void"] * 5,
            n_instr=5,
            api_calls=["Ljava/lang/System;->gc()V"],
            susi_cats=["NET"],
            susi_dominant_cat="NET",
            filtered=False,
        ),
        MethodRecord(
            apk_sha="A" * 64,
            class_name="Ljava/lang/Object;",
            method_sig="hashCode()I",
            full_id="Ljava/lang/Object;->hashCode()I",
            instructions=["const/4 v0, 0x0"] * 5,
            n_instr=5,
            api_calls=[],
            susi_cats=[],
            susi_dominant_cat=None,
            filtered=True,
        ),
    ]


def test_methods_parquet_roundtrip(tmp_path: Path):
    p = tmp_path / "out.parquet"
    orig = _sample_rows()
    write_methods_parquet(orig, p)
    loaded = read_methods_parquet(p)
    assert len(loaded) == 2
    assert loaded[0].model_dump() == orig[0].model_dump()
    assert loaded[1].model_dump() == orig[1].model_dump()


@pytest.mark.skipif(
    not Path(
        "data/processed/"
        "0D64BB3C121E1986766505E182F16FB8DCC4188224F3094F99B9F905873DDC4A.txt"
    ).is_file(),
    reason="sample processed txt not in workspace",
)
def test_write_real_sample_parquet_roundtrip(tmp_path: Path):
    from androserum.data.method_extractor import extract_methods

    txt = Path(
        "data/processed/"
        "0D64BB3C121E1986766505E182F16FB8DCC4188224F3094F99B9F905873DDC4A.txt"
    )
    rows = extract_methods(txt)
    out = tmp_path / "0D64.parquet"
    write_methods_parquet(rows, out)
    back = read_methods_parquet(out)
    assert len(back) == len(rows)
    assert back[0].full_id == rows[0].full_id
    assert back[-1].full_id == rows[-1].full_id
