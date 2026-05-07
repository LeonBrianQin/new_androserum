"""Unit tests for SuSi list parsing and Dalvik normalization."""

from __future__ import annotations

from pathlib import Path

from androserum.data.schema import MethodRecord
from androserum.data.susi_index import (
    build_susi_index,
    dalvik_api_from_soot_line,
    parse_susi_data_line,
)
from androserum.data.susi_tagger import tag_method_susi

_SAMPLE_SUSI = """\
UNIQUE_IDENTIFIER:
<android.telephony.TelephonyManager: java.lang.String getDeviceId()> (UNIQUE_IDENTIFIER)
<android.telephony.PhoneNumberUtils: byte[] numberToCalledPartyBCD(java.lang.String)> (UNIQUE_IDENTIFIER)
"""


def test_dalvik_from_soot_get_device_id():
    sig = "<android.telephony.TelephonyManager: java.lang.String getDeviceId()>"
    assert dalvik_api_from_soot_line(sig) == (
        "Landroid/telephony/TelephonyManager;->getDeviceId()Ljava/lang/String;"
    )


def test_dalvik_from_soot_byte_array_param():
    sig = "<android.telephony.PhoneNumberUtils: byte[] numberToCalledPartyBCD(java.lang.String)>"
    assert dalvik_api_from_soot_line(sig) == (
        "Landroid/telephony/PhoneNumberUtils;->numberToCalledPartyBCD(Ljava/lang/String;)[B"
    )


def test_parse_susi_line_with_permission_tail():
    line = (
        "<android.telephony.TelephonyManager: java.lang.String getDeviceId()> "
        "android.permission.READ_PHONE_STATE (UNIQUE_IDENTIFIER)"
    )
    soot, cat = parse_susi_data_line(line)
    assert cat == "UNIQUE_IDENTIFIER"
    assert soot == "<android.telephony.TelephonyManager: java.lang.String getDeviceId()>"


def test_parse_skips_lines_without_category_tail():
    assert parse_susi_data_line("<android.telephony.TelephonyManager: int x()> tail") is None


def test_build_index_and_tag(tmp_path: Path):
    p = tmp_path / "mini.txt"
    p.write_text(_SAMPLE_SUSI, encoding="utf-8")
    idx = build_susi_index([p])
    assert "Landroid/telephony/TelephonyManager;->getDeviceId()Ljava/lang/String;" in idx
    row = MethodRecord(
        apk_sha="0" * 64,
        class_name="Lcom/x/Y;",
        method_sig="a()V",
        full_id="Lcom/x/Y;->a()V",
        instructions=[
            "invoke-virtual {v0}, Landroid/telephony/TelephonyManager;"
            "->getDeviceId()Ljava/lang/String;"
        ],
        n_instr=1,
        api_calls=[
            "Landroid/telephony/TelephonyManager;->getDeviceId()Ljava/lang/String;"
        ],
        filtered=False,
    )
    tagged = tag_method_susi(row, idx)
    assert tagged.susi_cats == ["UNIQUE_IDENTIFIER"]
    assert tagged.susi_dominant_cat == "UNIQUE_IDENTIFIER"
