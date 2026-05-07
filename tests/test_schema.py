"""Tests for ``androserum.data.schema.MethodRecord``."""

import pytest
from pydantic import ValidationError

from androserum.data.schema import MethodRecord, make_full_id


def test_make_full_id():
    assert (
        make_full_id("Ljava/lang/Object;", "<init>()V")
        == "Ljava/lang/Object;-><init>()V"
    )


def test_method_record_ok():
    r = MethodRecord(
        apk_sha="0D64BB3C121E1986766505E182F16FB8DCC4188224F3094F99B9F905873DDC4A",
        class_name="Ljava/lang/Object;",
        method_sig="<init>()V",
        full_id="Ljava/lang/Object;-><init>()V",
        instructions=["return-void"],
        n_instr=1,
        api_calls=[],
        susi_cats=[],
        susi_dominant_cat=None,
        filtered=False,
    )
    assert r.apk_sha.isupper()
    assert r.n_instr == 1


def test_apk_sha_normalized_to_upper():
    r = MethodRecord(
        apk_sha="0d64bb3c121e1986766505e182f16fb8dcc4188224f3094f99b9f905873ddc4a",
        class_name="Ljava/lang/Object;",
        method_sig="<init>()V",
        full_id="Ljava/lang/Object;-><init>()V",
        instructions=[],
        n_instr=0,
    )
    assert r.apk_sha.isupper()
    assert (
        r.apk_sha
        == "0D64BB3C121E1986766505E182F16FB8DCC4188224F3094F99B9F905873DDC4A"
    )


def test_apk_sha_mixed_case_normalized():
    """CSV / lists often have arbitrary casing; AndroZoo accepts hex case-insensitively."""
    r = MethodRecord(
        apk_sha="0d64BB3c121E1986766505E182f16FB8dcc4188224F3094F99b9F905873Ddc4a",
        class_name="Ljava/lang/Object;",
        method_sig="<init>()V",
        full_id="Ljava/lang/Object;-><init>()V",
        instructions=[],
        n_instr=0,
    )
    assert r.apk_sha == "0D64BB3C121E1986766505E182F16FB8DCC4188224F3094F99B9F905873DDC4A"


def test_class_name_must_be_descriptor():
    with pytest.raises(ValidationError):
        MethodRecord(
            apk_sha="0" * 64,
            class_name="java/lang/Object",  # missing L and ;
            method_sig="<init>()V",
            full_id="java/lang/Object-><init>()V",  # will also fail full_id check
            instructions=[],
            n_instr=0,
        )


def test_class_name_allows_hyphen_segment():
    """Real APKs have classes like 'Lcom/.../auth-api-phone/zzj;' (R8/D8 packaging).

    Per Dalvik dex-format SimpleName, ``-`` is a valid SimpleNameChar. We must
    not reject these or Phase 2 dies on real-world Google Play Services bytecode.
    """
    cls = "Lcom/google/android/gms/internal/auth-api-phone/zzj;"
    r = MethodRecord(
        apk_sha="0" * 64,
        class_name=cls,
        method_sig="<init>()V",
        full_id=f"{cls}-><init>()V",
        instructions=[],
        n_instr=0,
    )
    assert r.class_name == cls


@pytest.mark.parametrize(
    "bad_cls",
    [
        "Lcom/foo;bar;",
        "Lcom/foo<bar>;",
        "Lcom/foo[bar];",
        "Lcom/foo bar;",
    ],
)
def test_class_name_rejects_reserved_chars(bad_cls: str):
    with pytest.raises(ValidationError):
        MethodRecord(
            apk_sha="0" * 64,
            class_name=bad_cls,
            method_sig="<init>()V",
            full_id=f"{bad_cls}-><init>()V",
            instructions=[],
            n_instr=0,
        )


def test_full_id_must_match_class_and_sig():
    with pytest.raises(ValidationError):
        MethodRecord(
            apk_sha="0" * 64,
            class_name="Ljava/lang/Object;",
            method_sig="<init>()V",
            full_id="Ljava/lang/Object;->wrong()V",
            instructions=[],
            n_instr=0,
        )


def test_n_instr_must_match_instructions_len():
    with pytest.raises(ValidationError):
        MethodRecord(
            apk_sha="0" * 64,
            class_name="Ljava/lang/Object;",
            method_sig="<init>()V",
            full_id="Ljava/lang/Object;-><init>()V",
            instructions=["a", "b"],
            n_instr=1,
        )
