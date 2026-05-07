"""Structured representation of one Dalvik method row (Phase 2 → parquet).

Field semantics follow ``PROJECT_HANDOFF.md`` §7.4. The ``full_id`` string
must stay character-for-character aligned with Androguard node labels in
Phase 5 (``Lcom/foo/Bar;->m(II)V``).

``class_name`` is the *type descriptor* form (``L...;``), not the slash form
(``com/foo/Bar``) emitted at the top of DexBERT ``processed/*.txt`` files ---
the method extractor is responsible for normalizing slashes → descriptors.
"""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Dalvik type descriptor for a class (internal name).
#
# Per the Dalvik dex-format spec (SimpleName), a class internal name is built
# from segments of ``[A-Za-z0-9$_\-]`` (plus some Unicode) joined by ``/``.
# Real-world APKs really do ship classes like
# ``Lcom/google/android/gms/internal/auth-api-phone/zzj;`` (R8/D8 packaging),
# so the leading-letter-only rule from JLS does NOT hold here.
#
# We deliberately do not accept ``;`` ``[`` ``<`` ``>`` ``:`` ``(`` ``)``
# (those are reserved as descriptor / signature separators). Whitespace is
# also rejected to keep ``full_id`` parseable later.
_CLASS_NAME_RE = re.compile(r"^L[A-Za-z0-9_$\-/]+;$")


def make_full_id(class_name: str, method_sig: str) -> str:
    """Join normalized ``class_name`` and ``method_sig`` into an FCG node id."""
    c = class_name.strip()
    m = method_sig.strip()
    return f"{c}->{m}"


class MethodRecord(BaseModel):
    """One method instance mined from a DexBERT-style ``processed/<sha>.txt`` file."""

    model_config = ConfigDict(extra="forbid")

    apk_sha: str = Field(
        ...,
        description=(
            "SHA-256 of the APK (64 hex chars). Input may be upper, lower, or mixed; "
            "validator canonicalizes to UPPER so rows match ``<SHA>.apk`` from our "
            "download script and AndroZoo query strings (API is case-insensitive for hex)."
        ),
    )
    class_name: str = Field(
        ...,
        description="Type descriptor, e.g. ``Lcom/foo/Bar;`` (not slash form).",
    )
    method_sig: str = Field(
        ...,
        description="Dex/proto part only, same line as ``MethodName:`` minus the prefix.",
    )
    full_id: str = Field(
        ...,
        description="FCG identity ``{class_name}->{method_sig}`` for Phase 5 alignment.",
    )
    instructions: list[str] = Field(default_factory=list)
    n_instr: int = Field(..., ge=0)
    api_calls: list[str] = Field(
        default_factory=list,
        description="Static callee ids like ``Lpkg/Cls;->m(II)V`` from invoke-* lines.",
    )
    susi_cats: list[str] = Field(
        default_factory=list,
        description="Distinct SuSi categories hit by ``api_calls`` (may be empty).",
    )
    susi_dominant_cat: Optional[str] = Field(
        None,
        description="Mode of ``susi_cats``; primary weak label for Phase 4.",
    )
    filtered: bool = Field(
        False,
        description="True if this method was marked trivial / excluded from training.",
    )

    @field_validator("apk_sha", mode="before")
    @classmethod
    def _normalize_apk_sha(cls, v: object) -> str:
        if not isinstance(v, str):
            raise TypeError("apk_sha must be str")
        # Accept any 64-hex case from lists / CSV / filenames; canonical form is UPPER
        # (matches ``androzoo.py`` out paths and keeps parquet joins simple).
        s = v.strip().upper()
        if len(s) != 64 or any(c not in "0123456789ABCDEF" for c in s):
            raise ValueError(f"apk_sha must be 64 hex chars, got {v!r}")
        return s

    @field_validator("class_name", mode="before")
    @classmethod
    def _strip_class(cls, v: object) -> str:
        if not isinstance(v, str):
            raise TypeError("class_name must be str")
        s = v.strip()
        if not _CLASS_NAME_RE.match(s):
            raise ValueError(
                f"class_name must look like Lxx/yy/ZZ; (got {s!r}); "
                "normalize from DexBERT slash form first."
            )
        return s

    @field_validator("method_sig", "full_id", mode="before")
    @classmethod
    def _strip_nonempty(cls, v: object) -> str:
        if not isinstance(v, str):
            raise TypeError("expected str")
        s = v.strip()
        if not s:
            raise ValueError("must be non-empty after strip()")
        return s

    @model_validator(mode="after")
    def _full_id_consistency(self) -> "MethodRecord":
        expected = make_full_id(self.class_name, self.method_sig)
        if self.full_id != expected:
            raise ValueError(
                f"full_id mismatch: stored {self.full_id!r} != expected {expected!r}"
            )
        return self

    @model_validator(mode="after")
    def _n_instr_matches_len(self) -> "MethodRecord":
        if self.n_instr != len(self.instructions):
            raise ValueError(
                f"n_instr={self.n_instr} != len(instructions)={len(self.instructions)}"
            )
        return self


__all__ = ["MethodRecord", "make_full_id"]
