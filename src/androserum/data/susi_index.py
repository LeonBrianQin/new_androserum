"""Build a Dalvik API → SuSi category lookup from SuSi ``SourcesSinks`` text lists.

Upstream layout (SuSi ``develop``)::

    SourceSinkLists/Android 4.2/SourcesSinks/Ouput_CatSources_v0_9.txt
    SourceSinkLists/Android 4.2/SourcesSinks/Ouput_CatSinks_v0_9.txt

Each data line looks like::

    <android.telephony.TelephonyManager: java.lang.String getDeviceId()> (UNIQUE_IDENTIFIER)
    <...> android.permission.READ_PHONE_STATE (UNIQUE_IDENTIFIER)

We normalize signatures to static-invoke style ids
``Landroid/...;->name(proto)ret`` so they match :func:`extract_api_calls_from_instructions`.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Mapping

__all__ = [
    "SusiIndex",
    "build_susi_index",
    "dalvik_api_from_soot_line",
    "parse_susi_data_line",
]

_PRIM: dict[str, str] = {
    "void": "V",
    "boolean": "Z",
    "byte": "B",
    "short": "S",
    "char": "C",
    "int": "I",
    "long": "J",
    "float": "F",
    "double": "D",
}

_CATEGORY_TAIL_RE = re.compile(r"\(([A-Z0-9_]+)\)\s*$")


def java_type_to_descriptor(java_t: str) -> str:
    """Map a Java type spell-out to a JVM field descriptor fragment."""
    t = java_t.strip()
    if t.endswith("[]"):
        return "[" + java_type_to_descriptor(t[:-2].strip())
    if t in _PRIM:
        return _PRIM[t]
    return "L" + t.replace(".", "/") + ";"


def _split_params(params: str) -> list[str]:
    """Split a Soot param list on commas, respecting simple ``<...>`` nesting."""
    p = params.strip()
    if not p:
        return []
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in p:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return [x for x in parts if x]


def parse_soot_signature(sig: str) -> tuple[str, str, str, str] | None:
    """Parse ``<fqcn: ret name(params)>`` into components."""
    s = sig.strip()
    if not (s.startswith("<") and s.endswith(">")):
        return None
    inner = s[1:-1]
    if ":" not in inner:
        return None
    colon = inner.index(":")
    fqcn = inner[:colon].strip()
    rhs = inner[colon + 1 :].strip()
    if "(" not in rhs or ")" not in rhs:
        return None
    open_p = rhs.index("(")
    close_p = rhs.rindex(")")
    params = rhs[open_p + 1 : close_p]
    pre = rhs[:open_p].strip()
    bits = pre.rsplit(None, 1)
    if len(bits) != 2:
        return None
    ret_t, name = bits
    return fqcn, ret_t, name, params


def dalvik_api_from_soot_line(sig: str) -> str | None:
    """``<pkg.Cls: Ret name(Args)>`` → ``Lpkg/Cls;->name(proto)Ret``."""
    parsed = parse_soot_signature(sig)
    if parsed is None:
        return None
    fqcn, ret_t, name, params = parsed
    class_d = java_type_to_descriptor(fqcn)
    plist = _split_params(params)
    proto = "".join(java_type_to_descriptor(x) for x in plist)
    ret_d = java_type_to_descriptor(ret_t)
    return f"{class_d}->{name}({proto}){ret_d}"


def parse_susi_data_line(line: str) -> tuple[str, str] | None:
    """Return ``(soot_sig, category)`` or None if not a SuSi data row."""
    line = line.strip()
    if not line.startswith("<"):
        return None
    m = _CATEGORY_TAIL_RE.search(line)
    if not m:
        return None
    cat = m.group(1)
    head = line[: m.start()].rstrip()
    if "<" not in head:
        return None
    lo = head.index("<")
    hi = head.rindex(">")
    if lo >= hi:
        return None
    return head[lo : hi + 1], cat


def _ingest_susi_text(text: str, sink: DefaultDict[str, set[str]]) -> int:
    n = 0
    for raw in text.splitlines():
        parsed = parse_susi_data_line(raw)
        if parsed is None:
            continue
        soot_sig, cat = parsed
        dalvik = dalvik_api_from_soot_line(soot_sig)
        if dalvik is None:
            continue
        sink[dalvik].add(cat)
        n += 1
    return n


def build_susi_index(paths: list[str | Path]) -> "SusiIndex":
    """Merge one or more SuSi ``Ouput_Cat*.txt`` files into a single index."""
    acc: DefaultDict[str, set[str]] = defaultdict(set)
    for p in paths:
        path = Path(p)
        if not path.is_file():
            raise FileNotFoundError(f"Susi list not found: {path}")
        _ingest_susi_text(path.read_text(encoding="utf-8", errors="replace"), acc)
    frozen = {k: frozenset(v) for k, v in acc.items()}
    return SusiIndex(frozen)


class SusiIndex:
    """Immutable map Dalvik invoke target → frozenset of SuSi category labels."""

    __slots__ = ("_m",)

    def __init__(self, mapping: Mapping[str, frozenset[str]]) -> None:
        self._m = dict(mapping)

    def categories_for_api(self, dalvik_api: str) -> frozenset[str]:
        return self._m.get(dalvik_api, frozenset())

    def __len__(self) -> int:
        return len(self._m)

    def __contains__(self, dalvik_api: str) -> bool:
        return dalvik_api in self._m
