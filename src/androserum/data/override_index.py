"""Extract override metadata for Phase 4 signal E (same override target).

Signal E definition
-------------------
Two methods are positives if they override the *same* parent/interface method.

We materialize this as a sidecar table per APK:

    data/overrides/<SHA>.parquet

Each row stores the method ``full_id`` plus zero or more ``override_keys`` such
as ``Ljava/lang/Runnable;->run()V`` or
``Landroid/view/View$OnClickListener;->onClick(Landroid/view/View;)V``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
import pandas as pd
from androguard.misc import AnalyzeAPK

from androserum.data.schema import make_full_id

__all__ = [
    "MethodOverrideRecord",
    "build_override_records_for_apk",
    "write_override_parquet",
    "read_override_parquet",
]


def _method_sig(name: str, descriptor: str) -> str:
    return f"{name}{descriptor}"


def _class_method_key(class_name: str, name: str, descriptor: str) -> str:
    return make_full_id(class_name, _method_sig(name, descriptor))


@dataclass(frozen=True)
class MethodOverrideRecord:
    apk_sha: str
    full_id: str
    override_keys: list[str]


def _is_override_candidate(method_name: str, access_flags: str) -> bool:
    """Cheap, conservative filter for methods worth considering for signal E.

    We explicitly reject methods that are almost certainly not useful override
    positives for functional clustering:

      * constructors / class initializers
      * private methods
      * static methods
    """
    if method_name in {"<init>", "<clinit>"}:
        return False
    flags = (access_flags or "").lower()
    if "private" in flags:
        return False
    if "static" in flags:
        return False
    return True


def _build_class_maps(dx) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
    """Return internal-method map and class hierarchy maps.

    Returns:
        method_sigs_by_class: ``class -> {name+descriptor}``
        super_by_class: ``class -> {super}`` (0/1 entry for dex class hierarchy)
        interfaces_by_class: ``class -> {iface1, iface2, ...}``
    """
    method_sigs_by_class: dict[str, set[str]] = {}
    super_by_class: dict[str, set[str]] = {}
    interfaces_by_class: dict[str, set[str]] = {}

    for cls in dx.get_internal_classes():
        vm = cls.get_vm_class()
        class_name = vm.get_name()
        method_sigs_by_class[class_name] = {
            _method_sig(m.get_name(), m.get_descriptor())
            for m in vm.get_methods()
            if _is_override_candidate(m.get_name(), m.get_access_flags_string())
        }
        super_name = vm.get_superclassname()
        super_by_class[class_name] = {super_name} if super_name else set()
        interfaces_by_class[class_name] = set(vm.get_interfaces() or [])

    return method_sigs_by_class, super_by_class, interfaces_by_class


def _all_ancestor_types(
    class_name: str,
    super_by_class: dict[str, set[str]],
    interfaces_by_class: dict[str, set[str]],
) -> list[str]:
    """Breadth-first traversal over super + interface edges.

    For internal classes we follow internal hierarchy links recursively.
    For external classes we keep the first seen external type and stop, because
    Androguard's external class objects do not expose further super/interface
    traversal in a stable way.
    """
    out: list[str] = []
    seen: set[str] = set()
    q: deque[str] = deque()

    for s in sorted(super_by_class.get(class_name, set())):
        if s:
            q.append(s)
    for i in sorted(interfaces_by_class.get(class_name, set())):
        if i:
            q.append(i)

    while q:
        cur = q.popleft()
        if cur in seen:
            continue
        seen.add(cur)
        out.append(cur)

        if cur in super_by_class:
            for nxt in sorted(super_by_class[cur]):
                if nxt and nxt not in seen:
                    q.append(nxt)
        if cur in interfaces_by_class:
            for nxt in sorted(interfaces_by_class[cur]):
                if nxt and nxt not in seen:
                    q.append(nxt)
    return out


def build_override_records_for_apk(apk_path: str | Path, apk_sha: str) -> list[MethodOverrideRecord]:
    """Analyze one APK and return override sidecar rows."""
    logger.remove()  # avoid verbose androguard debug spam in normal runs
    _, _, dx = AnalyzeAPK(str(apk_path))

    method_sigs_by_class, super_by_class, interfaces_by_class = _build_class_maps(dx)

    out: list[MethodOverrideRecord] = []
    for class_name, sigs in method_sigs_by_class.items():
        ancestors = _all_ancestor_types(class_name, super_by_class, interfaces_by_class)
        for sig in sorted(sigs):
            override_keys: list[str] = []
            for anc in ancestors:
                target_sigs = method_sigs_by_class.get(anc)
                if target_sigs is not None:
                    if sig in target_sigs:
                        override_keys.append(make_full_id(anc, sig))
                    continue

                # External ancestor: ask Androguard analysis for method list.
                ca = dx.get_class_analysis(anc)
                if ca is None:
                    continue
                try:
                    ext_methods = list(ca.get_methods())
                except Exception:
                    continue
                for m in ext_methods:
                    if m.name in {"<init>", "<clinit>"}:
                        continue
                    if _method_sig(m.name, m.descriptor) == sig:
                        override_keys.append(make_full_id(anc, sig))
                        break

            name, descriptor = sig.split("(", 1)
            full_id = _class_method_key(class_name, name, f"({descriptor}")
            out.append(
                MethodOverrideRecord(
                    apk_sha=apk_sha.upper(),
                    full_id=full_id,
                    override_keys=sorted(set(override_keys)),
                )
            )
    return out


def write_override_parquet(rows: list[MethodOverrideRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "apk_sha": [r.apk_sha for r in rows],
            "full_id": [r.full_id for r in rows],
            "override_keys": [r.override_keys for r in rows],
        }
    )
    df.to_parquet(path, index=False, engine="pyarrow")


def read_override_parquet(path: str | Path) -> dict[str, list[str]]:
    """Return ``full_id -> override_keys`` mapping."""
    p = Path(path)
    if not p.is_file():
        return {}
    df = pd.read_parquet(p, engine="pyarrow")
    out: dict[str, list[str]] = {}
    for rec in df.to_dict("records"):
        full_id = str(rec["full_id"])
        keys = rec.get("override_keys")
        if keys is None:
            out[full_id] = []
        elif isinstance(keys, list):
            out[full_id] = [str(x) for x in keys]
        else:
            out[full_id] = list(keys)
    return out
