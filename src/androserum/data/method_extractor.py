"""Turn DexBERT ``processed/<sha>.txt`` files into ``MethodRecord`` rows.

The text format (from ``instruction_generator`` / ``apk_processor``) is::

    <optional first line: 64-hex>.txt
    ClassName: com/foo/Bar          # slash-separated internal name
    MethodName: m(II)V              # dex proto tail only
    <smali instruction lines>
    <blank line ends method>

:class:`.schema.MethodRecord` fields ``class_name`` / ``full_id`` use
**type-descriptor** form (``Lcom/foo/Bar;``) so they match Androguard
node labels in Phase 5.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional

from androserum.data.schema import MethodRecord, make_full_id

__all__ = [
    "extract_methods",
    "extract_api_calls_from_instructions",
    "slash_class_to_descriptor",
    "is_trivial_filtered",
]

# ``invoke-*`` callee spans from ``Lpkg/Cls;`` through return type (to EOL).
_INVOKE_API_RE = re.compile(
    r"invoke-(?:virtual|direct|static|interface|super)(?:/range)?\s+"
    r"\{[^}]*\},\s+"
    r"(L[\w/$]+;->\S+)$"
)

_SHA_TXT_HEADER = re.compile(r"^[0-9A-Fa-f]{64}\.txt\s*$")


def slash_class_to_descriptor(slash: str) -> str:
    """``com/foo/Bar`` / ``com/foo/Bar$Inner`` → ``Lcom/foo/Bar;``."""
    s = slash.strip()
    if not s:
        raise ValueError("empty class slash path")
    s = s.lstrip("/")
    return f"L{s.replace('.', '/')};"


def extract_api_calls_from_instructions(instructions: Iterable[str]) -> List[str]:
    """Collect invoke targets in source order without duplicates."""
    seen: dict[str, None] = {}
    out: list[str] = []
    for line in instructions:
        m = _INVOKE_API_RE.search(line)
        if not m:
            continue
        callee = m.group(1)
        if callee not in seen:
            seen[callee] = None
            out.append(callee)
    return out


def method_base_name(method_sig: str) -> str:
    """``<init>(...)V`` → ``<init>``."""
    method_sig = method_sig.strip()
    idx = method_sig.find("(")
    if idx == -1:
        return method_sig
    return method_sig[:idx]


def is_trivial_filtered(
    class_descriptor: str,
    method_sig: str,
    n_instr: int,
) -> bool:
    """PROJECT_HANDOFF.md §7.4 trivial / noise filters."""
    if n_instr < 5:
        return True
    # Descriptor-level checks so both ``L.../R$...;`` and ``...BuildConfig`` hit.
    if "R$" in class_descriptor or "BuildConfig" in class_descriptor:
        return True
    base = method_base_name(method_sig)
    if base.startswith("access$"):
        return True
    if base.startswith("lambda$"):
        return True
    if base.startswith("$values"):
        return True
    if base.startswith("<clinit>"):
        return True
    return False


def _apk_sha_from_txt_path(path: Path) -> str:
    stem = path.stem.strip()
    if len(stem) == 64 and all(c in "0123456789abcdefABCDEF" for c in stem):
        return stem.upper()
    raise ValueError(
        f"cannot derive apk_sha from path {path!r} (stem must be 64 hex chars)"
    )


def _apk_sha_from_first_line(first_line: str) -> Optional[str]:
    s = first_line.strip()
    if _SHA_TXT_HEADER.match(s):
        return s[:-4].strip().upper()  # drop ".txt"
    return None


def extract_methods(
    txt_path: str | Path,
    *,
    apk_sha: Optional[str] = None,
    encoding: str = "utf-8",
) -> List[MethodRecord]:
    """Parse one ``processed`` text file; returns ordered ``MethodRecord`` list."""
    path = Path(txt_path)
    text = path.read_text(encoding=encoding, errors="replace")
    lines = text.splitlines()

    derived_sha = apk_sha
    if derived_sha is None:
        try:
            derived_sha = _apk_sha_from_txt_path(path)
        except ValueError:
            if lines:
                derived_sha = _apk_sha_from_first_line(lines[0])
            if not derived_sha:
                raise ValueError(
                    f"apk_sha not provided and cannot be inferred from {path}"
                ) from None

    out: list[MethodRecord] = []
    class_slash: Optional[str] = None
    method_sig: Optional[str] = None
    instructions: list[str] = []

    def flush() -> None:
        nonlocal method_sig, instructions
        if method_sig is None or class_slash is None:
            instructions = []
            method_sig = None
            return
        class_desc = slash_class_to_descriptor(class_slash)
        full_id = make_full_id(class_desc, method_sig)
        n = len(instructions)
        apis = extract_api_calls_from_instructions(instructions)
        filtered = is_trivial_filtered(class_desc, method_sig, n)
        out.append(
            MethodRecord(
                apk_sha=derived_sha,
                class_name=class_desc,
                method_sig=method_sig,
                full_id=full_id,
                instructions=list(instructions),
                n_instr=n,
                api_calls=apis,
                susi_cats=[],
                susi_dominant_cat=None,
                filtered=filtered,
            )
        )
        instructions = []
        method_sig = None

    for idx, raw in enumerate(lines):
        line = raw.strip()

        if idx == 0 and _SHA_TXT_HEADER.match(line):
            continue

        if line.startswith("ClassName:"):
            flush()
            class_slash = line[len("ClassName:") :].strip()
            method_sig = None
            instructions = []
            continue

        if line.startswith("MethodName:"):
            flush()
            method_sig = line[len("MethodName:") :].strip()
            instructions = []
            continue

        if not line:
            flush()
            continue

        if method_sig is not None and class_slash is not None:
            instructions.append(line)

    flush()
    return out
