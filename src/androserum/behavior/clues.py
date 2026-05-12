"""Phase 0.5: lightweight symbolic clue extraction from method rows."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from androserum.behavior.schema import MethodClue, MethodClueResult

__all__ = [
    "extract_method_clues",
]


_URL_RE = re.compile(r"https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+")
_FILE_RE = re.compile(r"(?:/[\w.\-]+){2,}")
_CONST_STRING_RE = re.compile(r'const-string(?:/jumbo)?\s+[vp0-9,\s]+,\s+"([^"]+)"')


def _flatten_listish(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return [str(x).strip() for x in value.tolist() if str(x).strip()]
    except Exception:
        pass
    s = str(value).strip()
    return [s] if s else []


def _extract_strings(instructions: list[str]) -> list[str]:
    out: list[str] = []
    for line in instructions:
        m = _CONST_STRING_RE.search(line)
        if m:
            out.append(m.group(1))
    return out


def _score_clue(
    *,
    has_network_api: bool,
    has_file_api: bool,
    has_reflection_api: bool,
    has_db_api: bool,
    has_location_api: bool,
    has_identifier_api: bool,
    has_log_api: bool,
    url_like_count: int,
    file_like_count: int,
    string_count: int,
    has_stringbuilder_pattern: bool,
) -> tuple[float, dict[str, float]]:
    score = 0.0
    breakdown: dict[str, float] = {}

    def add(name: str, value: float) -> None:
        nonlocal score
        score += value
        breakdown[name] = value

    if has_network_api:
        add("network_api", 0.35)
    if has_file_api:
        add("file_api", 0.30)
    if has_reflection_api:
        add("reflection_api", 0.20)
    if has_db_api:
        add("db_api", 0.20)
    if has_location_api:
        add("location_api", 0.20)
    if has_identifier_api:
        add("identifier_api", 0.20)
    if has_log_api:
        add("log_api", 0.08)
    if url_like_count:
        add("url_strings", min(0.20, 0.05 * url_like_count))
    if file_like_count:
        add("file_strings", min(0.15, 0.04 * file_like_count))
    if string_count:
        add("string_literals", min(0.10, 0.01 * string_count))
    if has_stringbuilder_pattern:
        add("stringbuilder_penalty", -0.08)

    return score, breakdown


def extract_method_clues(
    *,
    apk_sha: str,
    methods_dir: str | Path,
) -> MethodClueResult:
    path = Path(methods_dir) / f"{apk_sha}.parquet"
    df = pd.read_parquet(path, engine="pyarrow")

    clues: list[MethodClue] = []
    for rec in df.to_dict("records"):
        api_calls = _flatten_listish(rec.get("api_calls", []))
        susi_cats = _flatten_listish(rec.get("susi_cats", []))
        instructions = _flatten_listish(rec.get("instructions", []))
        strings = _extract_strings(instructions)
        url_like = [s for s in strings if _URL_RE.search(s)]
        file_like = [s for s in strings if _FILE_RE.search(s)]

        api_text = " ".join(api_calls)
        has_network_api = any(
            x in api_text
            for x in [
                "HttpURLConnection",
                "URLConnection",
                "Socket",
                "URL;->",
                "openConnection",
                "setRequestMethod",
            ]
        ) or ("NETWORK" in susi_cats or "NETWORK_INFORMATION" in susi_cats)
        has_file_api = any(
            x in api_text
            for x in [
                "File;->",
                "InputStream",
                "OutputStream",
                "FileInputStream",
                "FileOutputStream",
                "IOUtils",
            ]
        ) or ("FILE" in susi_cats)
        has_reflection_api = any(
            x in api_text
            for x in [
                "Method;->invoke",
                "Class;->forName",
                "Proxy;->newProxyInstance",
                "getDeclaredMethod",
            ]
        )
        has_db_api = ("DATABASE_INFORMATION" in susi_cats) or any(
            x in api_text for x in ["SQLiteDatabase", "Cursor", "query("]
        )
        has_location_api = ("LOCATION_INFORMATION" in susi_cats) or any(
            x in api_text for x in ["LocationManager", "getLastKnownLocation", "requestLocationUpdates"]
        )
        has_identifier_api = ("UNIQUE_IDENTIFIER" in susi_cats) or any(
            x in api_text for x in ["getDeviceId", "getLine1Number", "getSubscriberId", "TelephonyManager"]
        )
        has_log_api = ("LOG" in susi_cats) or any(
            x in api_text for x in ["Log;->", "Logger", "StringBuilder"]
        )
        has_stringbuilder_pattern = any("StringBuilder->" in api for api in api_calls)

        clue_score, breakdown = _score_clue(
            has_network_api=has_network_api,
            has_file_api=has_file_api,
            has_reflection_api=has_reflection_api,
            has_db_api=has_db_api,
            has_location_api=has_location_api,
            has_identifier_api=has_identifier_api,
            has_log_api=has_log_api,
            url_like_count=len(url_like),
            file_like_count=len(file_like),
            string_count=len(strings),
            has_stringbuilder_pattern=has_stringbuilder_pattern,
        )

        clues.append(
            MethodClue(
                apk_sha=str(rec["apk_sha"]),
                full_id=str(rec["full_id"]),
                class_name=str(rec["class_name"]),
                method_sig=str(rec["method_sig"]),
                susi_cats=susi_cats,
                susi_dominant_cat=rec.get("susi_dominant_cat"),
                has_network_api=has_network_api,
                has_file_api=has_file_api,
                has_reflection_api=has_reflection_api,
                has_db_api=has_db_api,
                has_location_api=has_location_api,
                has_identifier_api=has_identifier_api,
                has_log_api=has_log_api,
                has_stringbuilder_pattern=has_stringbuilder_pattern,
                string_literals=strings[:20],
                url_like_strings=url_like[:10],
                file_like_strings=file_like[:10],
                clue_score=clue_score,
                clue_breakdown=breakdown,
            )
        )

    stats = {
        "n_methods": len(clues),
        "n_network_like": sum(1 for c in clues if c.has_network_api),
        "n_file_like": sum(1 for c in clues if c.has_file_api),
        "n_reflection_like": sum(1 for c in clues if c.has_reflection_api),
        "n_db_like": sum(1 for c in clues if c.has_db_api),
        "n_location_like": sum(1 for c in clues if c.has_location_api),
        "n_identifier_like": sum(1 for c in clues if c.has_identifier_api),
        "n_url_strings": sum(1 for c in clues if c.url_like_strings),
        "n_file_strings": sum(1 for c in clues if c.file_like_strings),
    }
    return MethodClueResult(apk_sha=apk_sha, clues=clues, stats=stats)

