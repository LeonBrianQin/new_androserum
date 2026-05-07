"""Phase 0-2: APK download / disassembly / method extraction / SuSi tagging."""

from androserum.data.method_extractor import (
    extract_api_calls_from_instructions,
    extract_methods,
    is_trivial_filtered,
    slash_class_to_descriptor,
)
from androserum.data.method_parquet import read_methods_parquet, write_methods_parquet
from androserum.data.schema import MethodRecord, make_full_id
from androserum.data.susi_index import SusiIndex, build_susi_index, dalvik_api_from_soot_line
from androserum.data.susi_tagger import tag_method_susi

__all__ = [
    "MethodRecord",
    "SusiIndex",
    "build_susi_index",
    "dalvik_api_from_soot_line",
    "extract_api_calls_from_instructions",
    "extract_methods",
    "is_trivial_filtered",
    "make_full_id",
    "read_methods_parquet",
    "slash_class_to_descriptor",
    "tag_method_susi",
    "write_methods_parquet",
]
