"""Phase 0-2: APK download / disassembly / method extraction / SuSi tagging."""

from androserum.data.library_index import (
    MethodLibraryRecord,
    build_exact_full_id_library_sidecars,
    read_library_parquet,
    write_library_parquet,
)
from androserum.data.method_extractor import (
    extract_api_calls_from_instructions,
    extract_methods,
    is_trivial_filtered,
    slash_class_to_descriptor,
)
from androserum.data.method_parquet import read_methods_parquet, write_methods_parquet
from androserum.data.override_index import (
    MethodOverrideRecord,
    build_override_records_for_apk,
    read_override_parquet,
    write_override_parquet,
)
from androserum.data.schema import MethodRecord, make_full_id
from androserum.data.susi_index import SusiIndex, build_susi_index, dalvik_api_from_soot_line
from androserum.data.susi_tagger import tag_method_susi

__all__ = [
    "MethodLibraryRecord",
    "MethodRecord",
    "MethodOverrideRecord",
    "SusiIndex",
    "build_exact_full_id_library_sidecars",
    "build_override_records_for_apk",
    "build_susi_index",
    "dalvik_api_from_soot_line",
    "extract_api_calls_from_instructions",
    "extract_methods",
    "is_trivial_filtered",
    "make_full_id",
    "read_library_parquet",
    "read_override_parquet",
    "read_methods_parquet",
    "slash_class_to_descriptor",
    "tag_method_susi",
    "write_library_parquet",
    "write_override_parquet",
    "write_methods_parquet",
]
