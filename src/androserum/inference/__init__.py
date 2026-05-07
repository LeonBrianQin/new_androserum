"""Phase 3: frozen-encoder method embeddings (baseline)."""

from androserum.inference.frozen_encode import (
    encode_methods_parquet_file,
    instructions_to_cls_batch,
)

__all__ = [
    "encode_methods_parquet_file",
    "instructions_to_cls_batch",
]
