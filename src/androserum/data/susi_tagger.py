"""Annotate :class:`MethodRecord` rows with SuSi categories from :class:`SusiIndex`."""

from __future__ import annotations

from collections import Counter

from androserum.data.schema import MethodRecord
from androserum.data.susi_index import SusiIndex

__all__ = ["tag_method_susi"]


def tag_method_susi(row: MethodRecord, index: SusiIndex) -> MethodRecord:
    """Recompute ``susi_cats`` / ``susi_dominant_cat`` from ``api_calls``."""
    hits: list[str] = []
    for api in row.api_calls:
        hits.extend(sorted(index.categories_for_api(api)))
    if not hits:
        return row.model_copy(
            update={"susi_cats": [], "susi_dominant_cat": None},
        )
    distinct = sorted(frozenset(hits))
    dominant, _ = Counter(hits).most_common(1)[0]
    return row.model_copy(
        update={"susi_cats": distinct, "susi_dominant_cat": dominant},
    )
