"""Unit tests for Phase 5 FCG alignment helpers."""

from __future__ import annotations

import networkx as nx

from androserum.data.schema import MethodRecord
from androserum.fcg import align_call_graph_to_method_rows, method_to_full_id


class _FakeMethod:
    def __init__(self, class_name: str, name: str, descriptor: str):
        self._class_name = class_name
        self._name = name
        self._descriptor = descriptor

    def get_class_name(self) -> str:
        return self._class_name

    def get_name(self) -> str:
        return self._name

    def get_descriptor(self) -> str:
        return self._descriptor


def _row(full_id: str, *, filtered: bool = False) -> MethodRecord:
    class_name, method_sig = full_id.split("->", 1)
    return MethodRecord(
        apk_sha="A" * 64,
        class_name=class_name,
        method_sig=method_sig,
        full_id=full_id,
        instructions=[],
        n_instr=0,
        filtered=filtered,
    )


def test_method_to_full_id_matches_phase2_schema():
    m = _FakeMethod("Lcom/example/Foo;", "bar", "(I)V")
    assert method_to_full_id(m) == "Lcom/example/Foo;->bar(I)V"


def test_align_call_graph_to_method_rows_keeps_method_row_order():
    method_a = _FakeMethod("Lcom/example/A;", "alpha", "()V")
    method_b = _FakeMethod("Lcom/example/B;", "beta", "()V")
    external = _FakeMethod("Ljava/lang/Runnable;", "run", "()V")

    graph = nx.DiGraph()
    graph.add_node(
        method_a,
        external=False,
        entrypoint=True,
    )
    graph.add_node(
        method_b,
        external=False,
        entrypoint=False,
    )
    graph.add_node(
        external,
        external=True,
        entrypoint=False,
    )
    graph.add_edge(method_a, method_b)
    graph.add_edge(method_a, external)
    graph.add_edge(external, method_b)

    rows = [
        _row("Lcom/example/A;->alpha()V"),
        _row("Lcom/example/B;->beta()V", filtered=True),
        _row("Lcom/example/C;->gamma()V"),
    ]

    aligned, internal_edges, boundary_edges, summary = align_call_graph_to_method_rows(
        "A" * 64,
        rows,
        graph,
        include_boundary_edges=True,
    )

    assert [r.full_id for r in aligned] == [r.full_id for r in rows]

    assert aligned[0].graph_present is True
    assert aligned[0].entrypoint is True
    assert aligned[0].internal_out_degree == 1
    assert aligned[0].external_out_degree == 1

    assert aligned[1].graph_present is True
    assert aligned[1].filtered is True
    assert aligned[1].internal_in_degree == 1
    assert aligned[1].external_in_degree == 1

    assert aligned[2].graph_present is False
    assert aligned[2].internal_in_degree == 0
    assert aligned[2].internal_out_degree == 0

    assert len(internal_edges) == 1
    assert internal_edges[0].src_idx == 0
    assert internal_edges[0].dst_idx == 1

    assert len(boundary_edges) == 2
    assert {e.direction for e in boundary_edges} == {"in", "out"}
    assert all(e.other_external for e in boundary_edges)

    assert summary.methods_rows == 3
    assert summary.graph_internal_nodes_total == 2
    assert summary.graph_external_nodes_total == 1
    assert summary.internal_edges_count == 1
    assert summary.boundary_edges_count == 2
    assert summary.missing_graph_nodes_count == 1
    assert summary.missing_graph_nodes_sample == ["Lcom/example/C;->gamma()V"]
