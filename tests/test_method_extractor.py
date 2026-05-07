"""Tests for ``androserum.data.method_extractor``."""

from pathlib import Path

import pytest

from androserum.data.method_extractor import (
    extract_api_calls_from_instructions,
    extract_methods,
    is_trivial_filtered,
    slash_class_to_descriptor,
)

SAMPLE_TXT = Path(
    "data/processed/"
    "0D64BB3C121E1986766505E182F16FB8DCC4188224F3094F99B9F905873DDC4A.txt"
)


def test_slash_class_to_descriptor():
    assert slash_class_to_descriptor("net/rbgrn/opengl/EglHelper") == (
        "Lnet/rbgrn/opengl/EglHelper;"
    )
    assert slash_class_to_descriptor("com.foo.Bar") == "Lcom/foo/Bar;"


def test_extract_api_calls_order_and_dedupe():
    lines = [
        "invoke-direct {p0}, Ljava/lang/Object;-><init>()V",
        "invoke-interface {v0}, Ljava/io/Closeable;->close()V",
        "invoke-direct {p0}, Ljava/lang/Object;-><init>()V",  # duplicate
    ]
    assert extract_api_calls_from_instructions(lines) == [
        "Ljava/lang/Object;-><init>()V",
        "Ljava/io/Closeable;->close()V",
    ]


def test_is_trivial_filtered_rules():
    cls = "Lcom/example/Foo;"
    assert is_trivial_filtered(cls, "run()V", 3) is True  # n_instr
    assert is_trivial_filtered("Lcom/example/R$string;", "f()V", 10) is True
    assert is_trivial_filtered("Lcom/example/BuildConfig;", "f()V", 10) is True
    assert is_trivial_filtered(cls, "access$001()V", 10) is True
    assert is_trivial_filtered(cls, "lambda$new$0()V", 10) is True
    assert is_trivial_filtered(cls, "$values()[Lcom/x/E;", 10) is True
    assert is_trivial_filtered(cls, "<clinit>()V", 10) is True
    assert is_trivial_filtered(cls, "realWork()V", 10) is False


@pytest.fixture
def tiny_processed(tmp_path: Path) -> Path:
    p = tmp_path / (
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.txt"
    )
    content = """\
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.txt
ClassName: demo/pkg/Hello
MethodName: main([Ljava/lang/String;)V
    sget-object v0, Ljava/lang/System;->out:Ljava/io/PrintStream;
    const/4 v1, 0x0
    const/4 v2, 0x0
    invoke-virtual {v0, p0}, Ljava/io/PrintStream;->println(Ljava/lang/String;)V
    return-void

"""
    p.write_text(content, encoding="utf-8")
    return p


def test_extract_methods_tiny_file(tiny_processed: Path):
    rows = extract_methods(tiny_processed)
    assert len(rows) == 1
    m = rows[0]
    assert m.class_name == "Ldemo/pkg/Hello;"
    assert m.method_sig == "main([Ljava/lang/String;)V"
    assert m.full_id == "Ldemo/pkg/Hello;->main([Ljava/lang/String;)V"
    assert m.n_instr == 5
    assert len(m.api_calls) == 1
    assert "Ljava/io/PrintStream;->println(Ljava/lang/String;)V" in m.api_calls[0]
    assert m.filtered is False


@pytest.mark.skipif(not SAMPLE_TXT.is_file(), reason="sample processed txt not present")
def test_extract_methods_real_sample_count_and_first_init():
    rows = extract_methods(SAMPLE_TXT)
    assert len(rows) > 1000
    first = rows[0]
    assert first.class_name == "Lnet/rbgrn/opengl/EglHelper;"
    assert first.method_sig.startswith("<init>(")
    assert "Ljava/lang/Object;-><init>()V" in first.api_calls
    assert first.filtered is False
