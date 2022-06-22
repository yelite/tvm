import pytest

from tvm.script import tir as T
import tvm
from tvm import te
from tvm.runtime import ObjectPath

_as_tvm_script = tvm.get_global_func("experiment.AsTVMScript")

ROOT = ObjectPath.root()


def to_tvmscript(node, path_to_highlight=None, num_context_lines=-1) -> str:
    return _as_tvm_script(node, {"tir": "T"}, 4, path_to_highlight, num_context_lines)


def format_script(s: str) -> str:
    """
    Remove leading and trailing blank lines, and make the minimum idention 0
    """
    s = s.strip("\n")
    lines = [line for line in s.splitlines()]
    # TODO expand this line
    indent_level = min(
        len(line) - len(line.lstrip(" ")) for line in lines if len(line.lstrip(" ")) > 0
    )
    return "\n".join(line[indent_level:] for line in lines)


@T.prim_func
def main(A: T.Buffer[(8,), "float32"], B: T.Buffer[(8,), "float32"]):
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    for i in range(8):
        with T.block("B"):
            vi = T.axis.spatial(8, i)
            B[vi] = A[vi] + 1.0


def test_roundtrippable_basic():
    expected = """
        @T.prim_func
        def main(A: T.Buffer(shape=(8,)), B: T.Buffer(shape=(8,))) -> None:
            for i in T.serial(8):
                with T.block("B"):
                    vi = T.axis.spatial(8, i)
                    T.reads(A[vi])
                    T.writes(B[vi])
                    B[vi] = A[vi] + T.float32(1)
    """
    assert to_tvmscript(main) == format_script(expected)


def test_highlight_expr_int_imm():
    int_imm = te.const(77)
    assert to_tvmscript(int_imm, path_to_highlight=ROOT) == format_script(
        """
        77
        ^^
        """
    )


def make_params(*cases):
    assert len(cases) % 2 == 0
    return [
        pytest.param(path, expected_text, id="path_{}".format(i))
        for i, (path, expected_text) in enumerate(zip(cases[::2], cases[1::2]))
    ]


@pytest.mark.parametrize(
    "path, expected_text",
    make_params(
        ROOT.attr("a"),
        """
            77 + 6
            ^^
        """,
        ROOT.attr("b"),
        """
            77 + 6
                 ^
        """,
        ROOT,
        """
            77 + 6
            ^^^^^^
        """,
    ),
)
def test_highlight_expr_add(path, expected_text):
    e = tvm.tir.Add(te.const(77), te.const(6))
    assert to_tvmscript(e, path_to_highlight=path) == format_script(expected_text)


@pytest.mark.parametrize(
    "path, expected_text",
    make_params(
        ROOT.attr("value"),
        """
            T.cast(77, "float32")
                   ^^
        """,
        ROOT.attr("dtype"),
        """
            T.cast(77, "float32")
                       ^^^^^^^^^
        """,
        ROOT,
        """
            T.cast(77, "float32")
            ^^^^^^^^^^^^^^^^^^^^^
        """,
    ),
)
def test_highlight_expr_cast(path, expected_text):
    e = tvm.tir.Cast("float32", te.const(77))
    assert to_tvmscript(e, path_to_highlight=path) == format_script(expected_text)


@pytest.mark.parametrize(
    "path, expected_text",
    make_params(
        ROOT.attr("condition"),
        """
            T.Select(True, 555, 77)
                     ^^^^
        """,
        ROOT.attr("true_value"),
        """
            T.Select(True, 555, 77)
                           ^^^
        """,
        ROOT.attr("false_value"),
        """
            T.Select(True, 555, 77)
                                ^^
        """,
        ROOT,
        """
            T.Select(True, 555, 77)
            ^^^^^^^^^^^^^^^^^^^^^^^
        """,
    ),
)
def test_highlight_expr_select(path, expected_text):
    e = tvm.tir.Select(te.const(True), te.const(555), te.const(77))
    assert to_tvmscript(e, path_to_highlight=path) == format_script(expected_text)


@pytest.mark.parametrize(
    "path, expected_text",
    make_params(
        ROOT.attr("value"),
        """
            77
            ^^
        """,
        ROOT,
        """
            77
            ^^
        """,
    ),
)
def test_hightlight_stmt_evaluate(path, expected_text):
    s = tvm.tir.Evaluate(te.const(77))
    assert to_tvmscript(s, path_to_highlight=path) == format_script(expected_text)


@pytest.mark.parametrize(
    "path, expected_text",
    make_params(
        ROOT.attr("seq").array_index(0),
        """
            555
            ^^^
            77
        """,
        ROOT.attr("seq").missing_array_element(2),
        """
            555
            77
              ^
        """,
        ROOT.attr("seq"),
        """
            555
            ^^^
            77
            ^^
        """,
    ),
)
def test_hightlight_stmt_seq(path, expected_text):
    s = tvm.tir.SeqStmt([tvm.tir.Evaluate(te.const(555)), tvm.tir.Evaluate(te.const(77))])
    assert to_tvmscript(s, path_to_highlight=path) == format_script(expected_text)


@pytest.mark.parametrize(
    "path, expected_text",
    make_params(
        ROOT.attr("seq").array_index(0).attr("seq").array_index(1),
        """
            555
            77
            ^^
            8888
        """,
        ROOT.attr("seq").array_index(0).attr("seq").missing_array_element(2),
        """
            555
            77
              ^
            8888
        """,
    ),
)
def test_hightlight_stmt_seq_nested(path, expected_text):
    s = tvm.tir.SeqStmt(
        [
            tvm.tir.SeqStmt(
                [
                    tvm.tir.Evaluate(te.const(555)),
                    tvm.tir.Evaluate(te.const(77)),
                ]
            ),
            tvm.tir.Evaluate(te.const(8888)),
        ]
    )
    assert to_tvmscript(s, path_to_highlight=path) == format_script(expected_text)


@pytest.mark.parametrize(
    "path, expected_text",
    make_params(
        ROOT.attr("condition"),
        """
            with T.Assert(False, "hello"):
                          ^^^^^
                555
        """,
        ROOT.attr("message"),
        """
            with T.Assert(False, "hello"):
                                 ^^^^^^^
                555
        """,
        ROOT.attr("body"),
        """
            with T.Assert(False, "hello"):
                555
                ^^^
        """,
    ),
)
def test_hightlight_stmt_assert(path, expected_text):
    s = tvm.tir.AssertStmt(
        te.const(False), tvm.runtime.convert("hello"), tvm.tir.Evaluate(te.const(555))
    )
    assert to_tvmscript(s, path_to_highlight=path) == format_script(expected_text)


@pytest.mark.parametrize(
    "path, expected_text",
    make_params(
        ROOT.attr("var").attr("dtype"),
        """
            foo: T.int32 = T.var("int32")
                                 ^^^^^^^
            with T.let(foo, 555):
                foo + 77
        """,
        ROOT.attr("var").attr("type_annotation"),
        """
            foo: T.int32 = T.var("int32")
                 ^^^^^^^
            with T.let(foo, 555):
                foo + 77
        """,
        ROOT.attr("value"),
        """
            foo: T.int32 = T.var("int32")
            with T.let(foo, 555):
                            ^^^
                foo + 77
        """,
        ROOT.attr("body"),
        """
            foo: T.int32 = T.var("int32")
            with T.let(foo, 555):
                foo + 77
                ^^^^^^^^
        """,
        ROOT.attr("body").attr("value").attr("a"),
        """
            foo: T.int32 = T.var("int32")
            with T.let(foo, 555):
                foo + 77
                ^^^
        """,
    ),
)
def test_hightlight_stmt_let(path, expected_text):
    foo = te.var("foo")
    s = tvm.tir.LetStmt(foo, 555, tvm.tir.Evaluate(foo + te.const(77)))
    assert to_tvmscript(s, path_to_highlight=path) == format_script(expected_text)


@pytest.mark.parametrize(
    "path, expected_text",
    make_params(
        ROOT.attr("min"),
        """
            for xx in T.serial(10):
                               ^^
                xx + 77
        """,
        ROOT.attr("extent"),
        """
            for xx in T.serial(10):
                               ^^
                xx + 77
        """,
        ROOT.attr("kind"),
        """
            for xx in T.serial(10):
                      ^^^^^^^^
                xx + 77
        """,
        ROOT.attr("body").attr("value").attr("a"),
        """
            for xx in T.serial(10):
                xx + 77
                ^^
        """,
    ),
)
def test_highlight_stmt_for_simple_from_zero(path, expected_text):
    xx = te.var("xx")
    s = tvm.tir.For(
        xx, te.const(0), te.const(10), tvm.tir.ForKind.SERIAL, tvm.tir.Evaluate(xx + te.const(77))
    )
    assert to_tvmscript(s, path_to_highlight=path) == format_script(expected_text)


@pytest.mark.parametrize(
    "path, expected_text",
    make_params(
        ROOT.attr("seq").array_index(0),
        """
            100
            ^^^
            101
            102
            (... 7 lines skipped ...)
        """,
        ROOT.attr("seq").array_index(1),
        """
            100
            101
            ^^^
            102
            103
            (... 6 lines skipped ...)
        """,
        ROOT.attr("seq").array_index(3),
        """
            100
            101
            102
            103
            ^^^
            104
            105
            (... 4 lines skipped ...)
        """,
        ROOT.attr("seq").array_index(4),
        """
            (... 2 lines skipped ...)
            102
            103
            104
            ^^^
            105
            106
            (... 3 lines skipped ...)
        """,
        ROOT.attr("seq").array_index(6),
        """
            (... 4 lines skipped ...)
            104
            105
            106
            ^^^
            107
            108
            109
        """,
    ),
)
def test_highlight_print_two_context_lines(path, expected_text):
    s = tvm.tir.SeqStmt([tvm.tir.Evaluate(te.const(100 + i)) for i in range(10)])
    assert to_tvmscript(s, path_to_highlight=path, num_context_lines=2) == format_script(
        expected_text
    )


@pytest.mark.xfail
def test_roundtrippable_basic_fragment():
    # `B[vi] = A[vi] + 1.0`
    buffer_store_node = main.body.block.body.body.block.body

    expected = """
        B: T.Buffer[8, "float32"]
        vi: T.int32
        A: T.Buffer[8, "float32"]

        B[vi] = A[vi] + T.float32(1)
    """
    assert to_tvmscript(buffer_store_node) == format_script(expected)
