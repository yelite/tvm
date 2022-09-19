# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import pytest

import tvm
from tvm.ir import GlobalVar, PointerType, PrimType, Range, TupleType
from tvm.script import tir as T
from tvm.script.printer import script
from tvm.tir import (
    EQ,
    GE,
    GT,
    LE,
    LT,
    NE,
    Add,
    Allocate,
    And,
    AssertStmt,
    AttrStmt,
    Block,
    BlockRealize,
    Broadcast,
    BufferLoad,
    BufferRealize,
    BufferRegion,
    BufferStore,
    Call,
    Cast,
    CommReducer,
    Div,
    Evaluate,
    FloatImm,
    FloorDiv,
    FloorMod,
    For,
    ForKind,
    IfThenElse,
    IntImm,
    IterVar,
    Let,
    LetStmt,
    MatchBufferRegion,
    Max,
    Min,
    Mod,
    Mul,
    Not,
    Or,
    Prefetch,
    Ramp,
    Reduce,
    Select,
    SeqStmt,
    Shuffle,
    SizeVar,
    StringImm,
    Sub,
    Var,
    While,
    decl_buffer,
)


def format_script(s: str) -> str:
    """
    Remove leading and trailing blank lines, and make the minimum idention 0
    """
    s = s.strip("\n")

    non_empty_lines = [line for line in s.splitlines() if line and not line.isspace()]
    if not non_empty_lines:
        # no actual content
        return "\n"

    line_indents = [len(line) - len(line.lstrip(" ")) for line in non_empty_lines]
    spaces_to_remove = min(line_indents)

    cleaned_lines = "\n".join(line[spaces_to_remove:] for line in s.splitlines())
    if not cleaned_lines.endswith("\n"):
        cleaned_lines += "\n"
    return cleaned_lines


def as_tir_script(node):
    return script(node, "tir", {"tir": "T"})


@pytest.mark.parametrize(
    "ty, expected",
    [
        pytest.param(
            PrimType("int8"),
            """
            T.int8
            """,
            id="int",
        ),
        pytest.param(
            PrimType("float32"),
            """
            T.float32
            """,
            id="float",
        ),
        pytest.param(
            PointerType(PrimType("int32")),
            """
            T.Ptr(T.int32)
            """,
            id="pointer",
        ),
        pytest.param(
            PointerType(PrimType("int32"), "global"),
            """
            T.Ptr(T.int32, "global")
            """,
            id="with_scope",
        ),
        pytest.param(
            TupleType([]),
            """
            None
            """,
            id="none",
        ),
    ],
)
def test_type(ty, expected):
    assert format_script(expected) == as_tir_script(ty)


@pytest.mark.parametrize("var_type", [Var, SizeVar])
def test_var(var_type):
    var = var_type("x", "int8")

    assert as_tir_script(var) == format_script(
        """
        x: T.int8
        x
        """
    )


@pytest.mark.parametrize(
    "buffer, expected",
    [
        pytest.param(
            decl_buffer((5, 10), name="b"),
            """
            b: T.Buffer(shape=(5, 10))
            b
            """,
            id="simple",
        ),
        pytest.param(
            decl_buffer((5), name=""),
            """
            buf: T.Buffer(shape=(5,))
            buf
            """,
            id="no_name",
        ),
        pytest.param(
            decl_buffer((SizeVar("m", "int"), SizeVar("n", "int")), dtype="int8"),
            """
            m: T.int32
            n: T.int32
            buffer: T.Buffer("int8", shape=(m, n))
            buffer
            """,
            id="symbolic_shape",
        ),
        pytest.param(
            decl_buffer(
                (4, 10),
                dtype="int8",
                data=Var("p", PointerType(PrimType("int16"), "local")),
                strides=[2, 5],
                elem_offset=2,
                data_alignment=16,
                offset_factor=2,
                scope="local",
            ),
            """
            p: T.Ptr(T.int16, "local")
            buffer: T.Buffer("int8", shape=(4, 10), data=p, strides=(2, 5), elem_offset=2, scope="local", align=16, offset_factor=2)
            buffer
            """,
            id="all_param",
        ),
        pytest.param(
            decl_buffer(
                (4, 10),
                dtype="int32",
                data=Var("p", PointerType(PrimType("int32"))),
            ),
            """
            buffer: T.Buffer("int32", shape=(4, 10))
            buffer
            """,
            id="default_data_ptr",
        ),
        pytest.param(
            decl_buffer(
                (4, 10),
                dtype="bool",
            ),
            """
            buffer: T.Buffer("bool", shape=(4, 10))
            buffer
            """,
            # Boolean buffer has different ptr type (Int8) and buffer type (UInt1)
            id="bool_different_ptr_type",
        ),
    ],
)
def test_buffer(buffer, expected):
    assert as_tir_script(buffer) == format_script(expected)


def test_buffer_free_buffer_aliasing():
    ptr_var = Var("p", PointerType(PrimType("int16")))
    buffer_a = decl_buffer((4, 10), name="buffer_a", dtype="int8", data=ptr_var)
    buffer_b = decl_buffer((8, 5), name="buffer_b", dtype="int4", data=ptr_var)
    node = SeqStmt(
        [
            Evaluate(BufferLoad(buffer_a, [0, 0])),
            Evaluate(BufferLoad(buffer_b, [0, 1])),
        ]
    )
    # only prints one p
    expected = """
    p: T.Ptr(T.int16)
    buffer_a: T.Buffer("int8", shape=(4, 10), data=p)
    buffer_b: T.Buffer("int4", shape=(8, 5), data=p)
    T.evaluate(buffer_a[0, 0])
    T.evaluate(buffer_b[0, 1])
    """
    assert as_tir_script(node) == format_script(expected)


@pytest.mark.parametrize(
    "node, expected",
    [
        pytest.param(
            StringImm("test"),
            """
            "test"
            """,
            id="string",
        ),
        pytest.param(
            StringImm(""),
            """
            ""
            """,
            id="empty",
        ),
        pytest.param(
            StringImm("test1\ntest2\n"),
            r"""
            "test1\ntest2\n"
            """,
            id="multiline",
        ),
    ],
)
def test_string_imm(node, expected):
    assert as_tir_script(node) == format_script(expected)


@pytest.mark.parametrize(
    "node, expected",
    [
        pytest.param(
            IntImm("int32", 1),
            """
            1
            """,
            id="default-dtype",
        ),
        pytest.param(
            IntImm("int8", 0),
            """
            T.int8(0)
            """,
            id="int8",
        ),
        pytest.param(
            IntImm("int64", -1),
            """
            T.int64(-1)
            """,
            id="int64",
        ),
        pytest.param(
            IntImm("bool", 1),
            """
            True
            """,
            id="boolean-true",
        ),
        pytest.param(
            IntImm("bool", 0),
            """
            False
            """,
            id="boolean-true",
        ),
    ],
)
def test_int_imm(node, expected):
    assert as_tir_script(node) == format_script(expected)


@pytest.mark.parametrize(
    "node, expected",
    [
        pytest.param(
            FloatImm("float32", 1.5),
            """
            T.float32(1.5)
            """,
            id="f32",
        ),
        pytest.param(
            FloatImm("float16", 1.5),
            """
            T.float16(1.5)
            """,
            id="f16",
        ),
    ],
)
def test_float_imm(node, expected):
    assert as_tir_script(node) == format_script(expected)


@pytest.mark.parametrize(
    "node, expected",
    [
        pytest.param(
            Cast("float32", Var("x", dtype="int32")),
            """
            x: T.int32
            T.cast(x, "float32")
            """,
            id="with-var",
        ),
        pytest.param(
            Cast("int8", IntImm("int16", 1)),
            """
            T.cast(T.int16(1), "int8")
            """,
            id="with-var",
        ),
    ],
)
def test_cast(node, expected):
    assert as_tir_script(node) == format_script(expected)


@pytest.mark.parametrize(
    "node, expected",
    [
        pytest.param(
            Add(1, 0),
            """
            1 + 0
            """,
            id="add",
        ),
        pytest.param(
            Sub(1, 0),
            """
            1 - 0
            """,
            id="sub",
        ),
        pytest.param(
            Mul(-2.5, 1.5),
            """
            T.float32(-2.5) * T.float32(1.5)
            """,
            id="mul",
        ),
        pytest.param(
            Div(5, 2),
            """
            5 / 2
            """,
            id="div",
        ),
        pytest.param(
            FloorDiv(3, 2),
            """
            3 // 2
            """,
            id="floor-div",
        ),
        pytest.param(
            FloorMod(IntImm("int8", 5), IntImm("int8", 2)),
            """
            T.int8(5) % T.int8(2)
            """,
            id="floor-mod",
        ),
        pytest.param(
            LT(0, 1),
            """
            0 < 1
            """,
            id="lt",
        ),
        pytest.param(
            LE(1.0, 5.0),
            """
            T.float32(1) <= T.float32(5)
            """,
            id="le",
        ),
        pytest.param(
            GT(1, 0),
            """
            1 > 0
            """,
            id="gt",
        ),
        pytest.param(
            GE(Var("n", "int32"), 0),
            """
            n: T.int32
            n >= 0
            """,
            id="ge",
        ),
        pytest.param(
            EQ(Var("n", "int32"), Var("m", "int32")),
            """
            n: T.int32
            m: T.int32
            n == m
            """,
            id="eq",
        ),
        pytest.param(
            NE(1, 0),
            """
            1 != 0
            """,
            id="ne",
        ),
        pytest.param(
            And(IntImm("bool", 1), IntImm("bool", 0)),
            """
            True and False
            """,
            id="and",
        ),
        pytest.param(
            Or(IntImm("bool", 1), IntImm("bool", 0)),
            """
            True or False
            """,
            id="or",
        ),
    ],
)
def test_binary_op(node, expected):
    assert as_tir_script(node) == format_script(expected)


def test_mod():
    assert as_tir_script(Mod(5, 2)) == format_script(
        """
        T.truncmod(5, 2)
        """
    )


def test_min():
    assert as_tir_script(Min(5, 2)) == format_script(
        """
        T.min(5, 2)
        """
    )


def test_max():
    assert as_tir_script(Max(5, 2)) == format_script(
        """
        T.max(5, 2)
        """
    )


def test_not():
    assert as_tir_script(Not(IntImm("bool", 1))) == format_script(
        """
        not True
        """
    )


def test_select():
    node = Select(IntImm("bool", 1), 0, 1)
    assert as_tir_script(node) == format_script(
        """
        T.Select(True, 0, 1)
        """
    )


@pytest.mark.parametrize(
    "node, expected",
    [
        pytest.param(
            BufferLoad(decl_buffer((5, 10), name="b"), [0, 1]),
            """
            b: T.Buffer(shape=(5, 10))
            b[0, 1]
            """,
            id="normal",
        ),
        pytest.param(
            BufferLoad(
                decl_buffer((5,), name="b"),
                [
                    0,
                ],
            ),
            """
            b: T.Buffer(shape=(5,))
            b[0]
            """,
            id="1d",
        ),
        pytest.param(
            BufferLoad(decl_buffer((), name="b"), []),
            """
            b: T.Buffer(shape=())
            b[()]
            """,
            id="0d",
        ),
    ],
)
def test_buffer_load(node, expected):
    assert as_tir_script(node) == format_script(expected)


def test_ramp():
    node = Ramp(0, 1, 8)
    expected = """
    T.ramp(0, 1, 8)
    """
    assert as_tir_script(node) == format_script(expected)


def test_broadcast():
    node = Broadcast(0, 4)
    expected = """
    T.broadcast(0, 4)
    """
    assert as_tir_script(node) == format_script(expected)


def test_let():
    x = Var("x", "int32")
    y = Var("y", "int32")
    node = Let(x, 1, x + y)
    # Not var definition for x because x isn't free variable in this expression
    expected = """
    x: T.int32
    y: T.int32
    T.let(x, 1, x + y)
    """
    assert as_tir_script(node) == format_script(expected)


def test_call_tir_op():
    node = Call("float64", "tir.exp", [0.0])
    expected = """
    T.exp(T.float32(0))
    """
    assert as_tir_script(node) == format_script(expected)


def test_call_global_var():
    f_var = GlobalVar("test_f")
    node = Call("float32", f_var, [0, 1])
    expected = """
    test_f(0, 1)
    """
    assert as_tir_script(node) == format_script(expected)


def test_shuffle():
    x = Var("x", "int32")
    y = Var("y", "int32")
    node = Shuffle([x, 1, 10], [0, 1, y])
    expected = """
    x: T.int32
    y: T.int32
    T.shuffle([x, 1, 10], [0, 1, y])
    """
    assert as_tir_script(node) == format_script(expected)


def test_comm_reducer_single_value():
    x = Var("x", "int32")
    y = Var("y", "int32")
    node = CommReducer([x], [y], [x + y], [0])
    expected = """
    T.comm_reducer(lambda x, y: x + y, [0])
    """
    assert as_tir_script(node) == format_script(expected)


def test_comm_reducer_multi_value():
    x0 = Var("x0", "int32")
    x1 = Var("x1", "int32")
    y0 = Var("y0", "int32")
    y1 = Var("y1", "int32")
    node = CommReducer([x0, x1], [y0, y1], [x0 + y0, x1 * y1], [0, 1])
    expected = """
    T.comm_reducer(lambda x0, x1, y0, y1: (x0 + y0, x1 * y1), [0, 1])
    """
    assert as_tir_script(node) == format_script(expected)


def test_reduce():
    x = Var("x", "int32")
    y = Var("y", "int32")
    m = Var("m", "int32")
    comm_reducer = CommReducer([x], [y], [x + y], [0])
    node = Reduce(comm_reducer, [m], [IterVar(None, Var("i", "int32"), 2)], True, 0)
    expected = """
    i: T.int32
    m: T.int32
    T.reduce(T.comm_reducer(lambda x, y: x + y, [0]), [m], [T.iter_var(i, None, "CommReduce", "")], 0)
    """
    assert as_tir_script(node) == format_script(expected)


@pytest.mark.parametrize(
    "node, expected",
    [
        pytest.param(
            Range(0, 1),
            """
            _[0:1]
            """,
            id="normal",
        ),
        pytest.param(
            Range(10),
            """
            _[0:10]
            """,
            id="one-arg",
        ),
    ],
)
def test_range(node, expected):
    assert as_tir_script(node) == format_script(expected)


@pytest.mark.parametrize(
    "node, expected",
    [
        pytest.param(
            BufferRegion(decl_buffer((5, 10), name="b"), [Range(0, 4), Range(0, 9)]),
            """
            b: T.Buffer(shape=(5, 10))
            b[0:4, 0:9]
            """,
            id="normal",
        ),
        pytest.param(
            BufferRegion(decl_buffer((5, 10), name="b"), [Range(0, 1), Range(5, 9)]),
            """
            b: T.Buffer(shape=(5, 10))
            b[0, 5:9]
            """,
            id="scalar-range",
        ),
        pytest.param(
            BufferRegion(decl_buffer((5,), name="b"), [Range(0, 3)]),
            """
            b: T.Buffer(shape=(5,))
            b[0:3]
            """,
            id="1d",
        ),
        pytest.param(
            BufferRegion(decl_buffer((), name="b"), []),
            """
            b: T.Buffer(shape=())
            b[()]
            """,
            id="0d",
        ),
    ],
)
def test_buffer_region(node, expected):
    assert as_tir_script(node) == format_script(expected)


@pytest.mark.parametrize(
    "node, expected",
    [
        pytest.param(
            BufferStore(decl_buffer((5, 10), name="buf"), 1.0, [0, 1]),
            """
            buf: T.Buffer(shape=(5, 10))
            buf[0, 1] = T.float32(1)
            """,
            id="2d",
        ),
        pytest.param(
            BufferStore(decl_buffer((5,), name="buf"), 1.0, [0]),
            """
            buf: T.Buffer(shape=(5,))
            buf[0] = T.float32(1)
            """,
            id="1d",
        ),
        pytest.param(
            BufferStore(decl_buffer((), name="buf"), 1.0, []),
            """
            buf: T.Buffer(shape=())
            buf[()] = T.float32(1)
            """,
            id="0d",
        ),
    ],
)
def test_buffer_store(node, expected):
    assert as_tir_script(node) == format_script(expected)


def test_evaluate():
    var = Var("a", "int32")
    node = Evaluate(3 + var)
    expected = """
    a: T.int32
    T.evaluate(3 + a)
    """
    assert as_tir_script(node) == format_script(expected)


def test_assert_normal_form():
    body = SeqStmt([Evaluate(2), Evaluate(3)])
    node = SeqStmt([AssertStmt(1, StringImm("test"), body), Evaluate(4)])
    expected = """
    with T.Assert(1, "test"):
        T.evaluate(2)
        T.evaluate(3)
    T.evaluate(4)
    """
    assert as_tir_script(node) == format_script(expected)


def test_assert_concise_form():
    body = SeqStmt([Evaluate(2), Evaluate(3)])
    node = AssertStmt(1, StringImm("test"), body)
    expected = """
    assert 1, "test"
    T.evaluate(2)
    T.evaluate(3)
    """
    assert as_tir_script(node) == format_script(expected)


def test_assert_body_concise_form():
    body_assert1 = AssertStmt(2, StringImm("test"), Evaluate(3))
    body_assert2 = AssertStmt(4, StringImm("test"), Evaluate(5))
    body = SeqStmt([body_assert1, body_assert2])
    node = AssertStmt(1, StringImm("test"), body)
    expected = """
    assert 1, "test"
    with T.Assert(2, "test"):
        T.evaluate(3)
    assert 4, "test"
    T.evaluate(5)
    """
    assert as_tir_script(node) == format_script(expected)


def test_if_then_else():
    node = IfThenElse(1, Evaluate(2), Evaluate(3))
    expected = """
    if 1:
        T.evaluate(2)
    else:
        T.evaluate(3)
    """
    assert as_tir_script(node) == format_script(expected)


def test_if_then_else_empty_else():
    node = IfThenElse(1, Evaluate(2), None)
    expected = """
    if 1:
        T.evaluate(2)
    """
    assert as_tir_script(node) == format_script(expected)


def test_if_then_else_free_var():
    var = Var("a", "int32")
    node = IfThenElse(1, Evaluate(var), Evaluate(var + 1))
    expected = """
    a: T.int32
    if 1:
        T.evaluate(a)
    else:
        T.evaluate(a + 1)
    """
    assert as_tir_script(node) == format_script(expected)


def test_while():
    var = Var("a", "int32")
    node = While(var, Evaluate(var * 2))
    expected = """
    a: T.int32
    while a:
        T.evaluate(a * 2)
    """
    assert as_tir_script(node) == format_script(expected)


def test_prefetch():
    buf = decl_buffer((5, 10), name="buf")
    ranges = [Range(1, 2), Range(5, 6)]
    node = Prefetch(buf, ranges)
    expected = """
    buf: T.Buffer(shape=(5, 10))
    T.prefetch(buf[1:2, 5:6])
    """
    assert as_tir_script(node) == format_script(expected)


def test_let_stmt():
    x = Var("x", "int32")
    node = LetStmt(x, 1, Evaluate(x + 1))
    expected = """
    x: T.int32 = 1
    T.evaluate(x + 1)
    """
    assert as_tir_script(node) == format_script(expected)


def test_let_stmt_concise_scoping():
    x = Var("x", "int32")
    node = SeqStmt([LetStmt(x, 1, Evaluate(x + 1)), LetStmt(x, 2, Evaluate(x + 2))])
    expected = """
    x: T.int32 = T.var("int32")
    with T.let(x, 1):
        T.evaluate(x + 1)
    x: T.int32 = 2
    T.evaluate(x + 2)
    """
    assert as_tir_script(node) == format_script(expected)


def test_let_stmt_alias_buffer():
    buf0 = decl_buffer((10, 10), name="buf0")
    ptr = Var("ptr", PointerType(PrimType("float32")))
    buf1 = decl_buffer((5, 10), name="buf1", data=ptr)
    buf2 = decl_buffer((2, 5), name="buf2", data=ptr)
    node = LetStmt(ptr, buf0.data, SeqStmt([BufferStore(buf1, BufferLoad(buf2, [2, 4]), [1, 2])]))
    expected = """
    buf0: T.Ptr(T.float32)
    ptr: T.Ptr(T.float32) = buf0
    buf2 = T.decl_buffer(shape=(2, 5), data=ptr)
    buf1 = T.decl_buffer(shape=(5, 10), data=ptr)
    buf1[1, 2] = buf2[2, 4]
    """
    assert as_tir_script(node) == format_script(expected)


def test_allocate_stmt_all_params():
    buf = decl_buffer((10, 10), name="buf", scope="global")
    body = BufferStore(buf, 0, [1, 1])
    node = Allocate(buf.data, "float32", [10, 10], Var("cond", "bool"), body, {"attr": 3})
    expected = """
    cond: T.bool
    buf = T.allocate([10, 10], "float32", "global", cond, annotations={"attr": 3})
    buf[1, 1] = 0
    """
    assert as_tir_script(node) == format_script(expected)


def test_allocate_stmt_concise_scoping():
    buf = decl_buffer((10, 10), name="buf", scope="global")
    body = BufferStore(buf, 0, [1, 1])
    node = SeqStmt(
        [
            Allocate(buf.data, "float32", [10, 10], IntImm("bool", 1), body),
            Allocate(buf.data, "float32", [10, 10], IntImm("bool", 1), body),
        ]
    )
    expected = """
    with T.allocate([10, 10], "float32", "global") as buf:
        buf[1, 1] = 0
    buf = T.allocate([10, 10], "float32", "global")
    buf[1, 1] = 0
    """
    assert as_tir_script(node) == format_script(expected)


def test_allocate_aliasing_buffers():
    buf1 = decl_buffer((10, 10), name="buf1", scope="global")
    buf2 = decl_buffer((5, 20), name="buf2", data=buf1.data)
    buf_unrelated = decl_buffer((10, 10), name="buf_unrelated")
    body = BufferStore(buf1, BufferLoad(buf2, [2, 2]) + BufferLoad(buf_unrelated, [3, 3]), [1, 1])
    node = Allocate(buf1.data, "float32", [10, 10], IntImm("bool", 1), body)
    expected = """
    buf_unrelated: T.Buffer(shape=(10, 10))
    buf1 = T.allocate([10, 10], "float32", "global")
    buf2 = T.decl_buffer(shape=(5, 20), data=buf1.data)
    buf1[1, 1] = buf2[2, 2] + buf_unrelated[3, 3]
    """
    assert as_tir_script(node) == format_script(expected)


def test_attr_stmt():
    buf = decl_buffer((10, 10), name="buf")
    body = BufferStore(buf, 0, [0, 0])
    node = AttrStmt(buf.data, "storage_alignment", 128, body)
    expected = """
    buf: T.Ptr(T.float32)
    buf_1: T.Buffer(shape=(10, 10), data=buf)
    T.attr(buf, "storage_alignment", 128)
    buf_1[0, 0] = 0
    """
    assert as_tir_script(node) == format_script(expected)


def test_attr_stmt_concise_scoping():
    buf = decl_buffer((10, 10), name="buf")
    body = BufferStore(buf, 0, [0, 0])
    node = SeqStmt(
        [
            AttrStmt(buf.data, "storage_alignment", 128, body),
            AttrStmt(buf.data, "storage_alignment", 64, body),
        ]
    )
    expected = """
    buf: T.Ptr(T.float32)
    buf_1: T.Buffer(shape=(10, 10), data=buf)
    with T.attr(buf, "storage_alignment", 128):
        buf_1[0, 0] = 0
    T.attr(buf, "storage_alignment", 64)
    buf_1[0, 0] = 0
    """
    assert as_tir_script(node) == format_script(expected)


def test_attr_stmt_buffer_realize():
    buf = decl_buffer((10, 10), name="buf")
    body = BufferRealize(
        buf, [Range(0, 1), Range(1, 2)], Var("cond", "bool"), BufferStore(buf, 0, [0, 0])
    )
    node = AttrStmt(buf, "realize_scope", StringImm("global"), body)
    expected = """
    buf: T.Buffer(shape=(10, 10))
    cond: T.bool
    T.realize(buf[0:1, 1:2], "global", cond)
    buf[0, 0] = 0
    """
    assert as_tir_script(node) == format_script(expected)


def test_attr_stmt_launch_thread():
    iter_var = IterVar(Range(1, 5), Var("i", "int32"), 2, "blockIdx.x")
    body = Evaluate(iter_var)
    node = AttrStmt(iter_var, "thread_extent", 8, body)
    expected = """
    i = T.env_thread("blockIdx.x")
    T.launch_thread(i, 8)
    T.evaluate(i)
    """
    assert as_tir_script(node) == format_script(expected)


def test_attr_stmt_launch_thread_multiple():
    iter_var = IterVar(Range(1, 5), Var("i", "int32"), 2, "blockIdx.x")
    body = AttrStmt(iter_var, "thread_extent", 24, Evaluate(iter_var))
    node = AttrStmt(iter_var, "thread_extent", 8, body)
    expected = """
    i = T.env_thread("blockIdx.x")
    T.launch_thread(i, 8)
    T.launch_thread(i, 24)
    T.evaluate(i)
    """
    assert as_tir_script(node) == format_script(expected)


@pytest.mark.parametrize(
    "for_kind, f_name",
    [
        (ForKind.SERIAL, "serial"),
        (ForKind.PARALLEL, "parallel"),
        (ForKind.VECTORIZED, "vectorized"),
        (ForKind.UNROLLED, "unroll"),
    ],
)
def test_for_single_loop(for_kind, f_name):
    i = Var("i", "int32")
    node = For(i, 3, 13, for_kind, Evaluate(i), annotations={"abc": StringImm("xyz")})
    expected = f"""
    for i in T.{f_name}(3, 16, annotations={{"abc": "xyz"}}):
        T.evaluate(i)
    """
    assert as_tir_script(node) == format_script(expected)


def test_for_thread_binding():
    i = Var("i", "int32")
    iter_var = IterVar(Range(1, 5), i, 2, "blockIdx.x")
    node = For(i, 0, 16, ForKind.THREAD_BINDING, Evaluate(i), iter_var)
    expected = """
    for i in T.thread_binding(16, thread="blockIdx.x"):
        T.evaluate(i)
    """
    assert as_tir_script(node) == format_script(expected)


def test_for_merged_simple_loops():
    i = Var("i", "int32")
    j = Var("j", "int32")
    k = Var("k", "int32")
    loop3 = For(k, 0, 16, ForKind.SERIAL, Evaluate(i + j + k))
    loop2 = For(j, 0, 16, ForKind.SERIAL, loop3)
    loop1 = For(i, 0, 16, ForKind.SERIAL, loop2)
    node = loop1
    expected = """
    for i, j, k in T.grid(16, 16, 16):
        T.evaluate(i + j + k)
    """
    assert as_tir_script(node) == format_script(expected)


def test_for_nested_non_simple_loops_non_zero_min_val():
    i = Var("i", "int32")
    j = Var("j", "int32")
    k = Var("k", "int32")
    loop3 = For(k, 0, 16, ForKind.SERIAL, Evaluate(i + j + k))
    loop2 = For(j, 1, 16, ForKind.SERIAL, loop3)
    loop1 = For(i, 0, 16, ForKind.SERIAL, loop2)
    node = loop1
    expected = """
    for i in T.serial(16):
        for j in T.serial(1, 17):
            for k in T.serial(16):
                T.evaluate(i + j + k)
    """
    assert as_tir_script(node) == format_script(expected)


def test_for_nested_non_simple_loops_var_dependency():
    i = Var("i", "int32")
    j = Var("j", "int32")
    k = Var("k", "int32")
    loop3 = For(k, 0, 16, ForKind.SERIAL, Evaluate(i + j + k))
    loop2 = For(j, 0, i, ForKind.SERIAL, loop3)
    loop1 = For(i, 0, 16, ForKind.SERIAL, loop2)
    node = loop1
    expected = """
    for i in T.serial(16):
        for j, k in T.grid(i, 16):
            T.evaluate(i + j + k)
    """
    assert as_tir_script(node) == format_script(expected)


def test_for_nested_non_simple_loops_non_serial_for_kind():
    i = Var("i", "int32")
    j = Var("j", "int32")
    k = Var("k", "int32")
    loop3 = For(k, 0, 16, ForKind.SERIAL, Evaluate(i + j + k))
    loop2 = For(j, 0, 16, ForKind.VECTORIZED, loop3)
    loop1 = For(i, 0, 16, ForKind.SERIAL, loop2)
    node = loop1
    expected = """
    for i in T.serial(16):
        for j in T.vectorized(16):
            for k in T.serial(16):
                T.evaluate(i + j + k)
    """
    assert as_tir_script(node) == format_script(expected)


def test_block_simple():
    node = Block([], [], [], "main", Evaluate(1))
    expected = """
    T.evaluate(1)
    """
    assert as_tir_script(node) == format_script(expected)


def test_block_all_params():
    i = IterVar(Range(1, 5), Var("i", "int32"), IterVar.DataPar)
    j = IterVar(Range(1, 5), Var("j", "int32"), IterVar.DataPar)
    a = decl_buffer((5, 10), name="a")
    region_a = BufferRegion(a, [Range(0, 4), Range(0, 9)])
    b = decl_buffer((5, 12), name="b")
    region_b = BufferRegion(b, [Range(0, 2), Range(0, 7)])
    c = decl_buffer((5, 12), name="c")
    d = decl_buffer((4, 9), name="d")
    e = decl_buffer((2, 2), name="e")
    node = Block(
        [i, j],
        [region_a],
        [region_b],
        "main",
        Evaluate(1),
        init=Evaluate(2),
        alloc_buffers=[c],
        match_buffers=[
            MatchBufferRegion(d, region_a),
            MatchBufferRegion(e, BufferRegion(d, [Range(0, 2), Range(0, 2)])),
        ],
        annotations={"attr": StringImm("value")},
    )
    expected = """
    i: T.int32
    j: T.int32
    a: T.Buffer(shape=(5, 10))
    b: T.Buffer(shape=(5, 12))
    T.iter_var(i, (1, 5), "DataPar", "")
    T.iter_var(j, (1, 5), "DataPar", "")
    T.reads(a[0:4, 0:9])
    T.writes(b[0:2, 0:7])
    T.block_attr({"attr": "value"})
    c = T.alloc_buffer(shape=(5, 12))
    d = T.match_buffer(a[0:4, 0:9], shape=(4, 9))
    e = T.match_buffer(d[0:2, 0:2], shape=(2, 2))
    with T.init():
        T.evaluate(2)
    T.evaluate(1)
    """
    assert as_tir_script(node) == format_script(expected)


def test_block_realize_simple():
    i = Var("i", "int32")
    j = Var("j", "int32")
    v_i = IterVar(Range(1, 5), Var("v_i", "int32"), IterVar.DataPar)
    v_j = IterVar(Range(3, 7), Var("v_j", "int32"), IterVar.DataPar)
    block = Block([v_i, v_j], [], [], "main", Evaluate(1))
    node = BlockRealize([i, j + 2], True, block)
    expected = """
    i: T.int32
    j: T.int32
    with T.block("main"):
        v_i = T.axis.spatial((1, 5), i)
        v_j = T.axis.spatial((3, 7), j + 2)
        T.evaluate(1)
    """
    assert as_tir_script(node) == format_script(expected)


def test_block_realize_with_predicate():
    p = Var("p", "bool")
    i = Var("i", "int32")
    v_i = IterVar(Range(1, 5), Var("v_i", "int32"), IterVar.DataPar)
    block = Block([v_i], [], [], "main", Evaluate(1))
    node = BlockRealize([i], p, block)
    expected = """
    i: T.int32
    p: T.bool
    with T.block("main"):
        v_i = T.axis.spatial((1, 5), i)
        T.where(p)
        T.evaluate(1)
    """
    assert as_tir_script(node) == format_script(expected)


def test_block_realize_merged_simple_block_vars():
    i = Var("i", "int32")
    j = Var("j", "int32")
    k = Var("k", "int32")
    m = Var("m", "int32")
    n = Var("n", "int32")

    v_i = IterVar(Range(0, 16), Var("v_i", "int32"), IterVar.DimInfo)
    v_j = IterVar(Range(0, 16), Var("v_j", "int32"), IterVar.DataPar)
    v_k = IterVar(Range(0, 16), Var("v_k", "int32"), IterVar.CommReduce)
    v_m = IterVar(Range(0, 16), Var("v_m", "int32"), IterVar.DataPar)
    v_n = IterVar(Range(1, 17), Var("v_n", "int32"), IterVar.DataPar)
    block = Block([v_i, v_j, v_k, v_m, v_n], [], [], "main", Evaluate(1))
    block_realize = BlockRealize([i, j, k, m, n + 1], True, block)

    loop5 = For(n, 0, 16, ForKind.SERIAL, block_realize)
    loop4 = For(m, 0, 16, ForKind.SERIAL, loop5)
    loop3 = For(k, 0, 16, ForKind.SERIAL, loop4)
    loop2 = For(j, 0, 16, ForKind.SERIAL, loop3)
    loop1 = For(i, 0, 16, ForKind.SERIAL, loop2)
    node = loop1

    expected = """
    for i, j, k, m, n in T.grid(16, 16, 16, 16, 16):
        with T.block("main"):
            v_i = T.axis.opaque(16, i)
            v_j, v_k, v_m = T.axis.remap("SRS", [j, k, m])
            v_n = T.axis.spatial((1, 17), n + 1)
            T.evaluate(1)
    """
    assert as_tir_script(node) == format_script(expected)


def test_function_basic():
    @T.prim_func
    def f(a: T.handle) -> None:
        A = T.match_buffer(a, [128, 128], elem_offset=1)
        A[0, 1] = 1

    expected = """
    @T.prim_func
    def main(a: T.handle) -> None:
        A = T.match_buffer(a, shape=(128, 128), elem_offset=1)
        A[0, 1] = 1
    """
    assert as_tir_script(f) == format_script(expected)


def test_function_simple_buffer():
    @T.prim_func
    def f(a: T.handle) -> None:
        A = T.match_buffer(a, [128, 128])
        A[0, 1] = 1

    expected = """
    @T.prim_func
    def main(A: T.Buffer(shape=(128, 128))) -> None:
        A[0, 1] = 1
    """
    assert as_tir_script(f) == format_script(expected)


def test_function_all_params():
    @T.prim_func
    def f(a: T.handle, b: T.int32, c: T.handle) -> None:
        T.func_attr({"global_symbol": "test_func", "tir.noalias": True})
        m = T.var("int32")
        A = T.match_buffer(a, [128, 128])
        C = T.match_buffer(c, [4, m], elem_offset=16)
        T.preflattened_buffer(C, (2, 4, 2), data=C.data)
        A[0, 1] = 1

    expected = """
    @T.prim_func
    def test_func(A: T.Buffer(shape=(128, 128)), b: T.int32, c: T.handle) -> None:
        T.func_attr({"global_symbol": "test_func", "tir.noalias": True})
        m = T.var("int32")
        C = T.match_buffer(c, shape=(4, m), elem_offset=16)
        T.preflattened_buffer(C, shape=(2, 4, 2), data=C.data)
        A[0, 1] = 1
    """
    assert as_tir_script(f) == format_script(expected)


def test_function_in_ir_module():
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def f1(a: T.handle) -> None:
            A = T.match_buffer(a, [128, 128])
            A[0, 1] = 1

        @T.prim_func
        def f2(b: T.handle) -> None:
            B = T.match_buffer(b, [128, 128])
            B[0, 1] = 1

    expected = """
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def f2(B: T.Buffer(shape=(128, 128))) -> None:
            B[0, 1] = 1

        @T.prim_func
        def f1(A: T.Buffer(shape=(128, 128))) -> None:
            A[0, 1] = 1
    """
    assert as_tir_script(Module) == format_script(expected)
