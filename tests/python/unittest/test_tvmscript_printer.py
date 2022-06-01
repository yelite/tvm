import pytest

from tvm.script import tir as T
import tvm

_as_tvm_script = tvm.get_global_func("experiment.AsTVMScript")


def to_tvmscript(node) -> str:
    return _as_tvm_script(node, {"tir": "T"}, 4)


def format_script(s: str) -> str:
    """
    Remove leading and trailing blank lines, and make the minimum idention 0
    """
    s = s.strip("\n")
    lines = [line for line in s.splitlines()]
    # TODO expand this line
    indent_level = min(len(line) - len(line.lstrip(' ')) for line in lines if len(line.lstrip(' ')) > 0)
    return "\n".join(line[indent_level:] for line in lines)


@T.prim_func
def main(A: T.Buffer[(8,), "float32"], B: T.Buffer[(8,), "float32"]):
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    for i in range(8):
        with T.block("B"):
            vi = T.axis.spatial(8, i)
            B[vi] = A[vi] + 1.0


@pytest.mark.xfail(reason="still have formating issue on trailing new lines")
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

