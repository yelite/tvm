from tvm.script import tir as T


def tvmscript_to_string(node) -> str:
    return ""


def format_script(s: str) -> str:
    """
    Remove leading and trailing blank lines, and make the minimum idention 0
    """
    s = s.strip("\n")
    lines = [line for line in s.splitlines() if len(line.lstrip(' ')) > 0]
    indent_level = min(len(line) - len(line.lstrip(' ')) for line in lines)
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
        # from tvm.script import tir as T
        @T.prim_func
        def func(A: T.Buffer[(8,), "float32"], B: T.Buffer[(8,), "float32"]) -> None:
            # function attr dict
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            # body
            # with T.block("root")
            for i in T.serial(8):
                with T.block("B"):
                    vi = T.axis.spatial(8, i)
                    T.reads(A[vi])
                    T.writes(B[vi])
                    B[vi] = A[vi] + T.float32(1)
    """
    assert tvmscript_to_string(main) == format_script(expected)


def test_roundtrippable_basic_fragment():
    # `B[vi] = A[vi] + 1.0`
    buffer_store_node = main.body.block.body.body.block.body

    expected = """
        A: T.Buffer[(8,), "float32"]
        B: T.Buffer[(8,), "float32"]
        vi: "int32"
        B[vi] = A[vi] + 1.0
    """
    assert tvmscript_to_string(buffer_store_node) == format_script(expected)

