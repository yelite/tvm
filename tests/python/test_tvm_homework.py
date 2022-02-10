import pytest
import tvm


def test_tvm_homework_main():
    f = tvm._ffi.get_global_func("tvm_homework.main")
    assert f() == 1
