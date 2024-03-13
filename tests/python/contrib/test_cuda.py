import numpy as np
import tvm
import tvm.testing


def test_reduce_max_abs():
    target = "cuda"
    dev = tvm.device(target, 0)
    x_shape = (4, 4096)
    dtype = "float16"
    x = tvm.nd.array(np.random.uniform(-2, 1.4, x_shape).astype(dtype), dev)
    scalar = tvm.nd.array(np.array([0], dtype=dtype), dev)

    reduce = tvm._ffi.get_global_func("tvm.contrib.cuda.reduce_max_abs")
    reduce(x, scalar)
    tvm.testing.assert_allclose(scalar.numpy(), np.array([2], dtype=dtype))


if __name__ == "__main__":
    tvm.testing.main()
