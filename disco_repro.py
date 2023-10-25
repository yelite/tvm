import tempfile
import time
import threading

import numpy as np
import gc

import tvm
from tvm import relax as rx
from tvm.runtime import disco as di
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T

device = tvm.cpu()
 
def collect_callback(phase, info):
    if phase == "stop":
        for obj in info.objects:
            print(f"Collecting object {obj!r}")

# Register the callback function
gc.callbacks.append(collect_callback)


@I.ir_module
class TestMod:
    @T.prim_func
    def transpose(A: T.Buffer((8, 16), "float32"), B: T.Buffer((16, 8), "float32")):
        for i, j in T.grid(16, 8):
            with T.block("transpose"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vj, vi]

    @R.function
    def main(A: R.Tensor((8, 16), dtype="float32")) -> R.Tensor((16, 8), dtype="float32"):
        cls = TestMod
        with R.dataflow():
            B = R.call_tir(cls.transpose, (A,), out_sinfo=R.Tensor((16, 8), dtype="float32"))
            R.output(B)
        return B


def _numpy_to_worker_0(sess: di.Session, np_array: np.array):
    # x_array = sess.empty(np_array.shape, "float32", device=dev)
    x_array = sess.empty(np_array.shape, "float32")
    host_array = tvm.nd.array(np_array, device=device)
    sess.copy_to_worker_0(host_array, x_array)
    return x_array


def _numpy_from_worker_0(sess: di.Session, remote_array, shape, dtype):
    host_array = tvm.nd.empty(shape, dtype, device=device)
    sess.copy_from_worker_0(host_array, remote_array)
    sess.sync_worker_0()
    return host_array.numpy()


def f(sess, mod):
    sess.sync_worker_0()
    x_np = np.arange(8 * 16).astype("float32").reshape([8, 16])
    x_disc = _numpy_to_worker_0(sess, x_np)
    time.sleep(0.01)
    y_disc = mod["main"](x_disc)
    time.sleep(0.01)
    y_nd = _numpy_from_worker_0(sess, y_disc, shape=y_np.shape, dtype=y_np.dtype)


def main(lib_path):
    sess = di.ProcessSession(num_workers=2)
    mod = sess.load_vm_module(path, device=device)
    for _ in range(75):
        f(sess, mod)


with tempfile.TemporaryDirectory() as tmpdir:
    path = tmpdir + "/test.so"
    device = tvm.cpu()
    x_np = np.arange(8 * 16).astype("float32").reshape([8, 16])
    y_np = x_np.transpose()
    rx.build(TestMod, target="llvm").export_library(path)

    thread = threading.Thread(target=main, args=(path,))
    thread.start()

    for i in range(300):
        d = np.arange(8 * 16).astype("float32").reshape([8, 16])
        time.sleep(0.005)
        if i % 100 == 0:
            gc.collect()

    thread.join()
