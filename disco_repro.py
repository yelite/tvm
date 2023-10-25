import threading

import numpy as np

import tvm
from tvm.runtime import disco as di

dev = tvm.device("cuda", 0)


def _numpy_to_worker_0(sess: di.Session, np_array: np.array):
    # x_array = sess.empty(np_array.shape, "float32", device=dev)
    x_array = sess.empty(np_array.shape, "float32")
    host_array = tvm.nd.array(np_array, device=dev)
    sess.copy_to_worker_0(host_array, x_array)
    return x_array


def f(sess):
    x_np = np.arange(8 * 16).astype("float32").reshape([8, 16])
    x_disc = _numpy_to_worker_0(sess, x_np)


def main():
    sess = di.ProcessSession(num_workers=2)
    for _ in range(50):
        f(sess)


thread = threading.Thread(target=main)
thread.start()
thread.join()
