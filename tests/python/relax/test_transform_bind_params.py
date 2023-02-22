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

import numpy as np
import tvm
import tvm.script
import tvm.testing
from tvm import relax
from tvm.script import relax as R
from tvm.script import tir as T

use_np_array = tvm.testing.parameter(False, True)


def test_bind_params(use_np_array):
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def tir_matmul(x: T.handle, y: T.handle, z: T.handle) -> None:
            T.func_attr({"global_symbol": "tir_matmul"})
            A = T.match_buffer(x, (16, 16))
            B = T.match_buffer(y, (16, 16))
            C = T.match_buffer(z, (16, 16))
            for i0, j, k0, i1, k1 in T.grid(4, 16, 4, 4, 4):
                with T.block("matmul"):
                    vi = T.axis.S(16, i0 * 4 + i1)
                    vj = T.axis.S(16, j)
                    vk = T.axis.R(16, k0 * 4 + k1)
                    with T.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

        @R.function
        def main(
            x: R.Tensor((16, 16), "float32"), w: R.Tensor((16, 16), "float32")
        ) -> R.Tensor((16, 16), "float32"):
            gv0 = R.call_tir(tir_matmul, (x, w), R.Tensor((16, 16), dtype="float32"))
            return gv0

    x_np = np.random.rand(16, 16).astype(np.float32)
    w_np = np.random.rand(16, 16).astype(np.float32)
    x_tvm = tvm.nd.array(x_np)
    w_tvm = tvm.nd.array(w_np)
    params_dict = {"w": w_np if use_np_array else w_tvm}
    mod = relax.transform.BindParams("main", params_dict)(InputModule)
    assert len(mod["main"].params) == 1

    target = tvm.target.Target("llvm")
    ex_after = relax.build(mod, target)
    vm_after = relax.VirtualMachine(ex_after, tvm.cpu())
    res_after = vm_after["main"](x_tvm)

    ex_before = relax.build(InputModule, target)
    vm_before = relax.VirtualMachine(ex_before, tvm.cpu())
    res_before = vm_before["main"](x_tvm, w_tvm)

    tvm.testing.assert_allclose(res_before.numpy(), res_after.numpy())


if __name__ == "__main__":
    tvm.testing.main()
