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
"""
Wrap Your Tensor IR with PyTorch Module
======================
**Author**: `Yaoda Zhou <https://github.com/juda/>`_
This article is an introductory tutorial to wrap the Tensor IR code with PyTorch module.
By the decorator `as_torch`, users are able to import a Tensor IR code in PyTorch with a low cost. 
For us to follow this tutorial, PyTorch as well as TorchVision should be installed.
For avoiding potential "undefined symbol" issue, we strongly recommend to install PyTorch built with Cxx11 ABI from Conda, as
.. code-block:: bash
    conda install -c conda-forge pytorch-gpu
"""
# Import Tvm and PyTorch, as well as necessary libraries
import tvm
import torch
from tvm.contrib.torch import as_torch
from tvm.script import tir as T
import numpy as np
import torch.nn
import tvm.testing


######################################################################
# Define an example of vector add 
# (This example could be found at https://tvm.apache.org/docs/tutorial/tensor_ir_blitz_course.html)
# -------------------------------
# Our `as_torch` is a simple decorator: put it on any Tensor IR code and it will convert it into PyTorch module automatically.
@as_torch
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (8,), dtype="float32")
        B = T.match_buffer(b, (8,), dtype="float32")
        for i in range(8):
            with T.block("B"):
                vi = T.axis.spatial(8, i)
                B[vi] = A[vi] + 1.0

######################################################################
# Write a test case: Tvm's testing is used to compare two tensors
# -------------------------------
def test_tvmscript_torch_decorator():
    s1 = np.arange(8).astype("float32")

    # Define two torch tensors
    q1 = torch.arange(8).type(torch.float32)
    q2 = torch.zeros((8,), dtype=torch.float32)

    # Result from numpy
    numpy_result = s1 + 1

    tvm_module = MyModule

    # We call `MyModule` as PyTorch module's forward
    tvm_module(q1, q2)

    # Testing. No output implies that tensors are equal
    tvm.testing.assert_allclose(q2.numpy(), numpy_result, atol=1e-5, rtol=1e-5)
    
test_tvmscript_torch_decorator()

######################################################################
# Another example: matrix multiplication with a limit form meta-programming
# -------------------------------
# As above, we can add `as_torch` decorator to a Tensor IR function.
@as_torch
def matmul(M: int, N: int, K: int, dtype: str):
    @T.prim_func
    def f(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, [M, K], dtype=dtype)
        B = T.match_buffer(b, [N, K], dtype=dtype)
        C = T.match_buffer(c, [M, N], dtype=dtype)
        for i, j, k in T.grid(M, N, K):
            with T.block():
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]
    return f

######################################################################
# Test case for `matmul` function.
# -------------------------------
def test_tvmscript_torch_matmul():
    # Create two 128 x 128 matrixs as input
    s1 = np.random.rand(128,128).astype("float32")
    s2 = np.random.rand(128,128).astype("float32")
    s3 = np.zeros((128,128)).astype("float32")

    q1 = torch.from_numpy(s1)
    q2 = torch.from_numpy(s2)
    q3 = torch.from_numpy(s3)

    # Result from numpy
    numpy_result = np.matmul(s1, np.transpose(s2))

    # Instantiate the `matmul` function by passing the parameters of shapes and datatype
    tvm_module = matmul(128, 128, 128, "float32")

    # Run the operator
    tvm_module(q1, q2, q3)

    tvm.testing.assert_allclose(q3.numpy(), numpy_result, atol=1e-5, rtol=1e-5)
    
test_tvmscript_torch_matmul()