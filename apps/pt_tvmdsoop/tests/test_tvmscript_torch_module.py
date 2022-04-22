#!/usr/bin/env python

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
"""Test script for tvm torch module"""
import tvm
import torch
from tvm.contrib.torch import as_torch
from tvm.script import tir as T
import numpy as np
import torch.nn
import tvm.testing

@as_torch
@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])

    for i, j, k in T.grid(128, 128, 128):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

@T.prim_func
def matmul_no_deco(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])

    for i, j, k in T.grid(128, 128, 128):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

@as_torch
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle):
        # We exchange data between function by handles, which are similar to pointer.
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # Create buffer from handles.
        A = T.match_buffer(a, (8,), dtype="float32")
        B = T.match_buffer(b, (8,), dtype="float32")
        for i in range(8):
            # A block is an abstraction for computation.
            with T.block("B"):
                # Define a spatial block iterator and bind it to value i.
                vi = T.axis.spatial(8, i)
                B[vi] = A[vi] + 1.0


class MinuesOnes(torch.nn.Module):
    def __init__(self):
        super(MinuesOnes, self).__init__()
        self.mat = matmul

    def forward(self, input):
        ret = self.mat.forward(input) - 1
        return ret

def test_tvmscript_torch_matmul():
    s1 = np.ones((128,128)).astype("float32")
    s2 = np.ones((128,128)).astype("float32")
    s3 = np.zeros((128,128)).astype("float32")
    s1[0,0] = 0
    s2[4,4] = 0

    q1 = torch.from_numpy(s1)
    q2 = torch.from_numpy(s2)
    q3 = torch.from_numpy(s3)

    numpy_result = np.matmul(s1,s2)

    tvm_module = matmul

    res = tvm_module([q1, q2, q3])

    tvm.testing.assert_allclose(res.numpy(), numpy_result, atol=1e-5, rtol=1e-5)

def test_tvmscript_torch_decorator():
    s1 = np.arange(8).astype("float32")
    s2 = np.zeros((8,)).astype("float32")

    q1 = torch.from_numpy(s1)
    q2 = torch.from_numpy(s2)

    numpy_result = s1 + 1

    tvm_module = MyModule

    res = tvm_module([q1, q2])

    tvm.testing.assert_allclose(res.numpy(), numpy_result, atol=1e-5, rtol=1e-5)

def test_torch_with_tvmscirpt():
    s1 = np.ones((128,128)).astype("float32")
    s2 = np.ones((128,128)).astype("float32")
    s3 = np.zeros((128,128)).astype("float32")
    s1[0,0] = -10
    s2[4,4] = -20

    q1 = torch.from_numpy(s1)
    q2 = torch.from_numpy(s2)
    q3 = torch.from_numpy(s3)

    numpy_result = np.matmul(s1,s2) - 1

    tvm_module = MinuesOnes()

    res = tvm_module(input = [q1, q2, q3])

    tvm.testing.assert_allclose(res.numpy(), numpy_result, atol=1e-5, rtol=1e-5)


def test_tvmscript_torch_matmul_with_cxx():
    s1 = np.ones((128,128)).astype("float32")
    s2 = np.ones((128,128)).astype("float32")
    s3 = np.zeros((128,128)).astype("float32")
    s1[0,0] = 0
    s2[4,4] = 0

    q1 = torch.from_numpy(s1)
    q2 = torch.from_numpy(s2)
    q3 = torch.from_numpy(s3)

    numpy_result = np.matmul(s1,s2)

    tvm_module = matmul_no_deco

    # res = tvm_module([q1, q2, q3])

    # tvm.testing.assert_allclose(res.numpy(), numpy_result, atol=1e-5, rtol=1e-5)

if __name__ == "__main__":
    test_tvmscript_torch_matmul()
    test_tvmscript_torch_decorator()
    test_torch_with_tvmscirpt()
    test_tvmscript_torch_matmul_with_cxx()