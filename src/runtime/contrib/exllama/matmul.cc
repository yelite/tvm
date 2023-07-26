/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include "../../../3rdparty/exllama-cuda-kernels/exllama/cuda_func/q4_matmul.cuh"

namespace tvm {
namespace runtime {

TVM_REGISTER_GLOBAL("exllama.q4_matmul")
    .set_body_typed([](NDArray input, NDArray weight, NDArray scale, NDArray out) {
      ICHECK(input.IsContiguous());
      ICHECK(weight.IsContiguous());
      ICHECK(scale.IsContiguous());
      ICHECK(out.IsContiguous());

      ICHECK(input.DataType().is_float16());
      ICHECK(weight.DataType().is_int() && weight.DataType().bits() == 8);
      ICHECK(scale.DataType().is_float16());
      ICHECK(out.DataType().is_float16());
    });

}  // namespace runtime
}  // namespace tvm
