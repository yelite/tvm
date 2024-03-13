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

/*!
 * \file Externally defined CUDA kernels for use in TVM runtime
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <tvm/runtime/registry.h>

#include "../../cuda/cuda_common.h"

namespace tvm {
namespace contrib {

using namespace runtime;

template <typename T>
__device__ T device_max(T a, T b) {
  return max(a, b);
}

template <>
__device__ __half device_max(__half a, __half b) {
  return __hmax(a, b);
}

template <typename T>
__device__ T device_abs(T a) {
  return abs(a);
}

template <>
__device__ __half device_abs(__half a) {
  return __habs(a);
}

template <typename T>
__inline__ __device__ T warp_reduce_max(T val) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    val = device_max(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

// Single block reduce, assumes size % 1024 == 0
template <typename T>
__global__ void max_reduce_kernel_single_block(T* input, T* output, int size) {
  __shared__ T shared[32];

  int tid = threadIdx.x;
  T max_val = std::numeric_limits<T>::lowest();

  // Step 1: Each thread reduces across the elements it owns
  for (int i = tid; i < size; i += blockDim.x) {
    // use __hmax for float16
    max_val = device_max(max_val, device_abs(input[i]));
  }

  // Step 2: Perform reduce across warps
  max_val = warp_reduce_max(max_val);

  // Step 3: Write the reduced value from each warp to shared memory
  if (tid % warpSize == 0) {
    shared[tid / warpSize] = max_val;
  }
  __syncthreads();

  // Step 4: Perform a final reduction in the first warp across shared values
  if (tid < warpSize) {
    max_val = shared[tid];
    max_val = warp_reduce_max(max_val);
    if (tid == 0) {
      *output = max_val;
    }
  }
}

template __global__ void max_reduce_kernel_single_block<float>(float* input, float* output,
                                                               int size);
template __global__ void max_reduce_kernel_single_block<__half>(__half* input, __half* output,
                                                                int size);
template <typename T>
void LaunchMaxReduceKernelSingleBlock(DLTensor* input, DLTensor* output, int size) {
  T* input_ptr = static_cast<T*>(input->data);
  T* output_ptr = static_cast<T*>(output->data);

  int blocks = 1;
  int threads = 1024;
  max_reduce_kernel_single_block<T><<<blocks, threads>>>(input_ptr, output_ptr, size);
}

TVM_REGISTER_GLOBAL("tvm.contrib.cuda.reduce_max_abs").set_body([](TVMArgs args, TVMRetValue* ret) {
  DLTensor* input = args[0];
  DLTensor* output = args[1];

  int size = 1;
  for (int i = 0; i < input->ndim; ++i) {
    size *= input->shape[i];
  }

  CHECK_EQ(size % 1024, 0) << "tvm.contrib.cuda.reduce_max_abs currently only supports reducing "
                              "tensors that are an even factor of 1024 elements";

  auto dtype = DLDataType2String(input->dtype);

  if (dtype == "float32") {
    LaunchMaxReduceKernelSingleBlock<float>(input, output, size);
  } else if (dtype == "float16") {
    LaunchMaxReduceKernelSingleBlock<__half>(input, output, size);
  } else {
    LOG(FATAL) << "Unsupported input dtype: " << dtype;
  }
});

}  // namespace contrib
}  // namespace tvm
