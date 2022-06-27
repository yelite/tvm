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
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>

#include "ATen/DLConvertor.h"
#include "ATen/Tensor.h"
#include "torch/csrc/api/include/torch/types.h"
#include "tvm/runtime/data_type.h"
#include "tvm/runtime/device_api.h"
#include "tvm/runtime/ndarray.h"

namespace tvm {
namespace contrib {

struct DLPackTensorExt {
  DLManagedTensor* dl_managed_tensor;
  bool is_bool;
};

DLPackTensorExt toDlPackExt(const at::Tensor& src) {
  if (!src.is_contiguous()) {
    return toDlPackExt(src.contiguous());
  }

  if (src.dtype().isScalarType(torch::kBool)) {
    auto temp = src.toType(torch::kUInt8);
    return {.dl_managed_tensor = at::toDLPack(temp), .is_bool = false};
  }

  return {.dl_managed_tensor = at::toDLPack(src), .is_bool = false};
}

DLPackTensorExt toDlPackExt(const tvm::runtime::NDArray& src) {
  if (src.DataType().is_bool()) {
    auto temp = tvm::runtime::NDArray::Empty(src.Shape(), DataType::UInt(8), src->device);
    temp.CopyFrom(src);
    return {.dl_managed_tensor = temp.ToDLPack(), .is_bool = true};
  }

  return {.dl_managed_tensor = src.ToDLPack(), .is_bool = false};
}

at::Tensor TensorFromDlPackExt(DLPackTensorExt dlpack_ext) {
  at::Tensor tensor = at::fromDLPack(dlpack_ext.dl_managed_tensor);
  if (dlpack_ext.is_bool) {
    return tensor.toType(torch::kBool);
  }
  return tensor;
}

tvm::runtime::NDArray NDArrayFromDlPackExt(DLPackTensorExt dlpack_ext) {
  using tvm::runtime::NDArray;

  NDArray array;
  DLTensor& dl_tensor = dlpack_ext.dl_managed_tensor->dl_tensor;
  bool is_aligned =
      (reinterpret_cast<size_t>(static_cast<char*>(dl_tensor.data) + dl_tensor.byte_offset) %
           tvm::runtime::kAllocAlignment ==
       0);
  if (is_aligned) {
    // Zero-copy if data pointer is aligned
    array = NDArray::FromDLPack(dlpack_ext.dl_managed_tensor);
  } else {
    // Copy if data pointer isn't aligned to the kAllocAlignment of TVM
    array = NDArray::NewFromDLTensor(&dl_tensor, dl_tensor.device);
    dlpack_ext.dl_managed_tensor->deleter(dlpack_ext.dl_managed_tensor);
  }
  if (dlpack_ext.is_bool) {
    auto result = tvm::runtime::NDArray::Empty(array.Shape(), DataType::Bool(), array->device);
    result.CopyFrom(array);
    return result;
  }
  return array;
}

tvm::runtime::NDArray toTvmNDArray(const at::Tensor& src) {
  DLPackTensorExt dlpack_ext = toDlPackExt(src);
  tvm::runtime::NDArray result = NDArrayFromDlPackExt(dlpack_ext);
  // dlpack_ext.dlpack_tensor->deleter(dlpack_ext.dlpack_tensor);
  return result;
}

at::Tensor toTorchTensor(const tvm::runtime::NDArray& src) {
  DLPackTensorExt dlpack_ext = toDlPackExt(src);
  at::Tensor result = TensorFromDlPackExt(dlpack_ext);
  // dlpack_ext.dlpack_tensor->deleter(dlpack_ext.dlpack_tensor);
  return result;
}

}  // namespace contrib
}  // namespace tvm
