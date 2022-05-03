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
#include <torch/custom_class.h>
#include <torch/script.h>

namespace tvm {
namespace contrib {

using namespace at;

struct ATenDLMTensor {
  Tensor handle;
  DLManagedTensor tensor;
};

void deleter(DLManagedTensor* arg) {
  delete static_cast<ATenDLMTensor*>(arg->manager_ctx);
}

DLDevice GetDLDevice(const at::Tensor& tensor, const int64_t& device_id) {
  DLDevice ctx;
  ctx.device_id = device_id;
  switch (tensor.device().type()) {
    case DeviceType::CPU:
      ctx.device_type = DLDeviceType::kDLCPU;
      break;
    case DeviceType::CUDA:
#ifdef USE_ROCM
      // ROCM, if enabled will look like cuda to PyTorch
      // while everyone else should see HIP
      ctx.device_type = DLDeviceType::kDLROCM;
#else
      ctx.device_type = DLDeviceType::kDLCUDA;
#endif
      break;
    case at::DeviceType::OPENCL:
      ctx.device_type = DLDeviceType::kDLOpenCL;
      break;
    case DeviceType::HIP:
      ctx.device_type = DLDeviceType::kDLROCM;
      break;
    default:
      TORCH_CHECK(false, "Cannot pack tensors on " + tensor.device().str());
  }
  return ctx;
}

DLDataType GetDLDataType(const at::Tensor& t) {
  DLDataType dtype;
  dtype.lanes = 1;
  dtype.bits = t.element_size() * 8;
  switch (t.scalar_type()) {
    case ScalarType::Byte:
      dtype.code = DLDataTypeCode::kDLUInt;
      break;
    case ScalarType::Char:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    // NOLINTNEXTLINE(bugprone-branch-clone)
    case ScalarType::Double:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    case ScalarType::Float:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    // NOLINTNEXTLINE(bugprone-branch-clone)
    case ScalarType::Int:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case ScalarType::Long:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case ScalarType::Short:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case ScalarType::Half:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    case ScalarType::Bool:
      TORCH_CHECK(false, "Bool type is not supported by dlpack");
      break;
    case ScalarType::ComplexHalf:
      dtype.code = DLDataTypeCode::kDLComplex;
      break;
    case ScalarType::ComplexFloat:
      dtype.code = DLDataTypeCode::kDLComplex;
      break;
    case ScalarType::ComplexDouble:
      dtype.code = DLDataTypeCode::kDLComplex;
      break;
    case ScalarType::BFloat16:
      dtype.code = DLDataTypeCode::kDLBfloat;
      break;
    case ScalarType::QInt8:
    case ScalarType::QUInt8:
    case ScalarType::QInt32:
    case ScalarType::QUInt4x2:
      TORCH_CHECK(false, "QUInt/QInt types are not supported by dlpack");
      break;
    case ScalarType::Undefined:
      TORCH_CHECK(false, "Undefined is not a valid ScalarType");
    case ScalarType::NumOptions:
      TORCH_CHECK(false, "NumOptions is not a valid ScalarType");
  }
  return dtype;
}

/*!
* \brief Convert PyTorch tensor to DLPack tensor.
*
* \param src The PyTorch tensor datastructure
*/
DLManagedTensor* ToDLPack(const at::Tensor& src) {
    ATenDLMTensor* atDLMTensor(new ATenDLMTensor);
    atDLMTensor->handle = src;
    atDLMTensor->tensor.manager_ctx = atDLMTensor;
    atDLMTensor->tensor.deleter = &deleter;
    atDLMTensor->tensor.dl_tensor.data = src.data_ptr();

    int64_t device_id = 0;
    if (src.is_cuda()) {
      device_id = src.get_device();
    }

    atDLMTensor->tensor.dl_tensor.device = GetDLDevice(src, device_id);
    atDLMTensor->tensor.dl_tensor.ndim = src.dim();
    atDLMTensor->tensor.dl_tensor.dtype = GetDLDataType(src);
    atDLMTensor->tensor.dl_tensor.shape =
      const_cast<int64_t*>(src.sizes().data());
    atDLMTensor->tensor.dl_tensor.strides =
        const_cast<int64_t*>(src.strides().data());
    atDLMTensor->tensor.dl_tensor.byte_offset = 0;

    return &(atDLMTensor->tensor);

}


}
}