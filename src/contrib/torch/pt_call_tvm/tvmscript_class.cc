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
#include <tvm/runtime/registry.h>
#include <tvm/runtime/module.h>

#include <map>
#include <string>
#include <vector>

#include "../utils.h"

namespace tvm {
namespace contrib {

// NOTE : this struct should be defined before TVMScriptRuntimeClass
struct ThreadLocalStore {
  tvm::runtime::Module mod;
  static ThreadLocalStore* ThreadLocal() {
    thread_local ThreadLocalStore tls;
    return &tls;
  }
};

class TVMScriptRuntimeClass : public torch::jit::CustomClassHolder {
 public:
  TVMScriptRuntimeClass() {
    std::cout<<"TVMScriptRuntimeClass initializaion"<<std::endl;
    mod_ = ThreadLocalStore::ThreadLocal() -> mod;
  }

  c10::List<at::Tensor> forward(const c10::List<at::Tensor>& inputs) {

    const auto inputs_len = inputs.size();

    std::vector<DLTensor> args(inputs_len);
    std::vector<tvm::runtime::NDArray> args_arr(inputs_len);

    for (int i = 0; i < inputs_len; ++i) {
      tvm::contrib::pytorch::TensorAsBuf input_buf(inputs[i]);
      input_buf.CopyFromOrigin();
      input_buf.MakeDLTensor(&args[i]);
      args_arr[i] =
          tvm::runtime::NDArray::FromDLPack(new DLManagedTensor({args[i], nullptr, nullptr}));
    }

    std::vector<TVMValue> tvm_values(inputs_len + 1);
    std::vector<int> tvm_type_codes(inputs_len + 1);
    tvm::runtime::TVMArgsSetter setter(tvm_values.data(), tvm_type_codes.data());
    setter(0, "main");
    for (int k = 0; k < inputs_len; ++k) {
      setter(k + 1, args_arr[k]);
    }

    auto set_input = mod_.GetFunction("set_input", false);
    auto invoke = mod_.GetFunction("invoke", false);

    set_input.CallPacked(
        tvm::runtime::TVMArgs(tvm_values.data(), tvm_type_codes.data(), inputs_len + 1), nullptr);

    tvm::runtime::TVMRetValue ret = invoke("main");

    // we assume only 1 outpus
    std::vector<tvm::runtime::NDArray> output_arrs(1);
    auto output_mismatch_msg = [](int actual, int expected) {
      std::stringstream ss;
      ss << "num_outputs not equal, actual:[" << actual << "] != expected:[" << expected << "]";
      return ss.str();
    };

    std::vector<DLTensor> output_args(1);
    c10::List<at::Tensor> outputs;
    outputs.reserve(1);

    for (int i = 0; i < 1; ++i) {
      const auto& output_arr = output_arrs[i];
      std::vector<int64_t> output_shape(output_arr->shape, output_arr->shape + output_arr->ndim);

      torch::ScalarType output_dtype = torch::ScalarType::Undefined;
      // CHECK(GetTorchDtype(output_arr.DataType(), &output_dtype));

      // CHECK(device_type_ == kDLCPU || device_type_ == kDLCUDA);
      const c10::DeviceType pt_device_type = torch::kCPU;
      const auto options =
          torch::TensorOptions().dtype(output_dtype).device(pt_device_type, 0);

      outputs.emplace_back(torch::empty(output_shape, options));
      tvm::contrib::pytorch::TensorAsBuf output_buf(outputs[i]);
      output_buf.MakeDLTensor(&output_args[i]);
      output_arr.CopyTo(&output_args[i]);
      output_buf.CopyToOrigin();
    }
    return outputs;

  }


 private:

  tvm::runtime::Module mod_;

};


TVM_REGISTER_GLOBAL("tvmtorch.save_runtime_mod").set_body_typed([](tvm::runtime::Module mod) {
  ThreadLocalStore::ThreadLocal()->mod = mod;
});

TORCH_LIBRARY(tvm_torch, m) {
  m.class_<TVMScriptRuntimeClass>("TVMScriptRuntime")
  .def(torch::init<>())
  .def("forward", &TVMScriptRuntimeClass::forward);
}

}
}