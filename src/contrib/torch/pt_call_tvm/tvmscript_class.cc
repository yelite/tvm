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
#include <ATen/DLConvertor.h>
#include <dlpack/dlpack.h>
#include <torch/custom_class.h>
#include <torch/script.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/codegen.h>
#include <tvm/target/target.h>

#include <map>
#include <string>
#include <vector>

namespace tvm {
namespace contrib {

struct ThreadLocalStore {
  tvm::runtime::Module mod;
  static ThreadLocalStore* ThreadLocal() {
    thread_local ThreadLocalStore tls;
    return &tls;
  }
};

class TVMScriptRuntimeClass : public torch::jit::CustomClassHolder {
 public:
  TVMScriptRuntimeClass() { mod_ = ThreadLocalStore::ThreadLocal()->mod; }

  void forward(const c10::List<at::Tensor>& inputs) {
    int input_length = inputs.size();

    std::vector<DLManagedTensor*> tensors;

    for (int i = 0; i < input_length; ++i) tensors.push_back(toDLPack(inputs[i]));

    tvm::runtime::PackedFunc run = mod_.GetFunction("__tvm_main__");

    std::vector<TVMValue> tvm_values(input_length);
    std::vector<int> tvm_type_codes(input_length);
    tvm::runtime::TVMArgsSetter setter(tvm_values.data(), tvm_type_codes.data());
    for (int k = 0; k < input_length; ++k) {
      setter(k, &tensors[k]->dl_tensor);
    }

    run.CallPacked(tvm::runtime::TVMArgs(tvm_values.data(), tvm_type_codes.data(), input_length),
                   nullptr);

    for (int k = 0; k < input_length; ++k) {
      tensors[k]->deleter(tensors[k]);
    }
  }

 private:
  tvm::runtime::Module mod_;
};

class RelayRuntimeClass : public torch::jit::CustomClassHolder {
 public:
  RelayRuntimeClass() { mod_ = ThreadLocalStore::ThreadLocal()->mod; }

  c10::List<at::Tensor> forward(const c10::List<at::Tensor>& inputs) {
    int input_length = inputs.size();

    std::vector<DLManagedTensor*> tensors;

    for (int i = 0; i < input_length; ++i) tensors.push_back(toDLPack(inputs[i]));

    tvm::runtime::PackedFunc run = mod_.GetFunction("run");
    tvm::runtime::PackedFunc set_input = mod_.GetFunction("set_input");
    tvm::runtime::PackedFunc get_output = mod_.GetFunction("get_output");
    tvm::runtime::PackedFunc get_num_outputs = mod_.GetFunction("get_num_outputs");

    for (int k = 0; k < input_length; ++k) {
      std::vector<TVMValue> tvm_values(input_length);
      std::vector<int> tvm_type_codes(input_length);
      tvm::runtime::TVMArgsSetter setter(tvm_values.data(), tvm_type_codes.data());
      setter(0, k);
      setter(1, &tensors[k]->dl_tensor);
      set_input.CallPacked(tvm::runtime::TVMArgs(tvm_values.data(), tvm_type_codes.data(), 2),
                           nullptr);
    }

    run.CallPacked(tvm::runtime::TVMArgs(NULL, NULL, 0), nullptr);

    std::vector<TVMValue> tvm_values(input_length);
    std::vector<int> tvm_type_codes(input_length);
    tvm::runtime::TVMArgsSetter setter(tvm_values.data(), tvm_type_codes.data());
    setter(0, 0);
    tvm::runtime::TVMRetValue ret_output;
    tvm::runtime::TVMRetValue ret_num_outputs;

    get_output.CallPacked(tvm::runtime::TVMArgs(tvm_values.data(), tvm_type_codes.data(), 1),
                          &ret_output);

    get_num_outputs.CallPacked(tvm::runtime::TVMArgs(NULL, NULL, 0), &ret_num_outputs);

    // TODO: need to check if output_length == 1
    int64_t output_length = ret_num_outputs;

    c10::List<at::Tensor> outputs;
    outputs.reserve(output_length);

    for (int k = 0; k < output_length; ++k) {
      // TODO: need to check if we should use ret_output or get_output(0)
      tvm::runtime::NDArray results = get_output(k);
      at::Tensor atTensor = at::fromDLPack(results.ToDLPack());
      outputs.emplace_back(atTensor);
    }

    for (int k = 0; k < input_length; ++k) {
      tensors[k]->deleter(tensors[k]);
    }
    return outputs;
  }

  void SaveToFileCPU(const std::string& file_name, const std::string& format) {
    LLVMModuleNode x = mod_;
    x->SaveToFile(file_name, format);
  }

  void SaveToFileCUDA(const std::string& file_name, const std::string& format) {
    CUDAModuleNode x = mod_;
    x->SaveToFile(file_name, format);
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

TORCH_LIBRARY(tvm_tuning, m) {
  m.class_<RelayRuntimeClass>("RelayRuntime")
      .def(torch::init<>())
      .def("save_cpu", &RelayRuntimeClass::SaveToFileCPU)
      .def("save_cuda", &RelayRuntimeClass::SaveToFileCUDA)
      .def("forward", &RelayRuntimeClass::forward);
}

}  // namespace contrib
}  // namespace tvm