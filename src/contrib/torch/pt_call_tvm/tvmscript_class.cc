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
#include "../DLconverter.h"

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
  TVMScriptRuntimeClass() {

    mod_ = ThreadLocalStore::ThreadLocal() -> mod;
  }

  void forward(const c10::List<at::Tensor>& inputs) {


    int input_length = inputs.size();

    std::vector<DLManagedTensor*> tensors;

    for (int i = 0; i < input_length; ++i) tensors.push_back(ToDLPack(inputs[i]));

    tvm::runtime::PackedFunc rt_func = mod_.GetFunction("__tvm_main__");

    std::vector<TVMValue> tvm_values(input_length);
    std::vector<int> tvm_type_codes(input_length);
    tvm::runtime::TVMArgsSetter setter(tvm_values.data(), tvm_type_codes.data());
    for (int k = 0; k < input_length; ++k) {
      setter(k, &tensors[k]->dl_tensor);
    }

    rt_func.CallPacked(
        tvm::runtime::TVMArgs(tvm_values.data(), tvm_type_codes.data(), input_length), nullptr);
    
    for (int k = 0; k < input_length; ++k) {
      LOG(INFO) << "del: " << static_cast<ATenDLMTensor*>(tensors[k]->manager_ctx);
      tensors[k]->deleter;
    }
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